from DRAPE.checkpoint import (
    ROUTER_CENTERS_NAME,
    ROUTER_COUNTS_NAME,
    ROUTER_CONFIG_NAME,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
import os
import json

IMAGE_TAGS = ("<image>", "<im_start>", "<im_end>", "<im_patch>")


class NullSpaceProjection:
    """
    Null-space projection for continual learning anti-forgetting.

    After each task finishes training, we collect the feature representations
    (activations) that pass through the shared projection layer (mm_projector).
    We compute the covariance of these features, find the principal subspace via SVD,
    and build a projection matrix onto its null space.

    When training on the next task, we project the gradients of mm_projector
    onto this null space so that updates don't interfere with previously
    learned representations.
    """

    def __init__(self, threshold=0.99):
        self.cov_matrix = None
        self.num_samples = 0
        self.threshold = threshold
        self.projection_matrix = None

    def update(self, features):
        """
        Accumulate the covariance matrix with new features.

        Args:
            features: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        """
        if features.dim() == 3:
            features = features.reshape(-1, features.size(-1))

        features = features.detach().float()

        if self.cov_matrix is None:
            self.cov_matrix = torch.zeros(
                (features.size(-1), features.size(-1)),
                device=features.device,
                dtype=torch.float32,
            )

        cov = torch.matmul(features.t(), features)
        self.cov_matrix += cov
        self.num_samples += features.size(0)

    def compute_projection_matrix(self):
        """
        Compute the projection matrix onto the null space of the covariance matrix.

        Returns:
            projection_matrix: (D, D) matrix that projects gradients onto the null space
        """
        if self.cov_matrix is None:
            return None

        cov_norm = self.cov_matrix / self.num_samples
        U, S, V = torch.linalg.svd(cov_norm)

        # Determine rank by explained variance ratio
        explained_variance = torch.cumsum(S, dim=0) / torch.sum(S)
        rank = torch.searchsorted(explained_variance, self.threshold).item() + 1
        rank = min(rank, S.size(0) - 1)  # Ensure at least 1-dim null space

        # Null space basis: columns of V^T beyond the rank
        null_space = V[rank:].t()  # (D, D - rank)
        self.projection_matrix = torch.matmul(null_space, null_space.t())
        return self.projection_matrix

    def project_gradients(self, gradients):
        """
        Project gradients onto the null space.

        Args:
            gradients: (out_features, in_features) gradient tensor

        Returns:
            Projected gradients of the same shape
        """
        if self.projection_matrix is None:
            return gradients

        proj = self.projection_matrix.to(device=gradients.device, dtype=gradients.dtype)

        # For weight gradients: project along the input dimension
        # grad shape: (out_features, in_features), proj shape: (in_features, in_features)
        if gradients.dim() == 2:
            return torch.matmul(gradients, proj)
        elif gradients.dim() == 1:
            if gradients.size(0) == proj.size(0):
                return torch.matmul(gradients.unsqueeze(0), proj).squeeze(0)
            else:
                return gradients
        else:
            return gradients

    def merge(self, other):
        """
        Merge another NullSpaceProjection's accumulated covariance.
        Used when combining projections from multiple previous tasks.
        """
        if other.cov_matrix is None:
            return
        if self.cov_matrix is None:
            self.cov_matrix = other.cov_matrix.clone()
            self.num_samples = other.num_samples
        else:
            self.cov_matrix += other.cov_matrix.to(self.cov_matrix.device)
            self.num_samples += other.num_samples

    def save(self, path):
        """Save the null-space projection state."""
        state = {
            "cov_matrix": self.cov_matrix,
            "num_samples": self.num_samples,
            "threshold": self.threshold,
            "projection_matrix": self.projection_matrix,
        }
        torch.save(state, path)

    def load(self, path, device="cpu"):
        """Load the null-space projection state."""
        state = torch.load(path, map_location=device)
        self.cov_matrix = state["cov_matrix"]
        self.num_samples = state["num_samples"]
        self.threshold = state.get("threshold", self.threshold)
        self.projection_matrix = state.get("projection_matrix", None)


class NullSpaceGradientHook:
    """
    A gradient hook that projects mm_projector gradients onto the null space
    of previously learned feature representations.

    Usage:
        hook = NullSpaceGradientHook(projection_matrices)
        hook.register(model)
        # ... training ...
        hook.remove()
    """

    def __init__(self, projection_matrices):
        """
        Args:
            projection_matrices: dict mapping parameter names (or layer indices)
                to projection matrices. Each matrix is (D, D).
                For mm_projector, typically we have separate projections for
                each linear layer in the projector.
        """
        self.projection_matrices = projection_matrices
        self.hooks = []

    def _make_hook(self, proj_matrix):
        """Create a gradient hook function for a specific projection matrix."""
        def hook_fn(grad):
            if proj_matrix is None:
                return grad
            proj = proj_matrix.to(device=grad.device, dtype=grad.dtype)
            if grad.dim() == 2:
                return torch.matmul(grad, proj)
            elif grad.dim() == 1:
                # Bias gradient: project in the output dimension
                return grad
            return grad
        return hook_fn

    def register(self, model):
        """Register gradient hooks on mm_projector parameters."""
        for name, param in model.named_parameters():
            if "mm_projector" in name and param.requires_grad:
                # Find the matching projection matrix
                for key, proj_matrix in self.projection_matrices.items():
                    if key in name:
                        handle = param.register_hook(self._make_hook(proj_matrix))
                        self.hooks.append(handle)
                        break

    def remove(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


class CLIPRouter(nn.Module):
    """
    CLIP-based task router for continual learning.

    Supports:
    1) Order-independent single-center routing (K=1) via accumulated feature sums.
    2) Multi-center routing (K>1) where each task owns multiple prototypes.
    """

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        num_centers_per_task=1,
        kmeans_iters=20,
        shared_vision_tower=None,
        k1_center_mode="mean",
        k1_learn_epochs=4,
        k1_learn_batch_size=1024,
        k1_learn_lr=5e-3,
        k1_temperature=0.07,
        k1_sep_margin=0.02,
        k1_lambda_sep=0.5,
        k1_lambda_orth=0.1,
    ):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.num_centers_per_task = max(1, int(num_centers_per_task))
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.k1_center_mode = str(k1_center_mode).lower()
        self.k1_learn_epochs = max(1, int(k1_learn_epochs))
        self.k1_learn_batch_size = max(1, int(k1_learn_batch_size))
        self.k1_learn_lr = float(k1_learn_lr)
        self.k1_temperature = float(k1_temperature)
        self.k1_sep_margin = float(k1_sep_margin)
        self.k1_lambda_sep = float(k1_lambda_sep)
        self.k1_lambda_orth = float(k1_lambda_orth)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
        # Reuse LLaVA vision tower when available to avoid loading a second CLIP vision backbone.
        self._shared_refs = {"vision_tower": shared_vision_tower}
        self.vision_encoder = None
        self.visual_projection = None
        if shared_vision_tower is None:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
            self.visual_projection = self.vision_encoder.visual_projection
        else:
            if hasattr(shared_vision_tower, "image_processor"):
                self.image_processor = shared_vision_tower.image_processor
            # Only keep the lightweight visual projection head; backbone is reused from LLaVA.
            _tmp = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
            in_dim = int(_tmp.visual_projection.in_features)
            out_dim = int(_tmp.visual_projection.out_features)
            self.visual_projection = nn.Linear(in_dim, out_dim, bias=False)
            self.visual_projection.load_state_dict(_tmp.visual_projection.state_dict(), strict=True)
            del _tmp
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if self.visual_projection is not None:
            for param in self.visual_projection.parameters():
                param.requires_grad = False
        self.text_encoder.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()

        # Task centers: {task_name: (K, embed_dim)} normalized embeddings
        self.task_centers = nn.ParameterDict()
        # Track number of samples accumulated per task.
        self._task_sample_counts = {}
        # For strict order-independent K=1 updates.
        self._task_feature_sums = {}
        # For K>1 center recomputation.
        self._task_feature_buffers = {}

    def _collect_old_task_keys(self, task_name, device, dtype):
        old_keys = []
        for name, centers in self.task_centers.items():
            if name == task_name:
                continue
            c = centers.detach()
            if c.dim() == 2 and c.size(0) > 1:
                c = F.normalize(c.mean(dim=0, keepdim=True), dim=-1)
            elif c.dim() == 1:
                c = c.unsqueeze(0)
            old_keys.append(c.to(device=device, dtype=dtype))
        if not old_keys:
            return torch.empty((0, 0), device=device, dtype=dtype)
        return torch.cat(old_keys, dim=0)

    def _learn_single_center(self, features, old_keys):
        """Learn one task center with discriminative objective (key_router-style)."""
        features = F.normalize(features.float(), dim=-1)
        key = nn.Parameter(F.normalize(features.mean(dim=0, keepdim=True), dim=-1).clone())
        opt = torch.optim.Adam([key], lr=self.k1_learn_lr)
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.k1_learn_batch_size,
            shuffle=True,
            drop_last=False,
        )

        pbar = tqdm(range(self.k1_learn_epochs), desc="Learn key", ncols=100)
        for _ in pbar:
            for (z,) in loader:
                z = F.normalize(z, dim=-1)
                k = F.normalize(key, dim=-1)
                sim_new = torch.matmul(z, k.t()).squeeze(-1)  # [B]
                if old_keys.numel() > 0:
                    sim_old = torch.matmul(z, old_keys.t())  # [B, T_old]
                    logits = torch.cat([sim_new.unsqueeze(1), sim_old], dim=1) / self.k1_temperature
                    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                    loss_cls = F.cross_entropy(logits, targets)
                    loss_sep = torch.relu(sim_old.max(dim=1).values - sim_new + self.k1_sep_margin).mean()
                    loss_orth = torch.matmul(k, old_keys.t()).pow(2).mean()
                    loss = loss_cls + self.k1_lambda_sep * loss_sep + self.k1_lambda_orth * loss_orth
                else:
                    loss = (1.0 - sim_new).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return F.normalize(key.detach(), dim=-1)

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        for tag in IMAGE_TAGS:
            text = text.replace(tag, " ")
        text = text.strip()

        # Drop assistant side if a full chat prompt is provided.
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[0]

        # Keep only the last user turn if chat template markers exist.
        user_markers = ["### Human:", "USER:", "Human:", "User:"]
        last_pos = -1
        last_marker = None
        for m in user_markers:
            p = text.rfind(m)
            if p > last_pos:
                last_pos = p
                last_marker = m
        if last_marker is not None and last_pos >= 0:
            text = text[last_pos + len(last_marker) :]

        # Remove common system preamble lines.
        filtered_lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            lower = s.lower()
            if lower.startswith("a chat between"):
                continue
            if lower.startswith("system:") or lower.startswith("### system:"):
                continue
            filtered_lines.append(s)

        text = " ".join(filtered_lines) if filtered_lines else text
        return " ".join(text.split()).strip()

    def encode_text(self, texts, device):
        """
        Encode a list of text strings using CLIP's text encoder.

        Args:
            texts: list of strings
            device: torch device

        Returns:
            embeddings: (N, embed_dim) L2-normalized embeddings
        """
        texts = [self._normalize_text(t) for t in texts]
        inputs = self.tokenizer(
            texts, padding=True, return_tensors="pt", truncation=True, max_length=77
        ).to(device)
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.text_embeds  # (N, embed_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_image(self, images, device, dtype=None):
        """Encode image batch into CLIP image embedding space."""
        if images is None:
            return None

        shared_vision_tower = self._shared_refs.get("vision_tower", None)
        use_shared_backbone = shared_vision_tower is not None

        if torch.is_tensor(images):
            pixel_values = images.to(device=device)
            if pixel_values.dim() == 5:
                # [B, N, C, H, W] -> keep first frame/view
                pixel_values = pixel_values[:, 0]
            if use_shared_backbone and hasattr(shared_vision_tower, "config"):
                image_size = int(getattr(shared_vision_tower.config, "image_size", pixel_values.shape[-1]))
            else:
                image_size = int(getattr(self.vision_encoder.config, "image_size", pixel_values.shape[-1]))
            if pixel_values.shape[-1] != image_size or pixel_values.shape[-2] != image_size:
                pixel_values = F.interpolate(
                    pixel_values.float(), size=(image_size, image_size), mode="bilinear", align_corners=False
                )
            if dtype is None:
                if use_shared_backbone:
                    base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                    dtype = next(base.parameters()).dtype
                else:
                    dtype = next(self.vision_encoder.parameters()).dtype
            pixel_values = pixel_values.to(dtype=dtype)

            if use_shared_backbone:
                base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                outputs = base(pixel_values=pixel_values, output_hidden_states=False)
                pooled = outputs.last_hidden_state[:, 0, :]
                proj = self.visual_projection
                pooled = pooled.to(device=proj.weight.device, dtype=proj.weight.dtype)
                emb = proj(pooled)
            else:
                outputs = self.vision_encoder(pixel_values=pixel_values)
                emb = outputs.image_embeds
            return F.normalize(emb, dim=-1)

        img_inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device=device)
        if dtype is None:
            if use_shared_backbone:
                base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                dtype = next(base.parameters()).dtype
            else:
                dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)
        if use_shared_backbone:
            base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
            outputs = base(pixel_values=pixel_values, output_hidden_states=False)
            pooled = outputs.last_hidden_state[:, 0, :]
            proj = self.visual_projection
            pooled = pooled.to(device=proj.weight.device, dtype=proj.weight.dtype)
            emb = proj(pooled)
        else:
            outputs = self.vision_encoder(pixel_values=pixel_values)
            emb = outputs.image_embeds
        return F.normalize(emb, dim=-1)

    def reset_task_center(self, task_name):
        """Remove one task center and its running statistics."""
        if task_name in self.task_centers:
            del self.task_centers[task_name]
        self._task_sample_counts.pop(task_name, None)
        self._task_feature_sums.pop(task_name, None)
        self._task_feature_buffers.pop(task_name, None)

    def update_task_center(self, task_name, texts, device, encode_batch_size=512):
        """
        Update or create the task center embedding.

        The task center is the L2-normalized mean of all text embeddings
        belonging to that task. For incremental updates, we use a running
        weighted average.

        Args:
            task_name: string identifier for the task
            texts: list of text descriptions for this task
            device: torch device
            encode_batch_size: encode chunk size used for CLIP text forward
        """
        if texts is None or len(texts) == 0:
            return

        with torch.no_grad():
            encode_batch_size = max(1, int(encode_batch_size))
            if len(texts) <= encode_batch_size:
                embeddings = self.encode_text(texts, device)
            else:
                vecs = []
                for start in range(0, len(texts), encode_batch_size):
                    chunk = texts[start : start + encode_batch_size]
                    emb = self.encode_text(chunk, device)
                    vecs.append(emb)
                embeddings = torch.cat(vecs, dim=0)
            if embeddings.numel() == 0:
                return
            if self.num_centers_per_task <= 1:
                if self.k1_center_mode == "learned":
                    old_keys = self._collect_old_task_keys(task_name, device=device, dtype=embeddings.dtype)
                    if old_keys.numel() > 0:
                        old_keys = old_keys.to(device=device, dtype=embeddings.dtype)
                    center = self._learn_single_center(embeddings, old_keys)
                else:
                    center = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)
                self.task_centers[task_name] = center
                return

            # K > 1: keep all seen features for this task and recompute centers.
            if task_name not in self._task_feature_buffers:
                self._task_feature_buffers[task_name] = embeddings
            else:
                self._task_feature_buffers[task_name] = torch.cat(
                    [self._task_feature_buffers[task_name], embeddings], dim=0
                )

            self._task_sample_counts[task_name] = int(self._task_feature_buffers[task_name].size(0))
            features = self._task_feature_buffers[task_name]
            centers = self._compute_kmeans_centers(
                features=features,
                num_centers=min(self.num_centers_per_task, features.size(0)),
                num_iters=self.kmeans_iters,
            )
            self.task_centers[task_name] = centers

    def update_task_center_from_embeddings(self, task_name, embeddings, device):
        """Update task center directly from precomputed embeddings [N, D]."""
        if embeddings is None:
            return
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
        if embeddings.numel() == 0:
            return
        embeddings = F.normalize(embeddings.float().to(device), dim=-1)
        if self.num_centers_per_task <= 1:
            if self.k1_center_mode == "learned":
                old_keys = self._collect_old_task_keys(task_name, device=device, dtype=embeddings.dtype)
                if old_keys.numel() > 0:
                    old_keys = old_keys.to(device=device, dtype=embeddings.dtype)
                center = self._learn_single_center(embeddings, old_keys)
            else:
                center = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)
            self.task_centers[task_name] = center
            self._task_sample_counts[task_name] = int(embeddings.size(0))
            return
        centers = self._compute_kmeans_centers(
            features=embeddings,
            num_centers=min(self.num_centers_per_task, embeddings.size(0)),
            num_iters=self.kmeans_iters,
        )
        self.task_centers[task_name] = centers
        self._task_sample_counts[task_name] = int(embeddings.size(0))

    def _compute_kmeans_centers(self, features, num_centers, num_iters):
        """
        Deterministic farthest-point init + k-means refinement.
        """
        features = F.normalize(features.float(), dim=-1)
        if features.size(0) <= num_centers:
            return features[:num_centers]

        mean_vec = features.mean(dim=0, keepdim=True)  # (1, D)
        dists = torch.sum((features - mean_vec) ** 2, dim=1)  # (N,)
        first_idx = torch.argmax(dists).item()
        centers = [features[first_idx:first_idx + 1]]

        for _ in range(1, num_centers):
            all_centers = torch.cat(centers, dim=0)  # (k, D)
            # Pick point farthest to the nearest existing center.
            sim = torch.matmul(features, all_centers.t())  # (N, k)
            min_dist = 1.0 - sim.max(dim=1).values
            next_idx = torch.argmax(min_dist).item()
            centers.append(features[next_idx:next_idx + 1])

        centers = torch.cat(centers, dim=0)  # (K, D)

        for _ in range(num_iters):
            sim = torch.matmul(features, centers.t())  # (N, K)
            assign = torch.argmax(sim, dim=1)  # (N,)
            new_centers = []
            for center_idx in range(centers.size(0)):
                mask = assign == center_idx
                if mask.any():
                    c = features[mask].mean(dim=0, keepdim=True)
                    c = F.normalize(c, dim=-1)
                else:
                    c = centers[center_idx:center_idx + 1]
                new_centers.append(c)
            new_centers = torch.cat(new_centers, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        return centers

    def forward(self, texts, device, top_k=1, images=None, image_embeddings=None):
        """
        Route each text sample to the most similar task.

        Args:
            texts: list of text strings
            device: torch device
            top_k: number of top tasks to return (default 1)

        Returns:
            best_tasks: list of task name strings (one per input text)
            similarities: (N, num_tasks) cosine similarity matrix
        """
        if len(self.task_centers) == 0:
            return None, None
        with torch.no_grad():
            text_embeddings = self.encode_text(texts, device)  # (N, Dt)
            sample_center = next(iter(self.task_centers.values()))
            center_dim = int(sample_center.shape[-1])
            text_dim = int(text_embeddings.shape[-1])

            if center_dim == text_dim:
                embeddings = text_embeddings
            elif center_dim == text_dim * 2:
                if image_embeddings is not None:
                    img_emb = image_embeddings.to(device=device, dtype=text_embeddings.dtype)
                    img_emb = F.normalize(img_emb, dim=-1)
                else:
                    image_dtype = None
                    if self.vision_encoder is not None:
                        image_dtype = next(self.vision_encoder.parameters()).dtype
                    else:
                        shared_vision_tower = self._shared_refs.get("vision_tower", None)
                        if shared_vision_tower is not None:
                            base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                            image_dtype = next(base.parameters()).dtype
                    img_emb = self.encode_image(
                        images=images,
                        device=device,
                        dtype=image_dtype,
                    )
                if img_emb is None:
                    img_emb = torch.zeros_like(text_embeddings)
                embeddings = torch.cat([text_embeddings, img_emb.to(text_embeddings.dtype)], dim=-1)
                embeddings = F.normalize(embeddings, dim=-1)
            else:
                # Fallback: align dimensions by truncating or zero-padding text embeddings.
                if center_dim < text_dim:
                    embeddings = text_embeddings[:, :center_dim]
                else:
                    pad = torch.zeros(
                        (text_embeddings.size(0), center_dim - text_dim),
                        device=text_embeddings.device,
                        dtype=text_embeddings.dtype,
                    )
                    embeddings = torch.cat([text_embeddings, pad], dim=-1)
                embeddings = F.normalize(embeddings, dim=-1)
        task_names = list(self.task_centers.keys())
        task_similarities = []
        for task_name in task_names:
            centers = self.task_centers[task_name].to(device=device, dtype=embeddings.dtype)  # (K, D)
            sim_to_centers = torch.matmul(embeddings.float(), centers.float().t())  # (N, K)
            task_score = sim_to_centers.max(dim=1).values  # (N,)
            task_similarities.append(task_score.unsqueeze(1))
        similarities = torch.cat(task_similarities, dim=1)  # (N, num_tasks)

        if top_k == 1:
            best_indices = torch.argmax(similarities, dim=-1)  # (N,)
            best_tasks = [task_names[i.item()] for i in best_indices]
        else:
            topk_indices = torch.topk(similarities, k=min(top_k, len(task_names)), dim=-1).indices
            best_tasks = [
                [task_names[j.item()] for j in topk_indices[i]]
                for i in range(len(texts))
            ]

        return best_tasks, similarities

    def get_task_index(self, task_name):
        """Get the index of a task in the task center list."""
        task_names = list(self.task_centers.keys())
        if task_name in task_names:
            return task_names.index(task_name)
        return -1

    def get_num_tasks(self):
        """Return the number of registered tasks."""
        return len(self.task_centers)

    def save_state(self, save_dir):
        """Save router state (task centers and sample counts)."""
        os.makedirs(save_dir, exist_ok=True)

        # Save task centers
        centers_dict = {
            name: param.data.cpu() for name, param in self.task_centers.items()
        }
        torch.save(centers_dict, os.path.join(save_dir, ROUTER_CENTERS_NAME))

        # Save sample counts
        with open(os.path.join(save_dir, ROUTER_COUNTS_NAME), "w") as f:
            json.dump(self._task_sample_counts, f)

        # Save minimal router config used for future loads.
        with open(os.path.join(save_dir, ROUTER_CONFIG_NAME), "w") as f:
            json.dump(
                {
                    "clip_model_name": self.clip_model_name,
                    "num_centers_per_task": self.num_centers_per_task,
                    "kmeans_iters": self.kmeans_iters,
                    "k1_center_mode": self.k1_center_mode,
                },
                f,
            )

    def load_state(self, save_dir, device="cpu"):
        """Load router state."""
        config_path = os.path.join(save_dir, ROUTER_CONFIG_NAME)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.num_centers_per_task = int(cfg.get("num_centers_per_task", self.num_centers_per_task))
            self.kmeans_iters = int(cfg.get("kmeans_iters", self.kmeans_iters))
            self.k1_center_mode = str(cfg.get("k1_center_mode", self.k1_center_mode)).lower()

        centers_path = os.path.join(save_dir, ROUTER_CENTERS_NAME)
        if os.path.exists(centers_path):
            centers_dict = torch.load(centers_path, map_location=device)
            for name, center in centers_dict.items():
                self.task_centers[name] = nn.Parameter(
                    center.to(device), requires_grad=False
                )

        counts_path = os.path.join(save_dir, ROUTER_COUNTS_NAME)
        if os.path.exists(counts_path):
            with open(counts_path, "r") as f:
                self._task_sample_counts = json.load(f)

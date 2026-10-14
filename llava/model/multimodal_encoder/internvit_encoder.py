import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPImageProcessor


class InternVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True, local_files_only=True
            )

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name, local_files_only=True
        )
        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name, trust_remote_code=True, local_files_only=True
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        hidden_states = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = hidden_states[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = hidden_states
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            pooled_features = []
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    pixel_values=image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = image_forward_out.last_hidden_state[:, 0, :].to(image.dtype)
                feats = self.feature_select(image_forward_out).to(image.dtype)
                pooled_features.append(pooled)
                image_features.append(feats)
            pooled_features = torch.cat(pooled_features, dim=0)
        else:
            image_forward_outs = self.vision_tower(
                pixel_values=images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            pooled_features = image_forward_outs.last_hidden_state[:, 0, :].to(images.dtype)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return pooled_features, image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

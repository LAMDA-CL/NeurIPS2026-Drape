import os

from transformers import AutoConfig

from .clip_encoder import CLIPVisionTower
from .internvit_encoder import InternVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    local_vision_tower = os.environ.get("LLAVA_LOCAL_VISION_TOWER")
    if local_vision_tower and os.path.exists(local_vision_tower):
        vision_tower = local_vision_tower

    if vision_tower is None:
        raise ValueError("vision_tower is not set")

    is_absolute_path_exists = os.path.exists(vision_tower)
    config = None
    if is_absolute_path_exists:
        try:
            config = AutoConfig.from_pretrained(
                vision_tower, trust_remote_code=True, local_files_only=True
            )
        except Exception:
            config = None

    model_type = getattr(config, "model_type", None)
    if model_type == "intern_vit_6b":
        return InternVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")

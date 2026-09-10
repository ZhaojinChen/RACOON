import torch
import torch.nn as nn

from model.DARQC.resnet_qc import resnet_qc_18


def _unwrap_model(model):
    return model.model if hasattr(model, "model") else model


class ResNet18_2D(nn.Module):
    def __init__(
        self,
        input_size=(3, 224, 224),
        dropout=0.2,
        pretrained=True,
        weights="DEFAULT",
        use_ref=False,
    ):
        super().__init__()
        if len(input_size) != 3:
            raise ValueError(
                f"input_size must be (C, H, W) for the 2D model, got {input_size}"
            )

        expected_channels = 6 if use_ref else 3
        if tuple(input_size[1:]) != (224, 224):
            raise ValueError(
                f"ResNetQC expects 224x224 inputs, got spatial size {tuple(input_size[1:])}"
            )
        if input_size[0] != expected_channels:
            raise ValueError(
                f"ResNetQC expects {expected_channels} channels when use_ref={use_ref}, "
                f"got {input_size[0]}"
            )

        normalized_weights = None if weights is None else str(weights).strip()
        if normalized_weights and normalized_weights.lower() in {"none", "null", "false", "0"}:
            pretrained = False

        if normalized_weights not in (None, "", "DEFAULT", "IMAGENET1K_V1"):
            print(
                f"Ignoring pretrained_weights='{weights}'. "
                "resnet_qc.py only exposes the boolean pretrained flag."
            )

        self.model = resnet_qc_18(
            pretrained=pretrained,
            dropout=dropout,
            num_classes=1,
            use_ref=use_ref,
        )

    def forward(self, x):
        return self.model(x)


def load_backbone_weights_resnet18_2d(model: nn.Module, checkpoint_path: str):
    if not checkpoint_path:
        return

    core_model = _unwrap_model(model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}. Expected a state_dict-like object."
        )

    model_state = core_model.state_dict()
    filtered = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("addon.") or new_key.startswith("fc."):
            continue
        if new_key in model_state and model_state[new_key].shape == value.shape:
            filtered[new_key] = value

    missing, unexpected = core_model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Missing keys when loading DARQC 2D backbone: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading DARQC 2D backbone: {unexpected}")


def freeze_backbone(model: nn.Module):
    core_model = _unwrap_model(model)
    for name, param in core_model.named_parameters():
        if name.startswith("addon.") or name.startswith("fc."):
            continue
        param.requires_grad = False

# in this code, we will use CNN for feature extraction and transformer for global context encoding for feature maps.
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# normalize the convolution layers for group batch normalization
class StdConv3d(nn.Conv3d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _is_mapping(value):
    return hasattr(value, "items")


def _to_container(value):
    if _is_mapping(value):
        return {k: _to_container(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ _to_container(v) for v in value ]
    return value


def _cfg_get_nested(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        if key in cfg:
            return cfg.get(key, default)
        params = cfg.get("params")
        if hasattr(params, "get") and key in params:
            return params.get(key, default)
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    params = getattr(cfg, "params", None)
    if params is not None:
        return _cfg_get_nested(params, key, default)
    return default


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config):
    if config is None:
        return None
    if not _is_mapping(config) or "target" not in config:
        return config
    params = _to_container(config.get("params", {}))
    return get_obj_from_str(config["target"])(**params)

class ResNetv2_block(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNet3DBackbone(nn.Module):
    def __init__(
        self,
        config=None,
        cin=None,
        block_units=None,
        width_factor=None,
        nInChannel=None,
        in_channels=None,
    ):
        super().__init__()
        if cin is None:
            cin = nInChannel
        if cin is None:
            cin = in_channels
        if config is not None:
            cin = _cfg_get(config, "nInChannel", cin)
            cin = _cfg_get(config, "in_channels", cin)
            block_units = _cfg_get(config, "block_units", block_units)
            width_factor = _cfg_get(config, "width_factor", width_factor)
        if cin is None or block_units is None or width_factor is None:
            raise ValueError("ResNet3DBackbone requires cin, block_units, and width_factor.")
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv3d(cin, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', ResNetv2_block(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', ResNetv2_block(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', ResNetv2_block(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', ResNetv2_block(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', ResNetv2_block(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', ResNetv2_block(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        x = self.root(x) 
        x = self.body[0](x)    
        x = self.body[1](x)
        x = self.body[2](x)
        return x

# ============================================================
# Patch Embedding for transformer
# ============================================================
class PatchEmbed3D(nn.Module):
    """
    Convert a 3D feature map into a sequence of tokens.

    Example:
      x: (B, Cc, 24, 28, 24)
      patch_size=4
      Conv3d => (B, D, 6, 7, 6)
      flatten => tokens: (B, N, D), where N=6*7*6=252
      + CLS => (B, 253, D)
    """
    def __init__(
        self,
        config=None,
        img_size=None,
        in_ch=None,
        hidden_size=None,
        patch_size=4,
        dropout=0.1,
        add_cls=True,
    ):
        super().__init__()
        if config is not None:
            img_size = _cfg_get(config, "img_size", img_size)
            in_ch = _cfg_get(config, "in_ch", in_ch)
            hidden_size = _cfg_get(config, "hidden_size", hidden_size)
            patch_size = _cfg_get(config, "patch_size", patch_size)
            dropout = _cfg_get(config, "dropout", dropout)
            add_cls = _cfg_get(config, "add_cls", add_cls)

        if img_size is None or in_ch is None or hidden_size is None:
            raise ValueError("PatchEmbed3D requires img_size, in_ch, and hidden_size.")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        patch_size = tuple(patch_size)
        if len(patch_size) != 3:
            raise ValueError("patch_size must have 3 values for 3D inputs.")
        img_size = tuple(img_size)

        self.add_cls = add_cls
        self.proj = nn.Conv3d(
            in_channels=in_ch,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        n_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        n_tokens = n_patches + (1 if add_cls else 0)

        if add_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cc, Hc, Wc, Dc) e.g. (B, Cc, 24, 28, 24)
        x = self.proj(x)  # (B, D, Hc/ps, Wc/ps, Dc/ps) => (B, D, 6, 7, 6)

        # Flatten spatial dims and convert to token sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        B, N, D = x.shape

        # Prepend CLS token if enabled
        if self.add_cls:
            cls = self.cls.expand(B, -1, -1)   # (B, 1, D)
            x = torch.cat([cls, x], dim=1)     # (B, 1+N, D)
            N = N + 1

        x = x + self.position_embeddings
        x = self.drop(x)
        return x  # (B, N, D)


class TransformerEncoder3D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers=4,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        attention_dropout=None,
        batch_first=True,
    ):
        super().__init__()
        if attention_dropout is not None:
            dropout = attention_dropout
        try:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=batch_first,
            )
            self.batch_first = True
        except TypeError:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
            )
            self.batch_first = False
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)
        x = self.encoder(x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return self.norm(x)


# ============================================================
# 4) Regression head: output 9 parameters
# ============================================================

class RegressionHead(nn.Module):
    """
    Map a global embedding (e.g., CLS token) to 9 regression outputs.
    """
    def __init__(self, hidden_size: int, out_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return self.net(x)  # (B, 9)


# ============================================================
# 5) Full model: 3D CNN -> Patch tokens -> Transformer -> CLS -> 9 params
# ============================================================

class CNNViT3DRegressor(nn.Module):
    """
    End-to-end model:
      x (B, 1, 192, 224, 192)
        -> CNN backbone -> feat (B, Cc, 24, 28, 24)
        -> PatchEmbed3D -> tokens (B, 253, D)  [if CLS]
        -> Transformer -> encoded (B, 253, D)
        -> take CLS -> (B, D)
        -> head -> (B, 9)
    """
    def __init__(
        self,
        config=None,
        cnn=None,
        patch_embed=None,
        transformer_encoder=None,
        hidden_size=None,
        out_dim=9,
        resnet_cfg=None,
        patch_cfg=None,
        transformer_cfg=None,
        input_channels=None,
    ):
        super().__init__()
        if config is not None:
            cfg = _to_container(config) if _is_mapping(config) else config
            resnet_cfg = _cfg_get(cfg, "resnet", resnet_cfg)
            patch_cfg = _cfg_get(cfg, "patch", patch_cfg)
            transformer_cfg = _cfg_get(cfg, "transformer", transformer_cfg)
            hidden_size = _cfg_get(cfg, "hidden_size", hidden_size)
            out_dim = _cfg_get(cfg, "out_dim", out_dim)
            input_channels = _cfg_get(cfg, "input_channels", input_channels)

        if cnn is None:
            cnn = self._build_resnet(resnet_cfg, input_channels)
        self.cnn = cnn

        resnet_out_ch = self._infer_resnet_out_channels(self.cnn)
        hidden_size = self._infer_hidden_size(hidden_size, patch_cfg, transformer_cfg)

        if patch_embed is None:
            patch_embed = self._build_patch_embed(patch_cfg, resnet_out_ch, hidden_size)
        self.patch = patch_embed

        if transformer_encoder is None:
            transformer_encoder = self._build_transformer(transformer_cfg, hidden_size)
        self.tr = transformer_encoder

        self.head = RegressionHead(hidden_size, out_dim)

    @staticmethod
    def _infer_resnet_out_channels(cnn):
        if hasattr(cnn, "out_channels"):
            return int(cnn.out_channels)
        if hasattr(cnn, "width"):
            return int(cnn.width) * 16
        return None

    @staticmethod
    def _infer_hidden_size(hidden_size, patch_cfg, transformer_cfg):
        if hidden_size is not None:
            return hidden_size
        hidden_size = _cfg_get_nested(patch_cfg, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
        hidden_size = _cfg_get_nested(transformer_cfg, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
        hidden_size = _cfg_get_nested(transformer_cfg, "d_model", None)
        if hidden_size is not None:
            return hidden_size
        raise ValueError("hidden_size is required for the transformer head.")

    @staticmethod
    def _build_resnet(resnet_cfg, input_channels):
        if resnet_cfg is None:
            raise ValueError("resnet_cfg is required when cnn is not provided.")
        if _is_mapping(resnet_cfg) and "target" in resnet_cfg:
            return instantiate_from_config(resnet_cfg)
        cin = _cfg_get(resnet_cfg, "nInChannel", None)
        cin = _cfg_get(resnet_cfg, "in_channels", cin)
        if cin is None:
            cin = input_channels
        if cin is None:
            raise ValueError("ResNet config requires nInChannel/in_channels or input_channels.")
        block_units = _cfg_get(resnet_cfg, "block_units", None)
        width_factor = _cfg_get(resnet_cfg, "width_factor", None)
        return ResNet3DBackbone(
            cin=cin,
            block_units=block_units,
            width_factor=width_factor,
        )

    @staticmethod
    def _build_patch_embed(patch_cfg, in_ch, hidden_size):
        if patch_cfg is None:
            raise ValueError("patch_cfg is required when patch_embed is not provided.")
        if _is_mapping(patch_cfg) and "target" in patch_cfg:
            return instantiate_from_config(patch_cfg)
        cfg_in_ch = _cfg_get(patch_cfg, "in_ch", None)
        if cfg_in_ch is None:
            cfg_in_ch = in_ch
        if cfg_in_ch is None:
            raise ValueError("Patch config requires in_ch or a backbone with width.")
        return PatchEmbed3D(
            config=patch_cfg,
            in_ch=cfg_in_ch,
            hidden_size=hidden_size,
        )

    @staticmethod
    def _build_transformer(transformer_cfg, hidden_size):
        if transformer_cfg is None:
            raise ValueError("transformer_cfg is required when transformer_encoder is not provided.")
        if _is_mapping(transformer_cfg) and "target" in transformer_cfg:
            return instantiate_from_config(transformer_cfg)
        return TransformerEncoder3D(
            hidden_size=hidden_size,
            num_layers=_cfg_get(transformer_cfg, "num_layers", 4),
            num_heads=_cfg_get(transformer_cfg, "num_heads", 8),
            mlp_dim=_cfg_get(transformer_cfg, "mlp_dim", 2048),
            dropout=_cfg_get(transformer_cfg, "dropout", 0.1),
            attention_dropout=_cfg_get(transformer_cfg, "attention_dropout", None),
            batch_first=_cfg_get(transformer_cfg, "batch_first", True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        tok = self.patch(feat)
        enc = self.tr(tok)
        if isinstance(enc, (tuple, list)):
            enc = enc[0]

        # Use CLS token as global representation (alternative: mean pooling)
        cls = enc[:, 0]          # (B, D)
        y = self.head(cls)       # (B, 9)
        return y

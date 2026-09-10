import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ResParam_model.ResNet18 import Residual_block


class ResNet18Backbone3D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.stemNet1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._makelayer(64, 64, 2, strides=1)
        self.layer2 = self._makelayer(64, 128, 2, strides=2)
        self.layer3 = self._makelayer(128, 256, 2, strides=2)
        self.layer4 = self._makelayer(256, 512, 2, strides=2)

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = [Residual_block(in_channels, out_channels, strides)]
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stemNet1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FiLM3D(nn.Module):
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #(N,module_din,1,1,1)
        betas = betas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (gammas * x) + betas #broadcast of gamma and betas to match the dimension of x by replication

class ParamFiLMGenerator(nn.Module):
    def __init__(
        self,
        param_dim=9,
        module_dim=128, 
        num_modules=4,
        gamma_option="linear",
        gamma_scale=1.0,
        gamma_shift=0.0,
        beta_option="linear",
        beta_scale=1.0,
        beta_shift=0.0,
    ):
        super().__init__()
        # Since module_num_layers is fixed to 1, hidden_dim is always 2 * module_dim
        self.module_dim = module_dim
        self.num_modules = num_modules
        self.hidden_dim = 2 * module_dim 

        # MLP maps input params to a vector containing all gamma/beta pairs for all modules
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, num_modules * self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.func_list = {
            "linear": None,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "exp": torch.exp,
        }
        
        self.gamma_option = gamma_option
        self.gamma_scale = gamma_scale
        self.gamma_shift = gamma_shift
        self.beta_option = beta_option
        self.beta_scale = beta_scale
        self.beta_shift = beta_shift

    def _modify_output(self, out):
        """
        Applies non-linearities, scaling, and shifting to the raw MLP output.
        out shape: (Batch, num_modules, 2 * module_dim)
        """
        gamma_func = self.func_list[self.gamma_option]
        beta_func = self.func_list[self.beta_option]
        
        # Split the last dimension into gamma and beta (128 each if module_dim=128)
        gamma = out[:, :, :self.module_dim]
        beta = out[:, :, self.module_dim:]

        # Apply transformations to Gamma
        if gamma_func is not None:
            gamma = gamma_func(gamma)
        if self.gamma_scale != 1.0:
            gamma = gamma * self.gamma_scale
        if self.gamma_shift != 0.0:
            gamma = gamma + self.gamma_shift

        # Apply transformations to Beta
        if beta_func is not None:
            beta = beta_func(beta)
        if self.beta_scale != 1.0:
            beta = beta * self.beta_scale
        if self.beta_shift != 0.0:
            beta = beta + self.beta_shift

        # Concatenate back along the last dimension
        return torch.cat([gamma, beta], dim=-1)

    def forward(self, params):
        # Input: (N, param_dim) -> Output: (N, num_modules * 2 * module_dim)
        h = self.mlp(params) 
        
        # Reshape to (Batch, num_modules, hidden_dim)
        h = h.view(-1, self.num_modules, self.hidden_dim) 
        
        # Transform raw values into final FiLM parameters
        film_params = self._modify_output(h)
        return film_params


class FiLMedNet(nn.Module):
    def __init__(self, vocab, feature_dim=(512, 6, 7,6), # feature dim should be the dimension of feature maps which is obtained just after CNN backbone and before FiLM layers
                input_channels=1,
                stem_num_layers=2,
                stem_batchnorm=False,
                stem_kernel_size=3,
                stem_stride=1, # this is not for resnet stem, but for the initial stem in FiLMedNet before the FiLM layers.
                stem_padding=None,
                num_modules=4,
                module_dim=128,
                module_residual=True,
                module_batchnorm=True,
                module_batchnorm_affine=False,
                module_dropout=0,
                module_input_proj=1,
                module_kernel_size=3,
                condition_method='bn-film', # use FiLM instead of concat
                condition_pattern=True, 
                use_gamma=True,
                use_beta=True,
                use_coords=1,
                FiLM_param_dim=9, 
                FiLM_gamma_option="linear",
                FiLM_gamma_scale=1.0,
                FiLM_gamma_shift=0.0,
                FiLM_beta_option="linear",
                FiLM_beta_scale=1.0,
                FiLM_beta_shift=0.0,
                avg_pooling = False,
                device = 'cuda',
                ):
        super(FiLMedNet, self).__init__()
        self.backbone = ResNet18Backbone3D(input_channels)
        self.num_modules = num_modules
        self.module_batchnorm = module_batchnorm
        self.module_dim = module_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.condition_method = condition_method
        self.condition_pattern = condition_pattern #
        self.use_coords_freq = use_coords
        self.ParamFiLMGenerator = ParamFiLMGenerator(FiLM_param_dim, module_dim, num_modules,FiLM_gamma_option, FiLM_gamma_scale, FiLM_gamma_shift, FiLM_beta_option,FiLM_beta_scale,FiLM_beta_shift)

        # Initialize helper variables
        self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)

        module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  
        module_W = feature_dim[2] // (stem_stride ** stem_num_layers) 
        module_D = feature_dim[3] // (stem_stride ** stem_num_layers)  
        self.coords = self._coord_map_3d(module_H, module_W, module_D,device)

        self.num_extra_channels = 3 if self.use_coords_freq > 0 else 0
        stem_feature_dim = feature_dim[0] + (self.stem_use_coords * self.num_extra_channels)
        self.stem = self.build_stem(
            in_channels=stem_feature_dim,
            out_channels=module_dim,
            num_layers=stem_num_layers,
            with_batchnorm=stem_batchnorm,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
        )
        self.block = FiLMedResBlock
        self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
        self.fwd_count = 0

        # Initialize FiLMed network body
        self.function_modules = nn.ModuleDict()
        for fn_num in range(self.num_modules):
            mod = self.block(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm,
                            with_cond=self.condition_pattern,
                            dropout=module_dropout,
                            num_extra_channels=self.num_extra_channels,
                            extra_channel_freq= self.use_coords_freq,
                            with_input_proj=module_input_proj,
                            num_cond_maps=self.num_cond_maps,
                            kernel_size=module_kernel_size,
                            batchnorm_affine=module_batchnorm_affine,
                            condition_method=condition_method)
            self.function_modules[str(fn_num)] = mod

        # Initialize output classifier
        flattened_size = (module_dim + self.num_extra_channels)*module_H*module_W*module_D
        self.If_avg_pool=avg_pooling
        if avg_pooling:
            self.avgpool = nn.AvgPool3d((module_H,module_W,module_D),stride=1) #we could consider for reduce the parameter we need to train for classification model
            self.classifier = nn.Sequential(
                nn.Linear(module_dim + self.num_extra_channels, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Output layer for regression
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Output layer for regression
            )

    def _coord_map_3d(self, D, H, W, device=None, start=-1.0, end=1.0):
        device = device if device else 'cpu' # device should passed to the function then

        z = torch.linspace(start, end, steps=D, device=device, dtype=torch.float32)
        y = torch.linspace(start, end, steps=H, device=device, dtype=torch.float32)
        x = torch.linspace(start, end, steps=W, device=device, dtype=torch.float32)
        zz = z[:, None, None].expand(D, H, W)
        yy = y[None, :, None].expand(D, H, W)
        xx = x[None, None, :].expand(D, H, W)
        return torch.stack([xx, yy, zz], dim=0)
    
    def build_stem(self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        with_batchnorm: bool = True,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ): #Simply CNN layers
        """
        Stem = [Conv -> (BN) -> ReLU] x num_layers
        Input : (N, in_channels, H, W)
        Output: (N, out_channels, H_out, W_out)
        """
        if padding is None:
            if kernel_size % 2 == 0:
                raise NotImplementedError("Only odd kernel_size supported when padding=None")
            padding = kernel_size // 2

        layers = []
        prev_channels = in_channels

        for _ in range(num_layers):
            layers.append(
                nn.Conv3d(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not with_batchnorm,  # common practice
                )
            )
            if with_batchnorm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, param): # x is the original mri image? 
        self.fwd_count += 1
        x = self.backbone(x) # get the feature map from the backbone of ResNet18 # (N,512,6,7,6)
        film = self.ParamFiLMGenerator(param) #the embedding output of param and this will be then used to modulate the output

        # Propagate up image features CNN
        batch_coords = None
        if self.use_coords_freq > 0:
            batch_coords = self.coords.unsqueeze(0).expand(x.size(0), *self.coords.size())
        if self.stem_use_coords:
            x = torch.cat([x, batch_coords], 1)

        feats = self.stem(x) #(N,module_dim,6,7,6)
        N, C, D, H, W = feats.size()
        
        if self.condition_method == 'concat':
            cond_params = film[:, :, :2*self.module_dim]  # (N, M, 2D)
            cond_maps = cond_params.view(N, self.num_modules, 2*self.module_dim, 1, 1, 1)
            cond_maps = cond_maps.expand(N, self.num_modules, 2*self.module_dim, D, H, W)
        else:
            gammas, betas = torch.split(film[:, :, :2*self.module_dim], self.module_dim, dim=-1)  # (N,M,D) & (N,M,D)
            if not self.use_gamma:
                gammas = self.default_weight.expand_as(gammas)  # (N,M,D)
            if not self.use_beta:
                betas = self.default_bias.expand_as(betas)      # (N,M,D)
                
        # Propagate up the network from low-to-high numbered blocks
        current_feats = feats
        for fn_num in range(self.num_modules):
            # Use str(fn_num) because you used nn.ModuleDict with string keys
            if self.condition_method == 'concat':
                current_feats = self.function_modules[str(fn_num)](
                    current_feats, extra_channels=batch_coords, cond_maps=cond_maps[:, fn_num]
                )
            else:
                current_feats = self.function_modules[str(fn_num)](
                    current_feats, gammas[:, fn_num, :], betas[:, fn_num, :], batch_coords
                )
        final_module_output = current_feats

        # Run the final classifier over the resultant, post-modulated features.
        if self.use_coords_freq > 0:
            final_module_output = torch.cat([final_module_output, batch_coords], 1) # flatten image
        
        if self.If_avg_pool:
            final_module_output = self.avgpool(final_module_output)
            final_module_output = final_module_output.view(final_module_output.size(0), -1)
            out = self.classifier(final_module_output)
        else: 
            final_module_output = final_module_output.view(final_module_output.size(0), -1) # flatten image
            out = self.classifier(final_module_output)
        return out


class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True,
                with_cond=False, dropout=0, num_extra_channels=0, extra_channel_freq=1,
                with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,condition_method='bn-film'):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond # decide whether to add FiLM layer
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.condition_method = condition_method
        
        if self.with_input_proj % 2 == 0:
            raise(NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise(NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond:
            self.film = FiLM3D()
        if self.with_input_proj:
            self.input_proj = nn.Conv3d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                    in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

        self.conv1 = nn.Conv3d(in_dim + self.num_cond_maps +
                            (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                                out_dim, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2)
        if self.condition_method == 'conv-film' and self.with_cond:
            self.film = FiLM3D()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm3d(out_dim, affine=((not self.with_cond) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond:
            self.film = FiLM3D()
        if dropout > 0:
            self.drop = nn.Dropout3d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
            and self.with_cond):
            self.film = FiLM3D()

    def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):

        if self.condition_method == 'block-input-film' and self.with_cond:
            x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = F.relu(self.input_proj(x))
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        #the difference between these three format is that the position of FiLM layer is different.
        if self.condition_method == 'conv-film' and self.with_cond:
            out = self.film(out, gammas, betas)
        if self.with_batchnorm:
            out = self.bn1(out)
        if self.condition_method == 'bn-film' and self.with_cond:
            out = self.film(out, gammas, betas)
        if self.dropout > 0:
            out = self.drop(out)
        out = F.relu(out)
        if self.condition_method == 'relu-film' and self.with_cond:
            out = self.film(out, gammas, betas)

        if self.with_residual:
            out = x + out
        if self.condition_method == 'block-output-film' and self.with_cond:
            out = self.film(out, gammas, betas)
        return out

def load_backbone_weights(model, checkpoint_path):
    if checkpoint_path is None:
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    filtered = {k: v for k, v in state.items() if not k.startswith("fnn.")}
    missing, unexpected = model.backbone.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Missing keys when loading backbone: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading backbone: {unexpected}")

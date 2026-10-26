import torch
import torch.nn as nn
import torch.nn.functional as F

#encoder DINOv2 pour extraire feature avant de train le modèle 
class DinoV2Encoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True, resize_to=224): #freeze pour le que 
        #le modèle soit entrainé que sur l'encoder decoder #resize pour que ce soit un multiple de 14
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.patch_size = 14
        self.resize_to = resize_to
        self.out_channels = self.model.embed_dim  # 384 pour vits14, 768 vitb14, 1024 vitl14
        self.grid_size = resize_to // self.patch_size  # 16 for resize_to=224        

        if freeze: #pour ne pas entrainer l'encoder poids pas mis à jour, gradients pas caluclés etc
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        

    def forward(self, x):
        # x: (B, 3, H, W) valeurs dans [0,1]
        B = x.shape[0]
        x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False) #deja a la bonne taille mais on garde pour le cas ou on change la taille de l'image

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1) #avec stats ImageNet pour normaliser les images avant de les passer dans le modèle DINOv2
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, C) N nombre de patchs par images (16x16=256), C nombre de features, B batch size 

        n_side = self.resize_to // self.patch_size  # 224/14 = 16
        C = patch_tokens.shape[-1]
        grid = patch_tokens.permute(0, 2, 1).reshape(B, C, n_side, n_side)  # (B, C, 16, 16)
        return grid
    


#transformer encoder decoder
class FeatureTransformer(nn.Module):
    def __init__(self, embed_dim=384, n_heads=6, n_encoder_layers=3, n_decoder_layers=3, #comme dans foundad paper (nb de layer)
                 mlp_ratio=4, dropout=0.1, n_tokens=256):
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        # créé ici pour être enregistré avant la création de l'optimizer
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_grid):
        B, C, H, W = x_grid.shape
        tokens = x_grid.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        assert tokens.shape[1] == self.pos_embed.shape[1], \
            f"n_tokens={tokens.shape[1]} != pos_embed {self.pos_embed.shape[1]}"
        tokens = tokens + self.pos_embed

        encoded = self.transformer_encoder(tokens)
        decoded = self.transformer_decoder(encoded)
        return decoded.permute(0, 2, 1).reshape(B, C, H, W)
    
class ChangeDetectionModel(nn.Module):
    def __init__(self, encoder, transformer):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        
    def forward(self, im1, im2):
        x = self.encoder(im1)
        with torch.no_grad():
            y = self.encoder(im2)
        x_hat = self.transformer(x)
        return x_hat, y

class ResNet50Encoder(nn.Module):
    """Example of a second frozen encoder: torchvision ResNet-50 (ImageNet, supervised), cut after layer3.
    layer3 has stride 16, so a 256 crop gives a 16x16 grid of 1024-channel features: same grid as DINOv2 with
    resize_to=224, so patch labels and all the analysis stay directly comparable."""

    def __init__(self, freeze=True, input_size=256, weights="IMAGENET1K_V2"):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet50(weights=weights)
        self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        self.input_size = input_size
        self.out_channels = 1024
        self.grid_size = input_size // 16
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
        self.eval()

    def forward(self, x):
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.body((x - mean) / std)  # (B, 1024, 16, 16)

class DinoV3SEncoder(nn.Module):
    """Frozen DINOv3 ViT-S/16 (web images, LVD-1689M), via Hugging Face transformers >= 4.56.
    Patch 16, so a 256 crop gives 16x16 tokens with no resize (same grid as DINOv2 at 224 and ResNet-50 layer3).
    last_hidden_state = output of the final LayerNorm; tokens ordered [CLS, registers, patches]."""

    def __init__(self, input_size=256):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.patch_size = self.model.config.patch_size              # 16
        self.input_size = input_size
        self.grid_size = input_size // self.patch_size              # 16
        self.out_channels = self.model.config.hidden_size           # 384
        self.n_prefix = 1 + self.model.config.num_register_tokens   # CLS + 4 registers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        B = x.shape[0]
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        out = self.model(pixel_values=(x - self.mean) / self.std)
        tokens = out.last_hidden_state[:, self.n_prefix:]           # keep patch tokens only
        n = self.grid_size
        assert tokens.shape[1] == n * n, f"{tokens.shape[1]} patch tokens, expected {n * n}"
        return tokens.permute(0, 2, 1).reshape(B, self.out_channels, n, n)
    

class DinoV3LEncoder(nn.Module):
    """Frozen DINOv3 ViT-L/16, web (LVD-1689M) or satellite (SAT-493M, Maxar RGB 0.6 m), via transformers.
    Same architecture and training method for both: only the pretraining data differ.
    Patch 16, so a 256 crop gives 16x16 tokens with no resize. 1024 channels per patch."""
    VARIANTS = {
        "dinov3_vitl16":     ("facebook/dinov3-vitl16-pretrain-lvd1689m",
                              (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet stats
        "dinov3_vitl16_sat": ("facebook/dinov3-vitl16-pretrain-sat493m",
                              (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),   # satellite stats (DINOv3 README)
    }

    def __init__(self, name, input_size=256):
        super().__init__()
        from transformers import AutoModel
        repo, mean, std = self.VARIANTS[name]
        self.model = AutoModel.from_pretrained(repo)
        self.patch_size = self.model.config.patch_size              # 16
        self.input_size = input_size
        self.grid_size = input_size // self.patch_size              # 16
        self.out_channels = self.model.config.hidden_size           # 1024
        self.n_prefix = 1 + self.model.config.num_register_tokens   # CLS + registers
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        B = x.shape[0]
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        out = self.model(pixel_values=(x - self.mean) / self.std)
        tokens = out.last_hidden_state[:, self.n_prefix:]           # keep patch tokens only
        n = self.grid_size
        assert tokens.shape[1] == n * n, f"{tokens.shape[1]} patch tokens, expected {n * n}"
        return tokens.permute(0, 2, 1).reshape(B, self.out_channels, n, n)

class SamViTBEncoder(nn.Module):
    """Frozen SAM ViT-B image encoder (facebook/sam-vit-base, trained on SA-1B masks), via transformers.
    SAM's position embeddings are fixed to 1024x1024 inputs, so the crop is upsampled to 1024.
    Output = SAM 'image embeddings' (after the neck): (B, 256, 64, 64), average-pooled 4x4 -> (B, 256, 16, 16),
    so each cell covers the same ground area as a patch of the other encoders."""

    def __init__(self, chunk=4):
        super().__init__()
        from transformers import SamModel
        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.encoder = sam.vision_encoder        # keep only the image encoder
        del sam
        self.sam_size = self.encoder.config.image_size                                # 1024
        self.sam_grid = self.sam_size // self.encoder.config.patch_size               # 64
        self.grid_size = 16
        self.out_channels = self.encoder.config.output_channels                       # 256
        self.chunk = chunk                       # images per forward pass, limits GPU memory at 1024x1024
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        x = F.interpolate(x, size=(self.sam_size, self.sam_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = [self.encoder(pixel_values=c).last_hidden_state for c in x.split(self.chunk)]
        f = torch.cat(feats)                                                          # (B, 256, 64, 64)
        assert f.shape[-1] == self.sam_grid, f"SAM grid {tuple(f.shape[-2:])}, expected {self.sam_grid}"
        return F.adaptive_avg_pool2d(f, self.grid_size)                          # (B, 256, 16, 16)

class SamViTBEncoder(nn.Module):
    """Frozen SAM ViT-B image encoder (facebook/sam-vit-base, trained on SA-1B masks), via transformers.
    SAM's position embeddings are fixed to 1024x1024 inputs, so the crop is upsampled to 1024.
    Output = SAM 'image embeddings' (after the neck): (B, 256, 64, 64), average-pooled 4x4 -> (B, 256, 16, 16),
    so each cell covers the same ground area as a patch of the other encoders."""

    def __init__(self, chunk=4):
        super().__init__()
        from transformers import SamModel
        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.encoder = sam.vision_encoder        # keep only the image encoder
        del sam
        self.sam_size = self.encoder.config.image_size                                # 1024
        self.sam_grid = self.sam_size // self.encoder.config.patch_size               # 64
        self.grid_size = 16
        self.out_channels = self.encoder.config.output_channels                       # 256
        self.chunk = chunk                       # images per forward pass, limits GPU memory at 1024x1024
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        x = F.interpolate(x, size=(self.sam_size, self.sam_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = [self.encoder(pixel_values=c).last_hidden_state for c in x.split(self.chunk)]
        f = torch.cat(feats)                                                          # (B, 256, 64, 64)
        assert f.shape[-1] == self.sam_grid, f"SAM grid {tuple(f.shape[-2:])}, expected {self.sam_grid}"
        return F.adaptive_avg_pool2d(f, self.grid_size)                               # (B, 256, 16, 16)
    
def build_encoder(cfg, device=None):
    """Single place where an encoder name from the config becomes a model.
    Every encoder must: take (B, 3, H, W) images in [0, 1], return a (B, C, h, w) feature grid that covers the
    whole crop, and expose .out_channels (C) and .grid_size (h = w). Add a new encoder = one class + one branch here."""
    name = cfg["encoder"]
    if name.startswith("dinov2"):
        enc = DinoV2Encoder(model_name=name, freeze=True, resize_to=cfg["resize_to"])
    elif name == "dinov3_vits16":
        enc = DinoV3SEncoder(input_size=cfg["crop"])
    elif name in ("dinov3_vitl16", "dinov3_vitl16_sat"):
        enc = DinoV3LEncoder(name, input_size=cfg["crop"])
    elif name == "sam_vitb":
        enc = SamViTBEncoder()
    elif name == "resnet50_l3":
        enc = ResNet50Encoder(freeze=True, input_size=cfg["crop"])
    else:
        raise ValueError(f"Unknown encoder {name!r} (add it to build_encoder in src/pds_cd/model.py)")
    enc = enc.eval()
    return enc.to(device) if device is not None else enc


def build_models(cfg, device):
    """Frozen encoder + FeatureTransformer, both built from the config."""
    enc = build_encoder(cfg, device)
    tr = FeatureTransformer(embed_dim=enc.out_channels, n_heads=cfg["n_heads"],
                            n_encoder_layers=cfg["n_encoder_layers"],
                            n_decoder_layers=cfg["n_decoder_layers"],
                            dropout=cfg["dropout"], n_tokens=enc.grid_size ** 2).to(device)
    return enc, tr
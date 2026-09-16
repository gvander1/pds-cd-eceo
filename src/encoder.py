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
    def __init__(self, embed_dim=384, n_heads=6, n_encoder_layers=4, n_decoder_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        self.pos_embed = None  # initialisé au premier forward selon n_side

    def _build_pos_embed(self, n_tokens, device): #position des tokens dans l'image (sera ajouté aux tokens pour que le transformer sache où se trouve chaque token dans l'image et modifiée avec le gradient descent)
        pos = torch.zeros(1, n_tokens, self.embed_dim, device=device)
        nn.init.trunc_normal_(pos, std=0.02)
        return nn.Parameter(pos)

    def forward(self, x_grid):
        B, C, H, W = x_grid.shape
        tokens = x_grid.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        if self.pos_embed is None or self.pos_embed.shape[1] != tokens.shape[1]: 
            self.pos_embed = self._build_pos_embed(tokens.shape[1], tokens.device) 

        tokens = tokens + self.pos_embed

        encoded = self.transformer_encoder(tokens)
        decoded = self.transformer_decoder(encoded)

        x_hat = decoded.permute(0, 2, 1).reshape(B, C, H, W)
        return x_hat
    
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

class Trainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.transformer.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def train_step(self, im1, im2):
        self.optimizer.zero_grad()
        x_hat, y = self.model(im1, im2)
        loss = self.loss_fn(x_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
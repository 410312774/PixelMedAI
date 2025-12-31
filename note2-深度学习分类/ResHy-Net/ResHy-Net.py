
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResHyNet(nn.Module):
    """
    ResHy-Net (minimal): ResNet18 CNN branch + Transformer branch with feature-level fusion.

    Input:
      x: (B, C, H, W)

    Outputs:
      logits: (B, num_classes)

    Notes:
      - CNN branch: ResNet18 backbone (no final FC)
      - Transformer branch: patch embedding + TransformerEncoder + global pooling
      - Fusion: concatenate [cnn_vec, tr_vec] -> MLP classifier
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        resnet_pretrained: bool = False,
        fusion_hidden: int = 512,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size

        # ---- ResNet18 branch (feature extractor) ----
        rn = resnet18(weights=("DEFAULT" if resnet_pretrained else None))
        if in_channels != 3:
            rn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet_stem_to_pool = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4,
        )
        self.resnet_avgpool = rn.avgpool  # AdaptiveAvgPool2d((1,1))
        self.cnn_out_dim = 512  # ResNet18 final feature dim

        # ---- Transformer branch (ViT-style) ----
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        n_patches = (image_size // patch_size) * (image_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.tr_norm = nn.LayerNorm(embed_dim)

        # ---- Fusion head ----
        fusion_in = self.cnn_out_dim + embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

        self._init_params()

    def _init_params(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # patch_embed conv already has default init; keep as is

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) where H=W=image_size (recommended)
        """
        B, C, H, W = x.shape

        # --- CNN branch ---
        cnn_feat = self.resnet_stem_to_pool(x)                 # (B, 512, H/32, W/32)
        cnn_vec = self.resnet_avgpool(cnn_feat).flatten(1)     # (B, 512)

        # --- Transformer branch ---
        p = self.patch_embed(x)                                # (B, E, H/P, W/P)
        p = p.flatten(2).transpose(1, 2)                       # (B, N, E)

        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, E)
        tok = torch.cat([cls, p], dim=1)                       # (B, 1+N, E)

        # If input size differs from configured image_size, interpolate positional embedding
        if tok.size(1) != self.pos_embed.size(1):
            tok = self._forward_with_interpolated_pos(tok, H, W)
        else:
            tok = self.pos_drop(tok + self.pos_embed)

        tok = self.attn_drop(tok)
        tok = self.transformer(tok)                            # (B, 1+N, E)
        tok = self.tr_norm(tok)
        tr_vec = tok[:, 0]                                     # CLS token (B, E)

        # --- Fusion ---
        fused = torch.cat([cnn_vec, tr_vec], dim=1)            # (B, 512+E)
        logits = self.fusion_head(fused)                       # (B, num_classes)
        return logits

    def _forward_with_interpolated_pos(self, tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Interpolate positional embeddings when runtime image size != configured image_size.
        Assumes square patch grid.
        """
        B, L, E = tok.shape
        # Separate cls and patch tokens
        cls_tok = tok[:, :1, :]  # (B,1,E)
        patch_tok = tok[:, 1:, :]  # (B,N,E)

        # Current grid
        gh = H // self.patch_size
        gw = W // self.patch_size
        N = gh * gw
        assert patch_tok.size(1) == N, "Patch token count mismatch; check patch_size vs input H/W"

        # Original pos_embed grid
        pos = self.pos_embed  # (1,1+N0,E)
        pos_cls = pos[:, :1, :]
        pos_patch = pos[:, 1:, :]  # (1,N0,E)
        n0 = pos_patch.size(1)
        g0 = int(n0 ** 0.5)
        assert g0 * g0 == n0, "Configured pos_embed is not a square grid"

        pos_patch = pos_patch.reshape(1, g0, g0, E).permute(0, 3, 1, 2)  # (1,E,g0,g0)
        pos_patch = F.interpolate(pos_patch, size=(gh, gw), mode="bicubic", align_corners=False)
        pos_patch = pos_patch.permute(0, 2, 3, 1).reshape(1, gh * gw, E)  # (1,N,E)

        pos_new = torch.cat([pos_cls, pos_patch], dim=1)  # (1,1+N,E)
        return self.pos_drop(tok + pos_new)


if __name__ == "__main__":
    # quick sanity check
    model = ResHyNet(in_channels=3, num_classes=2, image_size=224, patch_size=16, embed_dim=256, depth=4, num_heads=8)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("logits shape:", logits.shape)  # (2, 2)

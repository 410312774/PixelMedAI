import torch  
import torch.nn.functional as F  
from torch import nn  

from einops import rearrange  
from einops.layers.torch import Rearrange  
from ResNet import ResNet,ResNetBottleneck,get_inplanes

# ---------------------- Helper Functions ----------------------  

# Utility to pair single scalars into tuples  
def pair(t):  
    return t if isinstance(t, tuple) else (t, t)  

# Positional embedding using sin-cos for 3D features  
def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):  
    _, f, h, w, dim, device = *patches.shape, patches.device  

    z, y, x = torch.meshgrid(  
        torch.arange(f, device=device),  
        torch.arange(h, device=device),  
        torch.arange(w, device=device),  
        indexing="ij",  
    )  

    fourier_dim = dim // 6  
    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)  
    omega = 1.0 / (temperature ** omega)  

    z = z.flatten()[:, None] * omega[None, :]  
    y = y.flatten()[:, None] * omega[None, :]  
    x = x.flatten()[:, None] * omega[None, :]  

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)  
    pe = F.pad(pe, (0, dim - (fourier_dim * 6)))  # Pad if dim not divisible by 6  
    return pe.type(dtype)  


# ---------------------- Neural Network Classes ----------------------  

class FeedForward(nn.Module):  
    def __init__(self, dim, hidden_dim):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.LayerNorm(dim),  
            nn.Linear(dim, hidden_dim),  
            nn.GELU(),  
            nn.Linear(hidden_dim, dim),  
        )  

    def forward(self, x):  
        return self.net(x)  


class Attention(nn.Module):  
    def __init__(self, dim, heads=8, dim_head=64):  
        super().__init__()  
        inner_dim = dim_head * heads  
        self.heads = heads  
        self.scale = dim_head ** -0.5  
        self.norm = nn.LayerNorm(dim)  

        self.attend = nn.Softmax(dim=-1)  
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  
        self.to_out = nn.Linear(inner_dim, dim, bias=False)  

    def forward(self, x):  
        x = self.norm(x)  

        qkv = self.to_qkv(x).chunk(3, dim=-1)  
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)  

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  
        attn = self.attend(dots)  

        out = torch.matmul(attn, v)  
        out = rearrange(out, "b h n d -> b n (h d)")  
        return self.to_out(out)  


class Transformer(nn.Module):  
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):  
        super().__init__()  
        self.norm = nn.LayerNorm(dim)  
        self.layers = nn.ModuleList(  
            [  
                nn.ModuleList(  
                    [  
                        Attention(dim, heads=heads, dim_head=dim_head),  
                        FeedForward(dim, mlp_dim),  
                    ]  
                )  
                for _ in range(depth)  
            ]  
        )  

    def forward(self, x):  
        for attn, ff in self.layers:  
            x = attn(x) + x  
            x = ff(x) + x  
        return self.norm(x)  


class CrossAttention(nn.Module):  
    """  
    Handles cross-attention between two temporal features (Baseline and Follow-up).  
    """  
    def __init__(self, dim):  
        super().__init__()  
        self.attend = Attention(dim)  

    def forward(self, fa, fb):  
        updated_fa = self.attend(fb) + fa  
        updated_fb = self.attend(fa) + fb  
        return updated_fa, updated_fb  


class ResViT(nn.Module):  
    def __init__(  
        self,  
        *,  
        image_size,  
        image_patch_size,  
        frames,  
        frame_patch_size,  
        num_classes,  
        dim,  
        depth,  
        heads,  
        mlp_dim,  
        channels=3,  
        dim_head=64,  
    ):  
        super().__init__()  
        image_height, image_width = pair(image_size)  
        patch_height, patch_width = pair(image_patch_size)  

        assert (  
            image_height % patch_height == 0 and image_width % patch_width == 0  
        ), "Image dimensions must be divisible by the patch size."  
        assert frames % frame_patch_size == 0, "Frames must be divisible by the frame patch size"  

        num_patches = (  
            (image_height // patch_height)  
            * (image_width // patch_width)  
            * (frames // frame_patch_size)  
        )  
        patch_dim = channels * patch_height * patch_width * frame_patch_size  

        self.to_patch_embedding = nn.Sequential(  
            Rearrange(  
                "b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)",  
                p1=patch_height,  
                p2=patch_width,  
                pf=frame_patch_size,  
            ),  
            nn.LayerNorm(patch_dim),  
            nn.Linear(patch_dim, dim),  
            nn.LayerNorm(dim),  
        )  

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)  

        self.to_latent = nn.Identity()  
        self.linear_head = nn.Linear(dim, num_classes)  

        # Cross-Attention Module  
        self.cross_attention = CrossAttention(dim)  

    def forward(self, video, resnet, attention):  
        # Split into Baseline (BL) and Follow-up (FU)  
        video_bl, video_fu = video[:, :, :16, :, :], video[:, :, 16:, :, :]  

        # Feature Extraction via ResNet  
        def extract_features(inputs):  
            features = resnet(inputs)  # Shape: [B, 1, C, H, W]  
            features = features.squeeze(1).flatten(2, 3)  # Reshape for attention module  
            features = attention(features)  # Apply attention  
            return features  

        features_bl = extract_features(video_bl)  
        features_fu = extract_features(video_fu)  

        # Cross-Attention between BL and FU  
        updated_bl, updated_fu = self.cross_attention(features_bl, features_fu)  

        # Merge features  
        merged_features = updated_bl + updated_fu  
        merged_features = merged_features.reshape(merged_features.size(0), 1, 8, 16, 16)  

        # Patch Embedding  
        x = self.to_patch_embedding(merged_features)  

        # Add Positional Embedding  
        pe = posemb_sincos_3d(x)  
        x = rearrange(x, "b ... d -> b (...) d") + pe  

        # Transformer Encoding  
        x = self.transformer(x)  
        x = x.mean(dim=1)  

        # Output Head  
        x = self.to_latent(x)  
        return self.linear_head(x)  


# ---------------------- Testing the Model ----------------------  

# Instantiate SimpleViT  
vit = ResViT(  
    image_size=128,  
    frames=32,  
    channels=1,  
    image_patch_size=16,  
    frame_patch_size=2,  
    num_classes=2,  
    dim=1024,  
    depth=6,  
    heads=8,  
    mlp_dim=2048,  
)  

# Mock data and networks  
video = torch.randn(4, 1, 32, 128, 128)  
resnet = ResNet(ResNetBottleneck, [2, 2, 2, 2], get_inplanes(), n_classes=2,in_channels=1)
attention = Attention(dim=32)  # Attention for ResNet feature maps  
# Forward pass  
preds = vit(video, resnet, attention)  
print(preds.shape)  # Output: torch.Size([4, 2])
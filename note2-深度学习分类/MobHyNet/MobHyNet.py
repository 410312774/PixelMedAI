import torch
import torch.nn as nn
import torch.nn.functional as F
import MobileNetV2
# 多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        # x: (B, 1, dim) 或 (B, N, dim)
        attn_output, _ = self.mha(x, x, x)
        return attn_output

# MobHy-Net总体结构
class MobHyNet(nn.Module):
    def __init__(self, 
                 n_modalities=2,          
                 clinical_dim=100,         
                 backbone_kwargs={}):     
        super().__init__()
        self.backs = nn.ModuleList([
            MobileNetV2(num_classes=backbone_kwargs.get("num_classes", 1000),
                        sample_size=backbone_kwargs.get("sample_size", 128),
                        width_mult=backbone_kwargs.get("width_mult", 1.),
                        in_channels=1)
            for _ in range(n_modalities)
        ])
        for b in self.backs:
            b.classifier = nn.Identity()

        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(inplace=True)
        )
        total_feat = n_modalities * self.backs[0].last_channel + 128
        self.fusion_fc = nn.Linear(total_feat, 512)
        self.attn = MultiHeadSelfAttention(512, num_heads=4)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)    
        )
    def forward(self, images, clinical):
        feats = [b(img) for b, img in zip(self.backs, images)]
        feats = [f.view(f.size(0), -1) for f in feats]
        x_img = torch.cat(feats, dim=1)
        x_clin = self.clinical_fc(clinical)
        x_all = torch.cat([x_img, x_clin], dim=1)
        x_fusion = F.relu(self.fusion_fc(x_all)).unsqueeze(1)
        x_attn = self.attn(x_fusion).squeeze(1)
        out = self.out_fc(x_attn)
        return out

# 示例：假定有两个超声模态和10维临床特征
if __name__ == '__main__':
  
    model = MobHyNet(n_modalities=2, clinical_dim=10, backbone_kwargs={
        "num_classes": 4, "sample_size":128, "width_mult":1., "in_channels":1
    })

    # batch size=4, 两个模态输入
    imgs = [torch.randn(4,1,16,128,128), torch.randn(4,1,16,128,128)]
    clinical = torch.randn(4,100)
    out = model(imgs, clinical)
    print(out.shape)  # torch.Size([4,2])
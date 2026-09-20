import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """تقسیم تصویر به پچ‌های کوچک و افزودن موقعیت و توکن رده‌بندی (CLS)"""
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128, img_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        # استخراج پچ‌ها با لایه کانولوشن با stride برابر اندازه پچ
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (img_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(num_patches + 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)  # خروجی: (b, emb_size, h/p, w/p)
        x = x.flatten(2).transpose(1, 2)  # خروجی: (b, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # افزودن CLS token در ابتدای سکوئنس
        x = x + self.positions
        return x

class MultiHeadSelfAttention(nn.Module):
    """پیاده‌سازی مکانیسم Scaled Dot-Product Attention چندسره"""
    def __init__(self, emb_size: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attention), v)

        out = out.permute(0, 2, 1, 3).reshape(b, n, self.emb_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    """یک بلوک استاندارد ترنسفورمر شامل Pre-LayerNorm، Attention و MLP"""
    def __init__(self, emb_size: int = 128, num_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * mlp_ratio, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """معماری کامل ViT برای کلاسیفیکیشن تصویر"""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, 
                 num_classes: int = 10, emb_size: int = 128, depth: int = 4, 
                 num_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # استفاده از بردار خروجی توکن کلاس
        return self.classifier(cls_token_final)

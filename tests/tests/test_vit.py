import torch
import pytest
from src.model import VisionTransformer, PatchEmbedding, MultiHeadSelfAttention
from src.pruner import MagnitudePruner

def test_patch_embedding_dimensions():
    """تست صحت ابعاد پچ‌ها: تصویر ۳۲ در ۳۲ با پچ ۴ باید ۶۴ پچ + ۱ توکن CLS تولید کند"""
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    patch_embed = PatchEmbedding(in_channels=3, patch_size=4, emb_size=64, img_size=32)
    out = patch_embed(x)
    
    # (32 / 4)**2 = 64 + 1 (CLS Token) = 65
    assert out.shape == (batch_size, 65, 64), f"Expected shape (2, 65, 64), but got {out.shape}"

def test_multi_head_attention_shape():
    """تست صحت برابری ابعاد ورودی و خروجی در مکانیسم خودتوجهی"""
    batch_size = 2
    seq_len = 65
    emb_size = 64
    num_heads = 4
    
    x = torch.randn(batch_size, seq_len, emb_size)
    attn = MultiHeadSelfAttention(emb_size=emb_size, num_heads=num_heads)
    out = attn(x)
    
    assert out.shape == (batch_size, seq_len, emb_size), "Attention mechanism changed sequence dimensions!"

def test_full_vit_forward_pass():
    """تست خروجی نهایی مدل برای ۱۰ کلاس دیتاست CIFAR-10"""
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10, emb_size=64, depth=2, num_heads=2)
    
    output = model(dummy_input)
    assert output.shape == (batch_size, 10), f"Expected logits of shape (4, 10), but got {output.shape}"

def test_magnitude_pruning_sparsity():
    """تست ریاضیاتی صحت هرس کردن ۵۰ درصد از کوچک‌ترین وزن‌ها"""
    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10, emb_size=64, depth=2, num_heads=2)
    pruner = MagnitudePruner(model)
    
    # درخواست هرس ۵۰ درصد وزن‌ها
    sparsity = pruner.apply_unstructured_pruning(amount=0.50)
    
    # اسپارسیتی باید تقریباً نزدیک به ۵۰ درصد باشد
    assert 48.0 <= sparsity <= 52.0, f"Expected ~50% sparsity, but got {sparsity:.2f}%"

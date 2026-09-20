import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from src.model import VisionTransformer
from src.dataset import get_dataloaders
from src.pruner import MagnitudePruner

def evaluate(model, test_loader, device):
    """محاسبه درصد دقت مدل و زمان پاسخ‌دهی (Latency) به ازای هر سمپل"""
    model.eval()
    correct, total = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    latency_ms = (time.time() - start_time) * 1000 / len(test_loader.dataset)
    acc = 100.0 * correct / total
    return acc, latency_ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}\n")

    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = VisionTransformer(img_size=32, patch_size=4, emb_size=128, depth=4, num_heads=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("=== Step 1: Training Base Vision Transformer ===")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    base_acc, base_lat = evaluate(model, test_loader, device)

    # بنچمارک نرخ‌های مختلف هرس وزن‌ها
    print("\n=== Step 2: Running Magnitude Pruning Benchmark ===")
    results = [["Baseline (Dense)", 0.0, f"{base_acc:.2f}%", f"{base_lat:.3f} ms"]]
    prune_rates = [0.2, 0.4, 0.6, 0.8]

    for rate in prune_rates:
        cloned_model = copy.deepcopy(model)
        pruner = MagnitudePruner(cloned_model)
        sparsity = pruner.apply_unstructured_pruning(amount=rate)
        acc, lat = evaluate(cloned_model, test_loader, device)
        results.append([f"Pruned ({int(rate*100)}%)", f"{sparsity:.1f}%", f"{acc:.2f}%", f"{lat:.3f} ms"])

    print("\n" + tabulate(results, headers=["Configuration", "Sparsity", "Top-1 Accuracy", "Latency/Sample"], tablefmt="grid"))

if __name__ == "__main__":
    main()

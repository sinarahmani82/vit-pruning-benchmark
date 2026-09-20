import torch
import torch.nn as nn

class MagnitudePruner:
    """الگوریتم هرس وزن‌ها بر مبنای کوچک‌ترین قدرمطلق مقادیر"""
    def __init__(self, model: nn.Module):
        self.model = model

    def apply_unstructured_pruning(self, amount: float) -> float:
        """
        مقدار amount درصدی از کوچک‌ترین وزن‌ها را صفر می‌کند (مثلاً 0.3 یعنی 30 درصد).
        """
        all_weights = []
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                all_weights.append(module.weight.data.abs().view(-1))

        if not all_weights:
            return 0.0

        all_weights_tensor = torch.cat(all_weights)
        k = int(amount * all_weights_tensor.numel())
        if k == 0:
            return 0.0

        threshold, _ = torch.kthvalue(all_weights_tensor, k)

        total_zeros = 0
        total_elements = 0

        with torch.no_grad():
            for _, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    mask = module.weight.data.abs() > threshold
                    module.weight.data.mul_(mask)
                    total_zeros += (module.weight.data == 0).sum().item()
                    total_elements += module.weight.data.numel()

        sparsity = (total_zeros / total_elements) * 100.0
        return sparsity

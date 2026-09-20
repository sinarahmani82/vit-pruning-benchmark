### 📂 Project Structure

```text
vit-pruning-benchmark/
│
├── configs/
│   └── config.yaml          # Hyperparameter configurations
├── src/
│   ├── __init__.py
│   ├── dataset.py           # CIFAR-10 data pipeline & transforms
│   ├── model.py             # Vision Transformer architecture from scratch
│   ├── pruner.py            # Magnitude-based weight pruning engine
│   └── evaluate.py          # Latency & accuracy benchmarking
├── tests/
│   ├── __init__.py
│   └── test_vit.py          # Tensor shape & attention unit tests
├── requirements.txt         # Project dependencies
├── train.py                 # Main execution & benchmarking pipeline
└── README.md                # Research report & findings
```

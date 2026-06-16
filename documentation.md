# Technical Documentation — CIFAR-10 ResNet-18 Classifier

## 1. Problem Statement

Classify 32×32 RGB images into one of 10 categories:  
`plane · car · bird · cat · deer · dog · frog · horse · ship · truck`

Dataset: **CIFAR-10** — 50,000 training images, 10,000 test images, balanced (5,000 per class).

---

## 2. Architecture

### 2.1 Backbone — ResNet-18

ResNet-18 is a residual convolutional network with 18 weight layers. It was originally designed for 224×224 ImageNet images, but works well on CIFAR-10 after fine-tuning because its convolutional feature extractor generalises across natural image domains.

```
Input (3×32×32)
  └─ Conv1 7×7, stride 2 → BN → ReLU → MaxPool
  └─ Layer1: 2× BasicBlock (64 channels)
  └─ Layer2: 2× BasicBlock (128 channels)
  └─ Layer3: 2× BasicBlock (256 channels)
  └─ Layer4: 2× BasicBlock (512 channels)
  └─ AdaptiveAvgPool → Flatten
  └─ FC: 512 → 10  ← replaced head
```

Total parameters: ~11.2M  
All layers are fine-tuned (no layers are frozen).

### 2.2 Transfer Learning Strategy

- Weights are initialised from **ImageNet-1k** (`ResNet18_Weights.DEFAULT`).
- Only the final `fc` layer is replaced; all convolutional layers start from pretrained weights.
- Fine-tuning all layers (rather than freezing the backbone) works well on CIFAR-10 because the dataset is large enough to avoid catastrophic forgetting.

---

## 3. Data Pipeline

### 3.1 Normalisation

Each channel is normalised using CIFAR-10's precomputed statistics:

| Channel | Mean   | Std    |
|---------|--------|--------|
| R       | 0.4914 | 0.2023 |
| G       | 0.4822 | 0.1994 |
| B       | 0.4465 | 0.2010 |

Normalisation is applied to both train and test sets. It was **missing** in the original notebook — a common source of degraded accuracy.

### 3.2 Augmentation (training only)

| Transform             | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `RandomHorizontalFlip`| Invariance to left-right orientation                 |
| `RandomCrop(32, pad=4)`| Invariance to small translations; adds ~1–2% accuracy|
| `ToTensor`            | Converts PIL image to float tensor in [0,1]          |
| `Normalize`           | Zero-mean unit-variance per channel                  |

No augmentation is applied at test time — the test set must reflect real-world distribution.

---

## 4. Training Configuration

| Hyperparameter    | Value                  | Notes                                          |
|-------------------|------------------------|------------------------------------------------|
| Optimiser         | Adam                   | `lr=1e-3`, default betas                       |
| LR Scheduler      | CosineAnnealingLR      | `T_max=NUM_EPOCHS`; decays LR smoothly to ~0   |
| Loss              | CrossEntropyLoss       | Standard multi-class classification loss       |
| Batch size        | 64                     | Good GPU utilisation without OOM on 4GB VRAM   |
| Epochs            | 10                     | Sufficient for ~87–90% accuracy                |

### Why CosineAnnealingLR?

The cosine schedule reduces the learning rate from `lr` to ~0 following a cosine curve, allowing the model to escape local minima early and fine-tune into a flatter minimum later. It consistently outperforms a fixed LR by 1–2% on CIFAR-10.

---

## 5. Bugs Fixed from Original Notebook

| # | Location | Original Issue | Fix Applied |
|---|----------|---------------|-------------|
| 1 | Top of notebook | Loaded Iris dataset and ran `LinearRegression` — completely irrelevant | Removed entirely |
| 2 | `transforms.Compose` | Missing `Normalize` — model receives un-normalised inputs | Added `Normalize(CIFAR10_MEAN, CIFAR10_STD)` to both transforms |
| 3 | Data loading | No `testloader` — only `trainloader` was created | Added `test_dataset` + `test_loader` using `train=False` |
| 4 | Model init | `pretrained=True` — deprecated in PyTorch ≥ 2.0, raises a warning | Changed to `weights=ResNet18_Weights.DEFAULT` |
| 5 | Training loop | `losses.append(running_loss)` — appended raw sum, not per-batch average | Changed to `avg_train_loss = running_loss / len(train_loader)` |
| 6 | Evaluation | Accuracy measured on `trainloader` — meaningless (model sees training data) | Changed to `test_loader` |
| 7 | Evaluation | No validation tracking during training — only final accuracy | Added per-epoch `val_loss` + `val_acc` logged to `history` dict |
| 8 | Evaluation | No classification report or confusion matrix | Added `sklearn.metrics.classification_report` + seaborn heatmap |
| 9 | Inference | No model reload from disk — saving was never tested end-to-end | Added reload block using `load_state_dict(torch.load(...))` |
| 10 | Augmentation | Only `RandomHorizontalFlip` — minimal augmentation | Added `RandomCrop(32, padding=4)` |

---

## 6. Evaluation Metrics

### 6.1 Accuracy
Overall fraction of correct predictions on the 10,000-sample test set.

### 6.2 Classification Report
Per-class **precision**, **recall**, and **F1-score**:

- **Precision** = TP / (TP + FP) — how many predicted positives are actually positive
- **Recall** = TP / (TP + FN) — how many actual positives were caught
- **F1** = harmonic mean of precision and recall

### 6.3 Confusion Matrix
10×10 heatmap where cell (i, j) = number of samples from class i predicted as class j. Diagonal = correct predictions. Off-diagonal = confusion between similar classes (e.g. cat/dog, car/truck).

---

## 7. Output Files

| File                      | Description                                        |
|---------------------------|----------------------------------------------------|
| `resnet18_cifar10.pth`    | Saved `model.state_dict()` — load with `torch.load` |
| `sample_images.png`       | Grid of 16 augmented training images               |
| `training_curves.png`     | Train loss, val loss, val accuracy per epoch       |
| `confusion_matrix.png`    | 10×10 test-set confusion heatmap                   |
| `inference_samples.png`   | 16 test images with predicted vs actual labels     |

---

## 8. Extending This Project

- **Stronger augmentation**: add `ColorJitter`, `RandomGrayscale`, or `AutoAugment(policy=CIFAR10)` for +1–2% accuracy.
- **Better backbone**: swap `resnet18` → `resnet50` or `efficientnet_b0` for ~91–93%.
- **Mixed precision**: wrap training loop with `torch.cuda.amp.autocast()` + `GradScaler` for 2× speedup on modern GPUs.
- **Early stopping**: track best val accuracy and save checkpoint only when improved.

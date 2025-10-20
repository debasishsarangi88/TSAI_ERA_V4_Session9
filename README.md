# ImageNet Training with ResNet50 from Scratch

This project implements ImageNet training for ResNet50 from scratch, targeting 75% top-1 accuracy within a $25 budget.

## 🎯 Project Goals

- Train ResNet50 from scratch on ImageNet-1K
- Achieve 75% top-1 accuracy
- Stay within $25 budget constraint
- Deploy to EC2 and upload to Hugging Face

## 📁 Project Structure

```
Session9/
├── imagenet_training_resnet50.ipynb  # Main training notebook
├── model.py                          # ResNet50 architecture
├── inference.py                      # Model loading & prediction
├── app.py                           # Gradio web interface
├── upload_to_hf.py                  # HF Hub upload script
├── config.json                      # Model configuration
├── requirements.txt                 # Python dependencies
├── best_model.pth                  # Trained model weights
├── README.md                        # This file
└── .gitignore                       # Git ignore file
```

## 🚀 Hugging Face Deployment

All files needed for Hugging Face deployment are included in the main directory:

- **Model Architecture**: Complete ResNet50 implementation (`model.py`)
- **Inference Code**: Model loading and prediction functions (`inference.py`)
- **Gradio App**: Web interface for Hugging Face Spaces (`app.py`)
- **Upload Scripts**: Automated deployment to HF Hub (`upload_to_hf.py`)
- **Configuration**: Model settings and dependencies

### Quick Deployment
```bash
python upload_to_hf.py --username YOUR_USERNAME --repo_name resnet50-imagenet
```


## 💰 Budget Optimization

### Cost Breakdown

| Instance Type | Hourly Cost | Training Time | Total Cost | Status |
|---------------|-------------|---------------|------------|---------|
| g4dn.xlarge   | $0.526      | 24 hours     | $12.62     | ✅ Within budget |
| g4dn.2xlarge  | $0.752      | 18 hours     | $13.54     | ✅ Within budget |
| g4dn.4xlarge  | $1.204      | 12 hours     | $14.45     | ✅ Within budget |
| p3.2xlarge    | $3.06       | 8 hours      | $24.48     | ✅ Within budget |

### Optimization Strategies

- **Mixed Precision Training**: Use FP16 to reduce memory usage
- **Early Stopping**: Stop when 75% accuracy is reached
- **Efficient Data Loading**: Optimized data pipeline
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup

## 🏗️ Architecture

### ResNet50 Implementation

- **BasicBlock**: For ResNet18/34
- **Bottleneck**: For ResNet50/101/152
- **ResNet**: Main architecture with configurable layers
- **Parameters**: ~25.6M parameters
- **Model Size**: ~102 MB

### Training Configuration

```python
config = {
    'epochs': 90,
    'learning_rate': 0.1,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'batch_size': 32,
    'mixed_precision': True,
    'warmup_epochs': 5,
    'cosine_annealing': True
}
```

## 📊 Expected Results

Based on the [Stanford DAWN Benchmark](https://dawnd9.sites.stanford.edu/imagenet-training):

- **Target Accuracy**: 75% top-1 accuracy
- **Training Time**: 8-18 hours (depending on instance)
- **Cost**: $12-24 (well within $25 budget)
- **Model Size**: ~102 MB

## 🔧 Technical Details

### Data Augmentation

- Random resized crop (224x224)
- Random horizontal flip
- Color jittering
- Random rotation
- Normalization (ImageNet stats)

### Training Optimizations

- **Mixed Precision**: FP16 training for speed
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Label Smoothing**: 0.1 smoothing factor
- **Weight Decay**: L2 regularization

### Model Checkpointing

- Save best model based on validation accuracy
- Regular checkpoints every 10 epochs
- Final model export for Hugging Face

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size

## 📦 Production Data Loading (ImageNet‑1K)

When `PRODUCTION_MODE=True`, the notebook trains ResNet50 on ImageNet‑1K. The dataloader is robust and supports two common layouts:

### 1) Official torchvision ImageNet layout (preferred)
- Root: `./imagenet/`
- Structure required by `torchvision.datasets.ImageNet`:
  - `imagenet/train/<class_name>/*.JPEG`
  - `imagenet/val/<class_name>/*.JPEG`
  - Includes the ImageNet metadata index files expected by torchvision.

If this layout is detected, you will see:
```
✅ Loaded torchvision.datasets.ImageNet
```

### 2) Generic class‑folder layout (automatic fallback)
If the official index files are missing, the loader automatically falls back to `torchvision.datasets.ImageFolder` with the same folder structure:
```
imagenet/train/<class_name>/*.JPEG
imagenet/val/<class_name>/*.JPEG
```
In this case you will see:
```
⚠️ torchvision.datasets.ImageNet failed: <reason>
   Falling back to torchvision.datasets.ImageFolder (expects class subfolders)...
✅ Loaded torchvision.datasets.ImageFolder
```

### Changing the dataset path
- Default path in the notebook is `./imagenet/`.
- On EC2, update the call in `get_imagenet_dataset()` to your mount point, e.g. `/mnt/imagenet` or `/efs/imagenet`.

### Transforms used (production)
- Train: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, color jitter, rotation, ImageNet normalization.
- Val: `Resize(256)`, `CenterCrop(224)`, ImageNet normalization.

### Quick production checklist
- Set `TESTING_MODE=False`, `PRODUCTION_MODE=True`.
- Ensure dataset present at your chosen `data_path` with train/val class folders.
- Recommended storage: fast EBS (gp3/io2) or local NVMe; ~150–200 GB free.
- If you hit OOM, lower `batch_size` from 64 → 32 or 16; keep FP16 on.

   - Use gradient accumulation
   - Enable mixed precision

2. **Slow Training**:
   - Check GPU utilization
   - Optimize data loading
   - Use faster instance type

3. **Poor Accuracy**:
   - Increase training time
   - Adjust learning rate
   - Check data preprocessing

### Performance Monitoring

- Monitor GPU utilization: `nvidia-smi`
- Track training metrics with Weights & Biases
- Use early stopping to save costs

## 📚 References

- [Stanford DAWN Benchmark](https://dawnd9.sites.stanford.edu/imagenet-training)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ImageNet Dataset](https://www.image-net.org/)
- [PyTorch Accelerate](https://huggingface.co/docs/accelerate/)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

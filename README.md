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
├── setup_ec2.sh                      # EC2 deployment script
├── upload_to_hf.py                   # Hugging Face upload script
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Test on Google Colab

1. Open `imagenet_training_resnet50.ipynb` in Google Colab
2. Run all cells to test the training pipeline
3. The notebook uses CIFAR-100 for testing (similar structure to ImageNet)

### 2. Deploy to EC2

1. Launch an EC2 instance:
   - **Recommended**: `g4dn.2xlarge` (18 hours ≈ $13.54)
   - **Alternative**: `g4dn.4xlarge` (12 hours ≈ $14.45)
   - **Premium**: `p3.2xlarge` (8 hours ≈ $24.48)

2. Connect to your instance and run:
   ```bash
   # Copy the setup script
   scp setup_ec2.sh ec2-user@your-instance-ip:~/
   
   # Connect to instance
   ssh ec2-user@your-instance-ip
   
   # Run setup
   bash setup_ec2.sh
   ```

3. Download ImageNet dataset:
   - Register at [ImageNet](https://www.image-net.org/)
   - Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`
   - Extract to `./imagenet/` directory

4. Start training:
   ```bash
   ./run_training.sh
   ```

### 3. Upload to Hugging Face

1. Get your Hugging Face token:
   - Go to [HF Settings](https://huggingface.co/settings/tokens)
   - Create a new token

2. Set environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. Update `upload_to_hf.py` with your username

4. Run upload:
   ```bash
   python upload_to_hf.py
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

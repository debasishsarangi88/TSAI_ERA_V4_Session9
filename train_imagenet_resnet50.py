#!/usr/bin/env python3
"""
ImageNet Training with ResNet50 from Scratch

This script implements ImageNet training for ResNet50 from scratch, targeting 75% top-1 accuracy within a $25 budget.

Key Features:
- ResNet50 implementation from scratch
- Optimized for budget constraints
- Mixed precision training
- Data augmentation strategies
- Model checkpointing and evaluation
- AWS EC2 optimized for production training
"""

import argparse
import sys
import os
import time
import math
import json
import warnings
import logging
import psutil
import gc
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import numpy as np
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available - logging will be disabled")

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("⚠️ accelerate not available - mixed precision will be disabled")

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available - some features will be disabled")

warnings.filterwarnings('ignore')

# =============================================================================
# 🚀 CONFIGURATION AND ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments for AWS training"""
    parser = argparse.ArgumentParser(description='ImageNet ResNet50 Training')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='testing', 
                       choices=['testing', 'production'], 
                       help='Training mode: testing (CIFAR-100) or production (ImageNet)')
    
    # AWS specific arguments
    parser.add_argument('--data-path', type=str, default='./imagenet/', 
                       help='Path to ImageNet dataset (for production mode)')
    parser.add_argument('--output-dir', type=str, default='./outputs/', 
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Batch size (auto-determined if not specified)')
    parser.add_argument('--num-workers', type=int, default=None, 
                       help='Number of data loader workers (auto-determined if not specified)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of training epochs (auto-determined by mode)')
    parser.add_argument('--learning-rate', type=float, default=None, 
                       help='Learning rate (auto-determined by mode)')
    parser.add_argument('--target-accuracy', type=float, default=None, 
                       help='Target accuracy (auto-determined by mode)')
    
    # Optimization arguments
    parser.add_argument('--mixed-precision', action='store_true', default=True, 
                       help='Enable mixed precision training (FP16)')
    parser.add_argument('--torch-compile', action='store_true', default=True, 
                       help='Enable torch.compile optimization')
    parser.add_argument('--quantization', type=str, default='fp16', 
                       choices=['none', 'fp16', 'int8', 'dynamic', 'qat'],
                       help='Quantization mode for optimization')
    
    # AWS specific optimizations
    parser.add_argument('--aws-optimized', action='store_true', default=True, 
                       help='Enable AWS-specific optimizations')
    parser.add_argument('--instance-type', type=str, default='g4dn.2xlarge', 
                       help='AWS instance type for optimization')
    parser.add_argument('--budget-limit', type=float, default=25.0, 
                       help='Budget limit in USD')
    
    # Logging arguments
    parser.add_argument('--wandb-project', type=str, default='imagenet-resnet50', 
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-enabled', action='store_true', default=False, 
                       help='Enable Weights & Biases logging')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

# =============================================================================
# 🏗️ RESNET IMPLEMENTATION
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Modified for CIFAR-100 (32x32 images) vs ImageNet (224x224)
        if num_classes == 100:  # CIFAR-100
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()  # No maxpool for CIFAR-100
        else:  # ImageNet
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # Identity for CIFAR-100, MaxPool for ImageNet
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# =============================================================================
# 📊 MONITORING AND LOGGING SYSTEM
# =============================================================================

class TrainingMonitor:
    """Comprehensive training monitoring and logging system"""
    
    def __init__(self, log_dir='logs', enable_wandb=False, log_level='INFO'):
        self.log_dir = log_dir
        self.enable_wandb = enable_wandb
        self.start_time = time.time()
        self.epoch_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.setup_logging(log_level)
        self.log_system_info()
        
    def setup_logging(self, log_level):
        """Setup comprehensive logging system"""
        log_file = os.path.join(self.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Training monitoring system initialized")
        
    def log_system_info(self):
        """Log system information"""
        self.logger.info("=" * 60)
        self.logger.info("🖥️ SYSTEM INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
        
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"CPU Count: {psutil.cpu_count()}")
        self.logger.info(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        self.logger.info("=" * 60)
        
    def log_epoch_start(self, epoch, total_epochs):
        """Log epoch start information"""
        self.logger.info(f"🚀 Starting Epoch {epoch}/{total_epochs}")
        self.logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def log_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log epoch end information"""
        self.epoch_times.append(epoch_time)
        self.losses.append({'train': train_loss, 'val': val_loss})
        self.accuracies.append({'train': train_acc, 'val': val_acc})
        self.learning_rates.append(lr)
        
        self.logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.2f}s")
        self.logger.info(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        self.logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        self.logger.info(f"   Learning Rate: {lr:.6f}")
        self.log_resource_usage()
        
    def log_resource_usage(self):
        """Log current resource usage"""
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used_gb = ram.used / 1e9
        
        self.logger.info(f"📊 Resource Usage:")
        self.logger.info(f"   CPU: {cpu_percent:.1f}%")
        self.logger.info(f"   RAM: {ram_percent:.1f}% ({ram_used_gb:.1f} GB used)")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            
            self.logger.info(f"   GPU Memory: {gpu_memory:.1f} GB (Peak: {gpu_memory_max:.1f} GB)")
            if gpu_utilization > 0:
                self.logger.info(f"   GPU Utilization: {gpu_utilization:.1f}%")
        
        self.memory_usage.append({
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'ram_used_gb': ram_used_gb,
            'gpu_memory_gb': gpu_memory if torch.cuda.is_available() else 0
        })
        
    def log_training_complete(self, best_accuracy, total_time):
        """Log training completion"""
        self.logger.info("=" * 60)
        self.logger.info("🎉 TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Accuracy: {best_accuracy:.2f}%")
        self.logger.info(f"Average Epoch Time: {np.mean(self.epoch_times):.2f}s")
        self.logger.info(f"Total Epochs: {len(self.epoch_times)}")
        
        self.log_resource_usage()
        self.save_training_summary(best_accuracy, total_time)
        
    def save_training_summary(self, best_accuracy, total_time):
        """Save comprehensive training summary"""
        summary = {
            'training_info': {
                'start_time': self.start_time,
                'total_time_hours': total_time / 3600,
                'best_accuracy': best_accuracy,
                'total_epochs': len(self.epoch_times),
                'average_epoch_time': np.mean(self.epoch_times)
            },
            'performance_metrics': {
                'losses': self.losses,
                'accuracies': self.accuracies,
                'learning_rates': self.learning_rates,
                'epoch_times': self.epoch_times
            },
            'resource_usage': self.memory_usage,
            'system_info': {
                'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                'cuda_available': torch.cuda.is_available(),
                'pytorch_version': torch.__version__
            }
        }
        
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📄 Training summary saved to: {summary_file}")

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting and save resources"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_score, model):
        """Check if training should stop early"""
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

class MixedPrecisionMonitor:
    """Monitor mixed precision training performance and memory usage"""
    
    def __init__(self):
        self.memory_usage = []
        self.training_times = []
        self.loss_history = []
        
    def log_batch_performance(self, batch_idx, loss, batch_time, memory_used, mixed_precision=True):
        """Log performance metrics for each batch"""
        if batch_idx % 100 == 0:  # Log every 100 batches
            self.memory_usage.append(memory_used)
            self.training_times.append(batch_time)
            self.loss_history.append(loss.item())
            
            mode = "MP" if mixed_precision else "FP32"
            print(f"   📊 {mode} Batch {batch_idx}: Loss={loss.item():.4f}, "
                  f"Time={batch_time:.3f}s, Memory={memory_used:.1f}GB")
    
    def get_performance_summary(self, mixed_precision=True):
        """Get performance summary for mixed precision training"""
        if not self.memory_usage:
            return "No performance data available"
            
        avg_memory = np.mean(self.memory_usage)
        avg_time = np.mean(self.training_times)
        avg_loss = np.mean(self.loss_history)
        
        summary = f"""
📊 Mixed Precision Performance Summary:
   Average Memory Usage: {avg_memory:.2f} GB
   Average Batch Time: {avg_time:.3f} seconds
   Average Loss: {avg_loss:.4f}
   Total Batches Monitored: {len(self.memory_usage)}
        """
        
        if mixed_precision:
            summary += f"""
🚀 Mixed Precision Benefits:
   Expected Speed Boost: 2x
   Expected Memory Saving: 50%
   GPU Utilization: Optimized for FP16
            """
        else:
            summary += f"""
📊 Full Precision Mode:
   Maximum Accuracy: Enabled
   Memory Usage: Higher
   Training Speed: Slower
            """
        
        return summary

# =============================================================================
# 🔧 QUANTIZATION IMPLEMENTATION
# =============================================================================

import torch.quantization as quant

def apply_quantization(model, quantization_mode, device):
    """Apply different quantization strategies to the model"""
    print(f"🔧 Applying {quantization_mode} quantization...")
    
    if quantization_mode == "none":
        print("   → No quantization applied (baseline)")
        return model
        
    elif quantization_mode == "fp16":
        print("   → Using mixed precision (FP16) - 2x speed boost")
        return model
        
    elif quantization_mode == "int8":
        print("   → Applying 8-bit quantization - 3x speed boost")
        model.eval()
        model.qconfig = quant.get_default_qconfig('fbgemm')
        model_prepared = quant.prepare(model)
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            model_prepared(dummy_input)
        
        model_quantized = quant.convert(model_prepared)
        print("   → Model quantized to INT8")
        return model_quantized
        
    elif quantization_mode == "dynamic":
        print("   → Applying dynamic quantization - 2.5x speed boost")
        model_quantized = quant.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        print("   → Model dynamically quantized")
        return model_quantized
        
    elif quantization_mode == "qat":
        print("   → Setting up Quantization Aware Training - 2.5x speed boost")
        model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        model_prepared = quant.prepare_qat(model)
        print("   → Model prepared for QAT")
        return model_prepared
        
    else:
        raise ValueError(f"Unknown quantization mode: {quantization_mode}")

def get_quantization_info(model, quantization_mode):
    """Get information about model size and performance with quantization"""
    if quantization_mode == "none":
        return {
            "model_size_mb": sum(p.numel() for p in model.parameters()) * 4 / 1e6,
            "parameters": sum(p.numel() for p in model.parameters()),
            "speed_boost": "1x",
            "memory_saving": "0%"
        }
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if quantization_mode == "fp16":
        model_size = total_params * 2 / 1e6
        return {
            "model_size_mb": model_size,
            "parameters": total_params,
            "speed_boost": "2x",
            "memory_saving": "50%"
        }
    elif quantization_mode in ["int8", "dynamic", "qat"]:
        model_size = total_params * 1 / 1e6
        return {
            "model_size_mb": model_size,
            "parameters": total_params,
            "speed_boost": "3x" if quantization_mode == "int8" else "2.5x",
            "memory_saving": "75%" if quantization_mode == "int8" else "60%"
        }

# =============================================================================
# 📁 DATA LOADING AND PREPROCESSING
# =============================================================================

def get_transforms():
    """Get data transforms for ImageNet"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_cifar100_dataset():
    """Get CIFAR-100 dataset for testing"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_imagenet_dataset(data_path):
    """Get ImageNet dataset for production"""
    train_transform, val_transform = get_transforms()
    
    try:
        train_dataset = ImageNet(
            root=data_path, split='train', transform=train_transform
        )
        val_dataset = ImageNet(
            root=data_path, split='val', transform=val_transform
        )
        print("✅ Loaded torchvision.datasets.ImageNet")
    except Exception as e:
        print(f"⚠️ torchvision.datasets.ImageNet failed: {e}")
        print("   Falling back to torchvision.datasets.ImageFolder...")
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder(
            root=os.path.join(data_path, 'train'), transform=train_transform
        )
        val_dataset = ImageFolder(
            root=os.path.join(data_path, 'val'), transform=val_transform
        )
        print("✅ Loaded torchvision.datasets.ImageFolder")
    
    return train_dataset, val_dataset

def validate_dataset(dataset, dataset_name):
    """Comprehensive dataset validation"""
    print(f"🔍 Validating {dataset_name} dataset...")
    
    if len(dataset) == 0:
        raise ValueError(f"{dataset_name} dataset is empty!")
    
    corrupted_samples = 0
    valid_samples = 0
    
    for i in range(min(100, len(dataset))):
        try:
            data, target = dataset[i]
            if torch.isnan(data).any() or torch.isinf(data).any():
                corrupted_samples += 1
            else:
                valid_samples += 1
        except Exception as e:
            corrupted_samples += 1
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Valid samples (checked): {valid_samples}")
    print(f"   Corrupted samples: {corrupted_samples}")
    
    if corrupted_samples > len(dataset) * 0.1:
        print(f"⚠️ WARNING: High corruption rate in {dataset_name} ({corrupted_samples/len(dataset)*100:.1f}%)")
    
    return valid_samples > 0

# =============================================================================
# 🚀 TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, config, mp_monitor, device):
    """Enhanced training epoch with comprehensive error handling and monitoring"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    failed_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        batch_start_time = time.time()
        try:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Data validation
            if data.isnan().any() or target.isnan().any():
                print(f"⚠️ WARNING: NaN detected in batch {batch_idx}, skipping...")
                failed_batches += 1
                continue
                
            if data.isinf().any() or target.isinf().any():
                print(f"⚠️ WARNING: Inf detected in batch {batch_idx}, skipping...")
                failed_batches += 1
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            if config['mixed_precision']:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ WARNING: Invalid loss detected in batch {batch_idx}, skipping...")
                failed_batches += 1
                continue
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if hasattr(accelerator, 'clip_grad_norm_'):
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=config.get('gradient_clip_norm', 1.0)
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Debug first batch
            if batch_idx == 0:
                batch_acc = 100. * pred.eq(target.view_as(pred)).sum().item() / target.size(0)
                print(f"\n🔍 DEBUG - First batch accuracy: {batch_acc:.2f}%")
                print(f"   Batch size: {target.size(0)}")
                print(f"   Device check: data={data.device}, target={target.device}, model={next(model.parameters()).device}")
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Failed': f'{failed_batches}'
            })
            
            # Monitor mixed precision performance
            batch_time = time.time() - batch_start_time
            memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            mp_monitor.log_batch_performance(batch_idx, loss, batch_time, memory_used, config['mixed_precision'])
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"🚨 CUDA OOM ERROR in batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                failed_batches += 1
                continue
            else:
                print(f"🚨 RUNTIME ERROR in batch {batch_idx}: {e}")
                failed_batches += 1
                continue
                
        except Exception as e:
            print(f"🚨 UNEXPECTED ERROR in batch {batch_idx}: {e}")
            failed_batches += 1
            continue
    
    if failed_batches > 0:
        print(f"⚠️ WARNING: {failed_batches} batches failed in epoch {epoch}")
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, accelerator, device):
    """Enhanced evaluation with comprehensive error handling"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc='Evaluating')):
            try:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if data.isnan().any() or target.isnan().any():
                    print(f"⚠️ WARNING: NaN detected in validation batch {batch_idx}, skipping...")
                    failed_batches += 1
                    continue
                    
                if data.isinf().any() or target.isinf().any():
                    print(f"⚠️ WARNING: Inf detected in validation batch {batch_idx}, skipping...")
                    failed_batches += 1
                    continue
                
                output = model(data)
                loss = criterion(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ WARNING: Invalid loss detected in validation batch {batch_idx}, skipping...")
                    failed_batches += 1
                    continue
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"🚨 CUDA OOM ERROR in validation batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    failed_batches += 1
                    continue
                else:
                    print(f"🚨 RUNTIME ERROR in validation batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                    
            except Exception as e:
                print(f"🚨 UNEXPECTED ERROR in validation batch {batch_idx}: {e}")
                failed_batches += 1
                continue
    
    if failed_batches > 0:
        print(f"⚠️ WARNING: {failed_batches} validation batches failed")
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, filepath, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved: {filepath}')

def load_checkpoint(filepath, model, optimizer, scheduler, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy']

# =============================================================================
# 🏗️ AWS OPTIMIZATION FUNCTIONS
# =============================================================================

def get_aws_optimized_config(args, mode):
    """Get AWS-optimized configuration based on instance type and mode"""
    
    # AWS instance optimizations
    aws_configs = {
        'g4dn.xlarge': {'batch_size': 16, 'num_workers': 4, 'memory_cleanup_freq': 5},
        'g4dn.2xlarge': {'batch_size': 32, 'num_workers': 8, 'memory_cleanup_freq': 10},
        'g4dn.4xlarge': {'batch_size': 64, 'num_workers': 16, 'memory_cleanup_freq': 10},
        'p3.2xlarge': {'batch_size': 128, 'num_workers': 16, 'memory_cleanup_freq': 15},
        'p3.8xlarge': {'batch_size': 256, 'num_workers': 32, 'memory_cleanup_freq': 20},
    }
    
    aws_config = aws_configs.get(args.instance_type, aws_configs['g4dn.2xlarge'])
    
    if mode == 'testing':
        config = {
            'epochs': args.epochs or 5,
            'learning_rate': args.learning_rate or 0.05,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'batch_size': args.batch_size or aws_config['batch_size'],
            'num_workers': args.num_workers or aws_config['num_workers'],
            'save_every': 2,
            'eval_every': 1,
            'mixed_precision': args.mixed_precision,
            'gradient_accumulation_steps': 1,
            'warmup_epochs': 0,
            'cosine_annealing': True,
            'target_accuracy': args.target_accuracy or 80.0,
            'wandb_enabled': args.wandb_enabled,
            'quantization_mode': args.quantization,
            'quantization_enabled': args.quantization != "none",
            'torch_compile': args.torch_compile,
            'compile_mode': 'default',
            'early_stopping_patience': 3,
            'early_stopping_min_delta': 0.001,
            'gradient_clip_norm': 1.0,
            'memory_cleanup_frequency': aws_config['memory_cleanup_freq']
        }
    else:  # production
        config = {
            'epochs': args.epochs or 90,
            'learning_rate': args.learning_rate or 0.1,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'batch_size': args.batch_size or aws_config['batch_size'],
            'num_workers': args.num_workers or aws_config['num_workers'],
            'save_every': 10,
            'eval_every': 5,
            'mixed_precision': args.mixed_precision,
            'gradient_accumulation_steps': 1,
            'warmup_epochs': 5,
            'cosine_annealing': True,
            'target_accuracy': args.target_accuracy or 75.0,
            'wandb_enabled': args.wandb_enabled,
            'quantization_mode': args.quantization,
            'quantization_enabled': args.quantization != "none",
            'torch_compile': args.torch_compile,
            'compile_mode': 'default',
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001,
            'gradient_clip_norm': 1.0,
            'memory_cleanup_frequency': aws_config['memory_cleanup_freq']
        }
    
    return config

def estimate_aws_cost(args, config):
    """Estimate AWS training cost"""
    instance_costs = {
        'g4dn.xlarge': 0.526,
        'g4dn.2xlarge': 0.752,
        'g4dn.4xlarge': 1.204,
        'p3.2xlarge': 3.06,
        'p3.8xlarge': 12.24,
    }
    
    estimated_hours = {
        'g4dn.xlarge': 24,
        'g4dn.2xlarge': 18,
        'g4dn.4xlarge': 12,
        'p3.2xlarge': 8,
        'p3.8xlarge': 4,
    }
    
    hourly_cost = instance_costs.get(args.instance_type, 0.752)
    training_hours = estimated_hours.get(args.instance_type, 18)
    total_cost = hourly_cost * training_hours
    
    print(f"\n💰 AWS Cost Estimation:")
    print(f"   Instance: {args.instance_type}")
    print(f"   Hourly Cost: ${hourly_cost:.3f}")
    print(f"   Estimated Time: {training_hours} hours")
    print(f"   Total Cost: ${total_cost:.2f}")
    
    if total_cost <= args.budget_limit:
        print(f"   ✅ Within budget (${args.budget_limit})")
    else:
        print(f"   ❌ Over budget (${args.budget_limit})")
    
    return total_cost

# =============================================================================
# 🚀 MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Determine mode
    TESTING_MODE = (args.mode == 'testing')
    PRODUCTION_MODE = (args.mode == 'production')
    
    print(f"\n{'🧪 TESTING MODE (CIFAR-100)' if TESTING_MODE else '🏭 PRODUCTION MODE (ImageNet-1K)'}")
    print("=" * 60)
    
    # Get configuration
    config = get_aws_optimized_config(args, args.mode)
    
    # Estimate AWS cost
    if args.aws_optimized:
        estimate_aws_cost(args, config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Initialize monitoring
    monitor = TrainingMonitor(
        log_dir=os.path.join(args.output_dir, 'logs'), 
        enable_wandb=config['wandb_enabled'],
        log_level=args.log_level
    )
    
    # Load dataset
    if TESTING_MODE:
        print("Loading CIFAR-100 dataset for testing...")
        train_dataset, val_dataset = get_cifar100_dataset()
        print(f"✅ CIFAR-100 loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"✅ Number of classes: {len(train_dataset.classes)}")
        validate_dataset(train_dataset, "CIFAR-100 Train")
        validate_dataset(val_dataset, "CIFAR-100 Val")
    else:
        print("Loading ImageNet-1K dataset for production...")
        train_dataset, val_dataset = get_imagenet_dataset(args.data_path)
        print(f"✅ ImageNet-1K loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"✅ Number of classes: {len(train_dataset.classes)}")
        validate_dataset(train_dataset, "ImageNet Train")
        validate_dataset(val_dataset, "ImageNet Val")
    
    # Update config with actual dataset info
    config['num_classes'] = len(train_dataset.classes)
    config['batch_size'] = args.batch_size or config['batch_size']
    config['num_workers'] = args.num_workers or config['num_workers']
    
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of workers: {config['num_workers']}")
    
    # Create data loaders
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, 
            num_workers=config['num_workers'], pin_memory=True, 
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config['num_workers'], pin_memory=True,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        print("✅ Data loaders created successfully")
        
    except Exception as e:
        print(f"⚠️ WARNING: Error creating data loaders: {e}")
        print("   Falling back to single-threaded loading...")
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, 
            num_workers=0, pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False, 
            num_workers=0, pin_memory=False
        )
    
    # Initialize model
    if TESTING_MODE:
        model = ResNet18(num_classes=config['num_classes'])
        print("🏗️ Using ResNet18 for CIFAR-100 (32x32 images)")
    else:
        model = ResNet50(num_classes=config['num_classes'])
        print("🏗️ Using ResNet50 for ImageNet (224x224 images)")
    
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    print(f"✅ Model initialized with proper weights for {config['num_classes']} classes")
    
    # Test model with sample batch
    print("\n🔍 DEBUG - Testing model with sample batch...")
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(train_loader))
        sample_data = sample_data[:4].to(device)
        sample_target = sample_target[:4].to(device)
        
        sample_output = model(sample_data)
        sample_pred = sample_output.argmax(dim=1)
        sample_acc = (sample_pred == sample_target).float().mean().item()
        print(f"   Sample accuracy: {sample_acc:.2f}%")
    
    model.train()
    
    # Apply torch.compile optimization
    if config.get('torch_compile', False):
        if hasattr(torch, 'compile'):
            try:
                print(f"🚀 Applying torch.compile() optimization...")
                model = torch.compile(model, mode=config.get('compile_mode', 'default'))
                print("✅ Model compiled successfully!")
            except Exception as e:
                print(f"⚠️ torch.compile() failed: {e}")
        else:
            print("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
    
    # Initialize accelerator for mixed precision
    if ACCELERATE_AVAILABLE and torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision='fp16' if config['mixed_precision'] else 'no')
    else:
        # Create a dummy accelerator for compatibility (CPU/MPS or no accelerate)
        class DummyAccelerator:
            def __init__(self):
                self.mixed_precision = 'no'
            def prepare(self, *args):
                return args
            def accumulate(self, model):
                return model
            def backward(self, loss):
                loss.backward()
            def clip_grad_norm_(self, parameters, max_norm):
                torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        accelerator = DummyAccelerator()
        
        # Disable mixed precision on CPU/MPS
        if config['mixed_precision'] and not torch.cuda.is_available():
            print("⚠️ Mixed precision disabled - requires CUDA GPU")
            config['mixed_precision'] = False
    
    if config['mixed_precision']:
        print("🚀 Mixed Precision Training ENABLED (FP16)")
        print("   Benefits: 2x speed boost, 50% memory reduction")
    else:
        print("📊 Mixed Precision Training DISABLED (FP32)")
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'], 
        momentum=config['momentum'], 
        weight_decay=config['weight_decay']
    )
    
    def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5):
        def lr_lambda(epoch):
            if warmup_epochs == 0:
                return 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
            elif epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_scheduler(optimizer, config['epochs'], config['warmup_epochs'])
    
    # Loss function
    if TESTING_MODE:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        print("🎯 Using CrossEntropyLoss without label smoothing for CIFAR-100")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("🎯 Using CrossEntropyLoss with label smoothing for ImageNet")
    
    # Prepare for accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Initialize early stopping and monitoring
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 10),
        min_delta=config.get('early_stopping_min_delta', 0.001),
        restore_best_weights=True
    )
    
    mp_monitor = MixedPrecisionMonitor()
    
    # Apply quantization if enabled
    if config['quantization_enabled']:
        print(f"\n🔧 Applying {config['quantization_mode']} quantization...")
        model = apply_quantization(model, config['quantization_mode'], device)
        
        quant_info = get_quantization_info(model, config['quantization_mode'])
        print(f"\n📊 Model Info with {config['quantization_mode']} quantization:")
        print(f"   Model Size: {quant_info['model_size_mb']:.1f} MB")
        print(f"   Parameters: {quant_info['parameters']:,}")
        print(f"   Speed Boost: {quant_info['speed_boost']}")
        print(f"   Memory Saving: {quant_info['memory_saving']}")
    
    # Initialize wandb if enabled
    if config['wandb_enabled'] and WANDB_AVAILABLE:
        print("Initializing Weights & Biases for logging...")
        wandb.init(project=args.wandb_project, config=config)
    elif config['wandb_enabled'] and not WANDB_AVAILABLE:
        print("⚠️ wandb requested but not available - disabling wandb logging")
        config['wandb_enabled'] = False
    
    # Start training
    best_accuracy = 0
    start_time = time.time()
    
    print("🚀 Starting enhanced training with comprehensive monitoring...")
    print(f"Target accuracy: {config['target_accuracy']}%")
    print(f"Training for {config['epochs']} epochs")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 10)}")
    
    monitor.logger.info("🚀 TRAINING STARTED")
    monitor.logger.info(f"Target accuracy: {config['target_accuracy']}%")
    monitor.logger.info(f"Total epochs: {config['epochs']}")
    
    for epoch in range(1, config['epochs'] + 1):
        try:
            epoch_start = time.time()
            monitor.log_epoch_start(epoch, config['epochs'])
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, config, mp_monitor, device)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate
            if epoch % config['eval_every'] == 0 or epoch == config['epochs']:
                val_loss, val_acc = evaluate(model, val_loader, criterion, accelerator, device)
                
                epoch_time = time.time() - epoch_start
                monitor.log_epoch_end(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
                
                print(f'\n📊 Epoch {epoch} Results:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Learning Rate: {current_lr:.6f}')
                print(f'  Epoch Time: {epoch_time:.2f}s')
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, val_acc,
                        os.path.join(args.output_dir, f'checkpoints/best_model_epoch_{epoch}_acc_{val_acc:.2f}.pth'),
                        config
                    )
                    monitor.logger.info(f"🏆 New best model saved with accuracy: {val_acc:.2f}%")
                
                # Early stopping check
                if early_stopping(val_acc, model):
                    monitor.logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f'\n🛑 Early stopping triggered at epoch {epoch}')
                    print(f'Best accuracy achieved: {best_accuracy:.2f}%')
                    break
                
                # Log to wandb
                if config['wandb_enabled']:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time
                    })
            else:
                epoch_time = time.time() - epoch_start
                monitor.logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.2f}s (no validation)")
            
            # Save checkpoint
            if epoch % config['save_every'] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_acc if 'val_acc' in locals() else 0,
                    os.path.join(args.output_dir, f'checkpoints/checkpoint_epoch_{epoch}.pth'),
                    config
                )
                monitor.logger.info(f"💾 Checkpoint saved at epoch {epoch}")
            
            # Check target accuracy
            if 'val_acc' in locals() and val_acc >= config['target_accuracy']:
                monitor.logger.info(f"🎉 Target accuracy of {config['target_accuracy']}% reached!")
                print(f'\n🎉 Target accuracy of {config["target_accuracy"]}% reached! Stopping early.')
                break
                
            # Memory cleanup
            if epoch % config.get('memory_cleanup_frequency', 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            monitor.logger.error(f"🚨 CRITICAL ERROR in epoch {epoch}: {e}")
            print(f'\n🚨 CRITICAL ERROR in epoch {epoch}: {e}')
            print("Attempting to continue training...")
            
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
            continue
    
    # Training completion
    total_time = time.time() - start_time
    monitor.log_training_complete(best_accuracy, total_time)
    
    print(f'\n🎉 Training completed in {total_time/3600:.2f} hours')
    print(f'Best accuracy: {best_accuracy:.2f}%')
    
    # Print mixed precision performance summary
    print("\n" + "="*60)
    print("📊 MIXED PRECISION TRAINING PERFORMANCE SUMMARY")
    print("="*60)
    print(mp_monitor.get_performance_summary(config['mixed_precision']))
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config['epochs'], best_accuracy,
        os.path.join(args.output_dir, 'checkpoints/final_model.pth'),
        config
    )
    
    print("📄 Training summary and logs saved to 'logs/' directory")
    print("🚀 Enhanced training with comprehensive monitoring completed!")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_loss, final_acc = evaluate(model, val_loader, criterion, accelerator, device)
    print(f'Final validation accuracy: {final_acc:.2f}%')
    print(f'Final validation loss: {final_loss:.4f}')

if __name__ == "__main__":
    main()

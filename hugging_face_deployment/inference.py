"""
Inference script for ResNet50 ImageNet model
This script handles model loading and prediction for Hugging Face deployment
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import ResNet50, IMAGENET_CLASSES
import os


class ImageNetPredictor:
    def __init__(self, model_path, device='cpu', num_classes=1000):
        """
        Initialize the ImageNet predictor
        
        Args:
            model_path (str): Path to the trained model (.pth file)
            device (str): Device to run inference on ('cpu' or 'cuda')
            num_classes (int): Number of classes (1000 for ImageNet)
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load class labels
        if num_classes <= len(IMAGENET_CLASSES):
            self.class_labels = IMAGENET_CLASSES[:num_classes]
        else:
            # If we need more classes than available, create generic labels
            self.class_labels = IMAGENET_CLASSES + [f"class_{i}" for i in range(len(IMAGENET_CLASSES), num_classes)]
        
    def _load_model(self, model_path):
        """Load the trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model
        model = ResNet50(num_classes=self.num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image, top_k=5):
        """
        Make prediction on image
        
        Args:
            image: PIL Image, numpy array, or image path
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results with class names and probabilities
        """
        # Handle image path
        if isinstance(image, str):
            image = Image.open(image)
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Convert to numpy
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        results = []
        for i in range(top_k):
            class_idx = top_indices[i]
            # Handle case where class_idx might be out of bounds
            if class_idx < len(self.class_labels):
                class_name = self.class_labels[class_idx]
            else:
                class_name = f"class_{class_idx}"
            confidence = top_probs[i]
            
            results.append({
                'class_name': class_name,
                'class_id': int(class_idx),
                'confidence': float(confidence)
            })
        
        return {
            'predictions': results,
            'top_prediction': results[0]
        }
    
    def predict_batch(self, images, top_k=5):
        """
        Make predictions on a batch of images
        
        Args:
            images: List of PIL Images, numpy arrays, or image paths
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of prediction results for each image
        """
        results = []
        for image in images:
            result = self.predict(image, top_k)
            results.append(result)
        
        return results


def load_predictor(model_path, device='cpu'):
    """
    Convenience function to load predictor
    
    Args:
        model_path (str): Path to the trained model
        device (str): Device to run inference on
        
    Returns:
        ImageNetPredictor: Loaded predictor
    """
    return ImageNetPredictor(model_path, device)


# Example usage
if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_model.pth"  # Update with actual model path
    
    # Load predictor
    predictor = load_predictor(model_path, device='cpu')
    
    # Example prediction
    # image_path = "path/to/your/image.jpg"
    # result = predictor.predict(image_path)
    # print(f"Predicted class: {result['top_prediction']['class_name']}")
    # print(f"Confidence: {result['top_prediction']['confidence']:.3f}")
    
    print("ImageNet predictor loaded successfully!")
    print(f"Model supports {len(predictor.class_labels)} classes")
    print(f"Device: {predictor.device}")

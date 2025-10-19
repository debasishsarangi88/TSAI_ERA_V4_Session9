"""
Gradio App for ResNet50 ImageNet Classification
This is the main app file for Hugging Face Spaces deployment
"""

import gradio as gr
import torch
from PIL import Image
import json
import os
from inference import ImageNetPredictor
import warnings
warnings.filterwarnings('ignore')

# Global variables
predictor = None
model_loaded = False

def load_model():
    """Load the model and initialize predictor"""
    global predictor, model_loaded
    
    try:
        # Try to find the model file
        model_paths = [
            "best_model.pth",
            "final_model.pth", 
            "checkpoints/best_model.pth",
            "checkpoints/final_model.pth",
            "model.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("No model file found. Please ensure you have a .pth file in the repository.")
        
        # Load predictor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = ImageNetPredictor(model_path, device=device)
        model_loaded = True
        
        return f"✅ Model loaded successfully from {model_path} on {device}"
        
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def classify_image(image):
    """Classify the uploaded image"""
    global predictor, model_loaded
    
    if not model_loaded or predictor is None:
        return "❌ Model not loaded. Please load the model first.", None
    
    try:
        # Make prediction
        result = predictor.predict(image, top_k=5)
        
        # Format results
        predictions = result['predictions']
        top_pred = result['top_prediction']
        
        # Create results text
        results_text = f"**Top Prediction:** {top_pred['class_name']}\n"
        results_text += f"**Confidence:** {top_pred['confidence']:.3f}\n\n"
        results_text += "**Top 5 Predictions:**\n"
        
        for i, pred in enumerate(predictions, 1):
            results_text += f"{i}. {pred['class_name']} ({pred['confidence']:.3f})\n"
        
        return results_text, predictions
        
    except Exception as e:
        return f"❌ Error during classification: {str(e)}", None

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="ResNet50 ImageNet Classifier",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        """
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🖼️ ResNet50 ImageNet Classifier</h1>
            <p>Upload an image to classify it using our trained ResNet50 model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model loading section
                gr.Markdown("## 🔧 Model Status")
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded",
                    interactive=False
                )
                
                # Image upload
                gr.Markdown("## 📸 Upload Image")
                image_input = gr.Image(
                    type="pil",
                    label="Upload an image",
                    height=300
                )
                
                classify_btn = gr.Button("Classify Image", variant="secondary")
                
            with gr.Column(scale=2):
                # Results
                gr.Markdown("## 🎯 Classification Results")
                results_output = gr.Markdown(
                    value="Upload an image and click 'Classify Image' to see results",
                    label="Results"
                )
                
                # Detailed results table
                with gr.Accordion("📊 Detailed Results", open=False):
                    results_table = gr.Dataframe(
                        headers=["Rank", "Class Name", "Confidence"],
                        datatype=["number", "str", "number"],
                        label="Top 5 Predictions"
                    )
        
        # Example images
        gr.Markdown("## 🖼️ Example Images")
        gr.Examples(
            examples=[
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"],
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/256px-Cat_August_2010-4.jpg"],
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/256px-PNG_transparency_demonstration_1.png"]
            ],
            inputs=image_input,
            label="Click on an example to load it"
        )
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=model_status
        )
        
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[results_output, results_table]
        )
        
        # Auto-load model on startup
        app.load(
            fn=load_model,
            outputs=model_status
        )
    
    return app

def main():
    """Main function to run the app"""
    app = create_interface()
    return app

if __name__ == "__main__":
    # Create and launch the app
    app = main()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

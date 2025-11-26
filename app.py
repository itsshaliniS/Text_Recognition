"""
Flask Backend for OCR System
Provides REST API for text recognition from images
"""

import os
import sys
import pickle
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import base64

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CRNN
from src.utils import preprocess_image, ctc_decode, get_default_charset


# Initialize Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Global variables for model and charset
model = None
charset = None
device = None
config = None


def load_model_from_pickle(model_path='models/best_model.pkl'):
    """
    Load trained model from pickle file
    
    Args:
        model_path (str): Path to model pickle file
        
    Returns:
        tuple: (model, charset, device, config)
    """
    global model, charset, device, config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Using default charset and untrained model for demo purposes")
        
        # Use default charset
        charset = get_default_charset()
        num_classes = charset.get_num_classes()
        
        # Create untrained model
        model = CRNN(num_classes=num_classes, hidden_size=256, num_lstm_layers=2)
        model = model.to(device)
        model.eval()
        
        config = {
            'img_height': 32,
            'img_width': 128
        }
        
        return model, charset, device, config
    
    # Load from pickle
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract components
    model_state_dict = checkpoint['model_state_dict']
    num_classes = checkpoint['num_classes']
    charset = checkpoint['charset']
    config = checkpoint.get('config', {})
    
    # Create model
    model = CRNN(
        num_classes=num_classes,
        hidden_size=config.get('hidden_size', 256),
        num_lstm_layers=config.get('num_lstm_layers', 2)
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Number of classes: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, charset, device, config


def predict_text(image):
    """
    Predict text from image
    
    Args:
        image: PIL Image
        
    Returns:
        str: Predicted text
    """
    global model, charset, device, config
    
    if model is None:
        raise ValueError("Model not loaded. Please load model first.")
    
    # Get image dimensions from config
    img_height = config.get('img_height', 32)
    img_width = config.get('img_width', 128)
    
    # Preprocess image
    image_tensor = preprocess_image(image, target_height=img_height, target_width=img_width)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        log_probs = model(image_tensor)
    
    # Decode prediction
    batch_size = image_tensor.size(0)
    input_lengths = torch.full(
        size=(batch_size,),
        fill_value=log_probs.size(0),
        dtype=torch.long
    )
    
    predictions = ctc_decode(log_probs, input_lengths, charset)
    
    return predictions[0] if predictions else ""


@app.route('/')
def index():
    """
    Serve the main HTML page
    """
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """
    Serve static files
    """
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict text from uploaded image
    
    Expected input:
        - file: Image file (multipart/form-data)
        OR
        - image: Base64 encoded image (JSON)
        
    Returns:
        JSON response with predicted text
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get image from request
        image = None
        
        # Try to get from file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Try to get from JSON (base64)
        elif request.is_json:
            data = request.get_json()
            if 'image' in data:
                # Decode base64 image
                image_data = data['image']
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Predict text
        predicted_text = predict_text(image)
        
        # Return result
        return jsonify({
            'success': True,
            'text': predicted_text
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    })


@app.route('/info', methods=['GET'])
def info():
    """
    Get model information
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'num_classes': charset.get_num_classes() if charset else 0,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'config': config
    })


def initialize_app(model_path='models/best_model.pkl'):
    """
    Initialize the application by loading the model
    
    Args:
        model_path (str): Path to model file
    """
    global model, charset, device, config
    
    print("\n" + "="*60)
    print("Initializing OCR Flask Application")
    print("="*60)
    
    # Load model
    model, charset, device, config = load_model_from_pickle(model_path)
    
    print("="*60)
    print("✓ Application initialized successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Initialize app
    initialize_app()
    
    # Run Flask app
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid loading model twice
    )


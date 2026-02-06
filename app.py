import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)

# Constants
MODEL_PATH = 'best_model.pth'
CLASSES_PATH = 'classes.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            return json.load(f)
    else:
        # Defaults based on the Kaggle dataset classes we've seen
        return ["cloudy", "desert", "green_area", "water"]

CLASSES = load_classes()

def load_model():
    # EXACT user model: EfficientNet-B3
    model = models.efficientnet_b3(pretrained=False) # We load local weights
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Prediction accuracy will be random.")
        
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Inference Transform
# Resizing to 128 as the model was trained with RandomResizedCrop(128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        result = {
            'class': CLASSES[predicted.item()],
            'confidence': float(confidence.item()),
            'status': 'success'
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Server starting with classes: {CLASSES}")
    app.run(debug=True, port=5000)

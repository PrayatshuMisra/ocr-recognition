import torch.nn as nn
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)


# Load the model (same as before)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 28, 28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Output: (128, 28, 28)
        # MaxPooling and Dropout
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        # Fully connected layers
        self.fc1 = nn.Linear(128*3*3, 512)  # Adjusted based on input size after pooling
        self.fc2 = nn.Linear(512, 27)  # 27 classes for EMNIST (letters)

    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128*3*3)
        
        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

import os
model = SimpleCNN()
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'improved_pytorch_ocr.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Mapping of class indices to letters (EMNIST Letters is 1-indexed for A-Z)
idx_to_letter = {i: chr(i + 64) for i in range(1, 27)}
idx_to_letter[0] = '?'  # Reserved or blank class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 encoded image from request
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = request.json['image']
        
        # Decode base64 image
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Save original image for debugging (disabled in production)
        # image.save('debug_original.png')
        
        # Invert the image
        image = ImageOps.invert(image)
        # image.save('debug_inverted.png')
        
        # Resize and preprocess
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: transforms.functional.hflip(img)),
            transforms.Lambda(lambda img: transforms.functional.rotate(img, 90)), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            # Get top 3 predictions
            top_predictions = [
                {
                    'letter': idx_to_letter.get(idx.item(), '?'), 
                    'probability': prob.item()
                } 
                for idx, prob in zip(top_indices[0], top_probs[0])
            ]
        
        return jsonify({
            'predictions': top_predictions
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
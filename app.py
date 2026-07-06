import torch.nn as nn
import torch
import torch.nn.functional as F
torch.set_num_threads(1) # CRITICAL: Limits PyTorch to 1 CPU thread to prevent OOM on Render Free Tier

import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import numpy as np

app = Flask(__name__)


class ImprovedCNN(torch.nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128*3*3, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 27) 

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        
        x = x.view(x.size(0), -1)
        
        x = torch.nn.functional.relu(self.bn4(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

import os
model = ImprovedCNN()
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'improved_ocr.pth')
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
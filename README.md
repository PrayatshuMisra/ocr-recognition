# Scriptura | Optical Character Recognition

![Scriptura UI](https://img.shields.io/badge/UI-Glassmorphism-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-ee4c2c?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask)

Scriptura is a modern, elegant web application that utilizes a PyTorch Convolutional Neural Neural Network to perform real-time handwritten letter recognition. Draw any uppercase letter on the interactive canvas, and the AI will analyze the stroke and predict the character with confidence levels.

## Features

- **Advanced CNN:** Built and trained in PyTorch using the EMNIST (Letters) dataset.
- **Real-Time Analysis:** Fast inference backend powered by Flask.
- **Premium UI:** A stunning, responsive Glassmorphism interface featuring elegant typography (Playfair Display, Inter, and Pinyon Script).
- **Interactive Canvas:** Built with Fabric.js, including adjustable stroke weight for precise drawing.
- **Dynamic Feedback:** Smooth animations for prediction confidence bars and error handling.

## Tech Stack

- **Frontend:** HTML5, Vanilla CSS, JavaScript, Fabric.js
- **Backend:** Python, Flask
- **Machine Learning:** PyTorch, TorchVision, Pillow, NumPy

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your machine.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/scriptura.git
   cd scriptura
   ```
2. Create and activate a virtual environment (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On macOS/Linux use: source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
   *(Note: Do not open `index.html` directly from your file explorer; the backend API is required for the AI inference to work.)*

## Model Architecture & Training

The `improved_ocr.pth` model features a deep Convolutional Neural Network (CNN) structure:
- 3 Convolutional Layers (32, 64, 128 filters) with Max Pooling
- 2 Fully Connected Layers with Dropout regularization
- Maps to 26 alphabetic classes (A-Z)

To view the training process or re-train the model, check out the Jupyter notebooks in the `/final` directory.

## 🤝 Contributing
Contributions, issues, and feature requests are always welcome! Feel free to fork the repository and submit pull requests.

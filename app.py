# app.py
# .venv\Scripts\activate
# deactivate
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

# Import your model class
from vgg7_ann import VGG7

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Model Loading and Preparation ---

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model architecture
model = VGG7(num_classes=10).to(device)

# Load the trained model weights
model_path = '/Users/rohit/Desktop/snn_project/vgg7_cifar10_advanced.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: Model weights file not found at {model_path}. The model is untrained.")
model.eval()  # Set the model to evaluation mode

# Define the image transformations (MUST match training normalization)
transform = transforms.Compose([
    transforms.Resize((32, 32)),              # CIFAR-10 images are 32x32
    transforms.ToTensor(),                    # Convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # Match training normalization
])

# Define the CIFAR-10 class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image upload and returns a prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Open the image file
            image = Image.open(file.stream).convert('RGB')

            # Apply transformations and prepare for model
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Make a prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = classes[predicted_idx.item()]

            return jsonify({'prediction': predicted_class})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500)
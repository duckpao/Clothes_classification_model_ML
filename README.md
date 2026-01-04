👕 Clothes Classification Model (Deep Learning + FastAPI)

This project is a web application for clothing image classification using Deep Learning (Transfer Learning with MobileNetV2) and FastAPI.

Users can:

Upload a real-world clothing image (RGB)

Get the predicted clothing category

See the uploaded image together with prediction results on the web interface

🚀 Features

Upload clothing images from local device

Predict clothing category

Display:

Prediction label

Confidence score

Top-3 predictions

Uploaded image preview

🧠 Model & Dataset

Model: MobileNetV2 (Transfer Learning, pretrained on ImageNet)

Dataset: Fashion Product Images Dataset (Kaggle)

Input: RGB images, size 224 × 224

Framework: TensorFlow / Keras

Validation Accuracy: ~93%

⚠️ The trained model file (.h5) is NOT included in this repository.
You must train the model yourself or provide your own trained model.

📂 Project Structure
Clothes_classification_model_ML/
├── app.py                  # FastAPI backend
├── ml.py                   # (optional) ML scripts
├── templates/
│   └── index.html          # Web UI (Jinja2 template)
├── README.md
├── requirements.txt
└── .gitignore

⚙️ 1. System Requirements

Python 3.9 – 3.10 (recommended: 3.10)

pip

(Recommended) GPU for training the model

📦 2. Environment Setup
Step 1: Clone the repository
git clone https://github.com/duckpao/Clothes_classification_model_ML.git
cd Clothes_classification_model_ML

Step 2: Create virtual environment
python -m venv venv

Step 3: Activate virtual environment

Windows

venv\Scripts\activate


macOS / Linux

source venv/bin/activate

Step 4: Install dependencies
pip install -r requirements.txt

🧠 3. Train the Model (REQUIRED)

Since the model file is not included, you must train it first.

Recommended: Train on Kaggle Notebook

Create a Kaggle Notebook

Add dataset:

Fashion Product Images Dataset

Enable:

Accelerator: GPU (P100)

Internet: ON

Train the model using MobileNetV2

Save the trained model:

model.save("fashion_rgb_model.h5")


Download fashion_rgb_model.h5

Place the model file in the project root:
Clothes_classification_model_ML/
└── fashion_rgb_model.h5

🌐 4. Run the FastAPI Web App
Step 1: Start the server
uvicorn app:app --reload

Step 2: Open in browser
http://127.0.0.1:8000

Step 3: Use the app

Upload a clothing image (RGB)

View prediction results and image preview

🎨 5. Web Interface

Clean and simple UI

Displays:

Uploaded image

Predicted class

Confidence score

Top-3 predictions

🧪 6. Important Notes

This project does NOT use Fashion-MNIST

Images are processed as RGB (not grayscale)

Best results when:

Clothing is clearly visible

Image is well-lit

Minimal background noise

🔧 7. Tech Stack

Python

TensorFlow / Keras

FastAPI

Jinja2

HTML / CSS

Kaggle (for training)

🚀 8. Future Improvements

Clothing color recognition

Merge similar classes (e.g., Tops + Tshirts → Topwear)

JSON API for mobile apps

Cloud deployment (Render / VPS)

Bounding box / clothing region detection
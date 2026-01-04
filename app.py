from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("fashion_rgb_model.h5")

class_names = [
    "Dresses",
    "Jeans",
    "Sandals",
    "Tops",
    "Trousers",
    "Tshirts"
]

IMG_SIZE = (224, 224)

# =========================
# PREPROCESS
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # preprocess
    img = preprocess_image(image)
    preds = model.predict(img)[0]

    top_idx = int(np.argmax(preds))
    prediction = class_names[top_idx]
    confidence = float(preds[top_idx])

    # top 3
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(class_names[i], float(preds[i])) for i in top3_idx]

    # convert image to base64 for display
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": f"{confidence:.2%}",
            "top3": top3,
            "image_data": img_base64
        }
    )

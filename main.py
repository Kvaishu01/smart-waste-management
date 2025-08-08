from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = FastAPI()

# Allow frontend to call the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = "backend/waste_classifier.h5"
model = load_model(model_path)
class_names = ['carboard', 'glass', 'metal', 'paper','plastic','trash']  # customize if needed

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((128, 128))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_index = np.argmax(predictions)
        predicted_class = class_names[class_index]
        confidence = float(predictions[class_index])

        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

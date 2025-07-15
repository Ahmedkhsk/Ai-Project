from model_loader import ModelLoader
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import logging

def predict_pneumonia(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L')
        img = img.resize((150, 150))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        model = ModelLoader.get_instance().get_model()
        prediction = model.predict(img_array)

        has_pneumonia = bool(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0] if has_pneumonia else 1 - prediction[0][0])

        return {
            "has_pneumonia": has_pneumonia,
            "confidence": confidence
        }

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise


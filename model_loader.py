import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def load_model(self):
        if self._model is None:
            try:
                model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_model.keras')
                logger.info(f"Loading model from {model_path}")
                self._model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
        return self._model

    def get_model(self):
        return self.load_model()

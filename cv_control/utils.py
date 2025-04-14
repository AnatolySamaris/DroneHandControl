import os
from ament_index_python.packages import get_package_share_directory
from keras import models

def load_keras_model():
    package_dir = get_package_share_directory("cv_control")
    model_path = os.path.join(package_dir, "models", "model2.h5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = models.load_model(model_path)
    if model is None:
        raise Exception(f"Model of path '{model_path}' is None!!!")
    return model
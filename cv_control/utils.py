import os
from ament_index_python.packages import get_package_share_directory
from keras import models
import numpy as np
from collections import deque

class ValueFilter:
    """
    Для фильтрации управляющего воздействия.
    Применяется скользящее среднее и экспоненциальное сглаживание
    для снижения влияния шумов.
    """
    def __init__(self, window_size: int, alpha=0.2) -> None:
        self.window = deque([], maxlen=window_size)
        self.alpha = alpha
        self.smoothed_ema = None
    
    def update(self, new_value) -> float:
        self.window.append(new_value)

        # Медиана, для отсечения выбросов
        median = np.median(self.window)

        # Скользящее среднее
        cleaned = [x for x in self.window if abs(x - median) < 2 * np.std(self.window)]
        ma = np.mean(cleaned) if cleaned else median  # Если все значения сильно отличаются, берём медиану

        # Экспоненциальное сглаживание
        if self.smoothed_ema is None:
            self.smoothed_ema = ma
        else:
            self.smoothed_ema = self.alpha * ma + (1 - self.alpha) * self.smoothed_ema
        
        return self.smoothed_ema

# ==============================================
# ==============================================
# ==============================================

def load_keras_model():
    package_dir = get_package_share_directory("cv_control")
    model_path = os.path.join(package_dir, "models", "model2.h5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = models.load_model(model_path)
    if model is None:
        raise Exception(f"Model of path '{model_path}' is None!!!")
    return model
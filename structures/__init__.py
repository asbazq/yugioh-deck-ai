# Avoid importing TensorFlow-dependent modules at package import time.
# Import DetectionResult only; import EmbeddingMatrix explicitly where needed:
from .detection import *

__all__ = [
    # from detection.py
    'DetectionResult',
]

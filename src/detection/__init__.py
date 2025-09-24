"""
Módulo de detecção de plantas para o ArboreoMonitor.

Este módulo contém todas as funcionalidades relacionadas à detecção,
segmentação e classificação de plantas e árvores.
"""

from .plant_detector import PlantDetector
from .segmentation import PlantSegmentation
from .classifier import PlantClassifier

__all__ = ['PlantDetector', 'PlantSegmentation', 'PlantClassifier']

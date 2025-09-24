"""
Classificador de plantas para o ArboreoMonitor.

Este módulo implementa a classificação de plantas detectadas usando
modelos de deep learning e características visuais.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class PlantSpecies(Enum):
    """Espécies de plantas conhecidas."""
    # Árvores
    EUCALYPTUS = "eucalyptus"
    PINE = "pine"
    OAK = "oak"
    MAPLE = "maple"
    PALM = "palm"
    
    # Arbustos
    ROSE = "rose"
    HIBISCUS = "hibiscus"
    AZALEA = "azalea"
    
    # Gramíneas
    GRASS = "grass"
    BAMBOO = "bamboo"
    
    # Flores
    SUNFLOWER = "sunflower"
    TULIP = "tulip"
    
    # Desconhecido
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Resultado de uma classificação."""
    species: PlantSpecies
    confidence: float
    features: Dict[str, float]
    growth_stage: str  # 'seedling', 'young', 'mature', 'old'
    health_score: float  # 0-1


class PlantClassifier:
    """
    Classificador de plantas usando múltiplas abordagens.
    
    Suporta:
    - Classificação por deep learning
    - Classificação por características visuais
    - Detecção de estágio de crescimento
    - Avaliação de saúde da planta
    """
    
    def __init__(self, model_config: dict = None):
        """
        Inicializa o classificador de plantas.
        
        Args:
            model_config: Configuração dos modelos
        """
        self.model_config = model_config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = None
        self.classifier_model = None
        self.species_mapping = {}
        
        self._load_models()
    
    def _load_models(self):
        """Carrega os modelos de classificação."""
        try:
            # Carregar modelo de classificação (se disponível)
            model_path = self.model_config.get('classifier_model')
            if model_path and Path(model_path).exists():
                self.classifier_model = torch.load(model_path, map_location=self.device)
                self.classifier_model.eval()
                logger.info("Modelo de classificação carregado")
            
            # Carregar mapeamento de espécies
            mapping_path = self.model_config.get('species_mapping')
            if mapping_path and Path(mapping_path).exists():
                with open(mapping_path, 'rb') as f:
                    self.species_mapping = pickle.load(f)
                logger.info("Mapeamento de espécies carregado")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelos de classificação: {e}")
    
    def classify_plant(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClassificationResult:
        """
        Classifica uma planta em uma imagem.
        
        Args:
            image: Imagem da planta
            mask: Máscara da planta (opcional)
            
        Returns:
            Resultado da classificação
        """
        # Extrair características visuais
        features = self._extract_visual_features(image, mask)
        
        # Classificar espécie
        species, confidence = self._classify_species(image, features)
        
        # Determinar estágio de crescimento
        growth_stage = self._determine_growth_stage(features)
        
        # Avaliar saúde
        health_score = self._assess_plant_health(image, features)
        
        return ClassificationResult(
            species=species,
            confidence=confidence,
            features=features,
            growth_stage=growth_stage,
            health_score=health_score
        )
    
    def _extract_visual_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extrai características visuais da planta."""
        features = {}
        
        # Aplicar máscara se fornecida
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = image
        
        # Converter para HSV para análise de cor
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # Características de cor
        features['mean_hue'] = np.mean(hsv[:, :, 0])
        features['mean_saturation'] = np.mean(hsv[:, :, 1])
        features['mean_value'] = np.mean(hsv[:, :, 2])
        features['std_hue'] = np.std(hsv[:, :, 0])
        features['std_saturation'] = np.std(hsv[:, :, 1])
        features['std_value'] = np.std(hsv[:, :, 2])
        
        # Características de textura
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Haralick features (simplificado)
        features['texture_energy'] = np.sum(gray ** 2)
        features['texture_contrast'] = np.std(gray)
        
        # Características de forma (se máscara disponível)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                
                # Área e perímetro
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                features['area'] = area
                features['perimeter'] = perimeter
                features['compactness'] = (perimeter ** 2) / area if area > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                features['aspect_ratio'] = w / h if h > 0 else 0
                
                # Circularidade
                features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        return features
    
    def _classify_species(self, image: np.ndarray, features: Dict[str, float]) -> Tuple[PlantSpecies, float]:
        """Classifica a espécie da planta."""
        # Se modelo de deep learning disponível, usar ele
        if self.classifier_model is not None:
            return self._classify_with_deep_learning(image)
        
        # Caso contrário, usar classificação baseada em características
        return self._classify_with_features(features)
    
    def _classify_with_deep_learning(self, image: np.ndarray) -> Tuple[PlantSpecies, float]:
        """Classificação usando deep learning."""
        try:
            # Pré-processar imagem
            processed_image = self._preprocess_image_for_model(image)
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.classifier_model(processed_image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Mapear predição para espécie
                species = self.species_mapping.get(predicted.item(), PlantSpecies.UNKNOWN)
                
                return species, confidence.item()
                
        except Exception as e:
            logger.error(f"Erro na classificação por deep learning: {e}")
            return PlantSpecies.UNKNOWN, 0.0
    
    def _classify_with_features(self, features: Dict[str, float]) -> Tuple[PlantSpecies, float]:
        """Classificação baseada em características visuais."""
        # Regras simples baseadas em características
        # Estas podem ser expandidas com modelos mais sofisticados
        
        mean_hue = features.get('mean_hue', 0)
        mean_saturation = features.get('mean_saturation', 0)
        aspect_ratio = features.get('aspect_ratio', 1)
        circularity = features.get('circularity', 0)
        
        # Classificação baseada em regras
        if mean_hue < 30:  # Verde
            if aspect_ratio > 2:  # Alto e estreito
                return PlantSpecies.PINE, 0.7
            elif circularity > 0.7:  # Circular
                return PlantSpecies.PALM, 0.6
            else:
                return PlantSpecies.EUCALYPTUS, 0.5
        elif 30 <= mean_hue <= 60:  # Amarelo-verde
            return PlantSpecies.GRASS, 0.6
        elif mean_hue > 60:  # Amarelo/vermelho
            return PlantSpecies.SUNFLOWER, 0.5
        else:
            return PlantSpecies.UNKNOWN, 0.3
    
    def _determine_growth_stage(self, features: Dict[str, float]) -> str:
        """Determina o estágio de crescimento da planta."""
        area = features.get('area', 0)
        aspect_ratio = features.get('aspect_ratio', 1)
        circularity = features.get('circularity', 0)
        
        # Regras baseadas em características
        if area < 1000:  # Muito pequena
            return 'seedling'
        elif area < 5000 and aspect_ratio > 1.5:  # Pequena e alta
            return 'young'
        elif area > 20000 and circularity > 0.6:  # Grande e circular
            return 'mature'
        else:
            return 'mature'
    
    def _assess_plant_health(self, image: np.ndarray, features: Dict[str, float]) -> float:
        """Avalia a saúde da planta (0-1)."""
        # Análise de cor para detectar doenças
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detectar folhas amareladas (possível deficiência)
        yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))
        yellow_ratio = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Detectar folhas marrons (possível doença)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 255, 100]))
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Detectar folhas verdes saudáveis
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Calcular score de saúde
        health_score = green_ratio - (yellow_ratio * 0.5) - (brown_ratio * 0.8)
        health_score = max(0, min(1, health_score))  # Clamp entre 0 e 1
        
        return health_score
    
    def _preprocess_image_for_model(self, image: np.ndarray) -> torch.Tensor:
        """Pré-processa imagem para o modelo de deep learning."""
        # Redimensionar para tamanho esperado pelo modelo
        resized = cv2.resize(image, (224, 224))
        
        # Normalizar
        normalized = resized.astype(np.float32) / 255.0
        
        # Converter para tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify_multiple_plants(self, images: List[np.ndarray], masks: List[np.ndarray] = None) -> List[ClassificationResult]:
        """
        Classifica múltiplas plantas.
        
        Args:
            images: Lista de imagens de plantas
            masks: Lista de máscaras (opcional)
            
        Returns:
            Lista de resultados de classificação
        """
        results = []
        
        for i, image in enumerate(images):
            mask = masks[i] if masks and i < len(masks) else None
            result = self.classify_plant(image, mask)
            results.append(result)
            
        return results
    
    def get_species_statistics(self, classifications: List[ClassificationResult]) -> Dict[str, Dict[str, float]]:
        """
        Calcula estatísticas das classificações.
        
        Args:
            classifications: Lista de classificações
            
        Returns:
            Estatísticas por espécie
        """
        stats = {}
        
        for result in classifications:
            species = result.species.value
            
            if species not in stats:
                stats[species] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'avg_health': 0,
                    'growth_stages': {}
                }
            
            stats[species]['count'] += 1
            stats[species]['avg_confidence'] += result.confidence
            stats[species]['avg_health'] += result.health_score
            
            growth_stage = result.growth_stage
            if growth_stage not in stats[species]['growth_stages']:
                stats[species]['growth_stages'][growth_stage] = 0
            stats[species]['growth_stages'][growth_stage] += 1
        
        # Calcular médias
        for species in stats:
            count = stats[species]['count']
            stats[species]['avg_confidence'] /= count
            stats[species]['avg_health'] /= count
        
        return stats

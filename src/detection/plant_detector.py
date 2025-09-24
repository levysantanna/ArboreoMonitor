"""
Detector de plantas para o ArboreoMonitor.

Este módulo implementa a detecção de plantas e árvores usando modelos
de deep learning, incluindo YOLO e Detectron2.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PlantType(Enum):
    """Tipos de plantas detectadas."""
    TREE = "tree"
    SHRUB = "shrub"
    GRASS = "grass"
    FLOWER = "flower"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """Resultado de uma detecção de planta."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    plant_type: PlantType
    mask: Optional[np.ndarray] = None
    area: float = 0.0
    center: Tuple[int, int] = (0, 0)


class PlantDetector:
    """
    Detector de plantas usando múltiplos modelos de deep learning.
    
    Suporta:
    - YOLO para detecção rápida
    - Detectron2 para segmentação precisa
    - Modelos customizados para plantas específicas
    """
    
    def __init__(self, model_config: dict = None):
        """
        Inicializa o detector de plantas.
        
        Args:
            model_config: Configuração dos modelos
        """
        self.model_config = model_config or {}
        self.yolo_model = None
        self.detectron_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_models()
    
    def _load_models(self):
        """Carrega os modelos de detecção."""
        try:
            # Carregar modelo YOLO
            from ultralytics import YOLO
            model_path = self.model_config.get('yolo_model', 'yolov8n.pt')
            self.yolo_model = YOLO(model_path)
            logger.info("Modelo YOLO carregado")
            
            # Carregar modelo Detectron2 (se disponível)
            try:
                import detectron2
                from detectron2 import model_zoo
                from detectron2.engine import DefaultPredictor
                from detectron2.config import get_cfg
                
                cfg = get_cfg()
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                cfg.MODEL.DEVICE = str(self.device)
                
                self.detectron_model = DefaultPredictor(cfg)
                logger.info("Modelo Detectron2 carregado")
            except ImportError:
                logger.warning("Detectron2 não disponível, usando apenas YOLO")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
    
    def detect_plants(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Detection]:
        """
        Detecta plantas em uma imagem.
        
        Args:
            image: Imagem de entrada
            min_confidence: Confiança mínima para detecções
            
        Returns:
            Lista de detecções de plantas
        """
        detections = []
        
        # Detecção com YOLO
        yolo_detections = self._detect_with_yolo(image, min_confidence)
        detections.extend(yolo_detections)
        
        # Detecção com Detectron2 (se disponível)
        if self.detectron_model is not None:
            detectron_detections = self._detect_with_detectron2(image, min_confidence)
            detections.extend(detectron_detections)
        
        # Filtrar detecções duplicadas
        detections = self._filter_duplicates(detections)
        
        # Filtrar plantas muito pequenas (< 60cm estimado)
        detections = self._filter_small_plants(detections, image.shape)
        
        return detections
    
    def _detect_with_yolo(self, image: np.ndarray, min_confidence: float) -> List[Detection]:
        """Detecção usando YOLO."""
        if self.yolo_model is None:
            return []
            
        try:
            results = self.yolo_model(image, conf=min_confidence)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Converter coordenadas
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Mapear classe para tipo de planta
                        plant_type = self._map_class_to_plant_type(class_id)
                        
                        detection = Detection(
                            bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            confidence=float(confidence),
                            plant_type=plant_type,
                            area=(x2-x1) * (y2-y1),
                            center=(int((x1+x2)/2), int((y1+y2)/2))
                        )
                        detections.append(detection)
                        
            return detections
            
        except Exception as e:
            logger.error(f"Erro na detecção YOLO: {e}")
            return []
    
    def _detect_with_detectron2(self, image: np.ndarray, min_confidence: float) -> List[Detection]:
        """Detecção usando Detectron2."""
        if self.detectron_model is None:
            return []
            
        try:
            outputs = self.detectron_model(image)
            detections = []
            
            instances = outputs["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                if score >= min_confidence:
                    x1, y1, x2, y2 = box
                    
                    # Obter máscara se disponível
                    mask = None
                    if hasattr(instances, 'pred_masks'):
                        mask = instances.pred_masks[i].cpu().numpy()
                    
                    plant_type = self._map_class_to_plant_type(class_id)
                    
                    detection = Detection(
                        bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        confidence=float(score),
                        plant_type=plant_type,
                        mask=mask,
                        area=(x2-x1) * (y2-y1),
                        center=(int((x1+x2)/2), int((y1+y2)/2))
                    )
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.error(f"Erro na detecção Detectron2: {e}")
            return []
    
    def _map_class_to_plant_type(self, class_id: int) -> PlantType:
        """Mapeia ID da classe para tipo de planta."""
        # Mapeamento básico - pode ser customizado
        plant_mapping = {
            0: PlantType.TREE,    # person -> tree (exemplo)
            1: PlantType.SHRUB,   # bicycle -> shrub
            2: PlantType.GRASS,   # car -> grass
            # Adicionar mais mapeamentos conforme necessário
        }
        
        return plant_mapping.get(class_id, PlantType.UNKNOWN)
    
    def _filter_duplicates(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Remove detecções duplicadas usando NMS."""
        if len(detections) <= 1:
            return detections
            
        # Converter para formato OpenCV
        boxes = []
        scores = []
        
        for det in detections:
            x, y, w, h = det.bbox
            boxes.append([x, y, x+w, y+h])
            scores.append(det.confidence)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Aplicar NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return detections
    
    def _filter_small_plants(self, detections: List[Detection], image_shape: Tuple[int, int, int]) -> List[Detection]:
        """Filtra plantas muito pequenas (menos de 60cm estimado)."""
        height, width = image_shape[:2]
        min_area = (width * height) * 0.001  # 0.1% da imagem
        
        filtered = []
        for det in detections:
            if det.area >= min_area:
                filtered.append(det)
                
        return filtered
    
    def crop_plant_regions(self, image: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """
        Recorta regiões das plantas detectadas.
        
        Args:
            image: Imagem original
            detections: Lista de detecções
            
        Returns:
            Lista de imagens recortadas
        """
        cropped_images = []
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # Garantir que as coordenadas estão dentro da imagem
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                crop = image[y:y+h, x:x+w]
                cropped_images.append(crop)
                
        return cropped_images
    
    def visualize_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Visualiza detecções na imagem.
        
        Args:
            image: Imagem original
            detections: Lista de detecções
            
        Returns:
            Imagem com detecções visualizadas
        """
        vis_image = image.copy()
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # Cor baseada no tipo de planta
            color = self._get_plant_color(det.plant_type)
            
            # Desenhar bounding box
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Desenhar label
            label = f"{det.plant_type.value}: {det.confidence:.2f}"
            cv2.putText(vis_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Desenhar máscara se disponível
            if det.mask is not None:
                mask_colored = np.zeros_like(vis_image)
                mask_colored[det.mask] = color
                vis_image = cv2.addWeighted(vis_image, 0.8, mask_colored, 0.2, 0)
                
        return vis_image
    
    def _get_plant_color(self, plant_type: PlantType) -> Tuple[int, int, int]:
        """Retorna cor para visualização baseada no tipo de planta."""
        colors = {
            PlantType.TREE: (0, 255, 0),      # Verde
            PlantType.SHRUB: (0, 255, 255),    # Amarelo
            PlantType.GRASS: (255, 0, 0),      # Azul
            PlantType.FLOWER: (255, 0, 255),   # Magenta
            PlantType.UNKNOWN: (128, 128, 128) # Cinza
        }
        return colors.get(plant_type, (128, 128, 128))

"""
Segmentação de plantas para o ArboreoMonitor.

Este módulo implementa a segmentação precisa de plantas usando
técnicas de deep learning e processamento de imagem.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
from skimage import segmentation, measure
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


@dataclass
class PlantSegment:
    """Segmento de uma planta."""
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    plant_type: str
    confidence: float


class PlantSegmentation:
    """
    Segmentador de plantas usando múltiplas técnicas.
    
    Suporta:
    - Segmentação semântica com deep learning
    - Segmentação baseada em cor (HSV)
    - Segmentação por watershed
    - Segmentação por threshold adaptativo
    """
    
    def __init__(self, model_config: dict = None):
        """
        Inicializa o segmentador de plantas.
        
        Args:
            model_config: Configuração dos modelos
        """
        self.model_config = model_config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def segment_plants(self, image: np.ndarray, method: str = 'hybrid') -> List[PlantSegment]:
        """
        Segmenta plantas em uma imagem.
        
        Args:
            image: Imagem de entrada
            method: Método de segmentação ('color', 'watershed', 'deep_learning', 'hybrid')
            
        Returns:
            Lista de segmentos de plantas
        """
        if method == 'color':
            return self._segment_by_color(image)
        elif method == 'watershed':
            return self._segment_by_watershed(image)
        elif method == 'deep_learning':
            return self._segment_by_deep_learning(image)
        elif method == 'hybrid':
            return self._segment_hybrid(image)
        else:
            raise ValueError(f"Método de segmentação não suportado: {method}")
    
    def _segment_by_color(self, image: np.ndarray) -> List[PlantSegment]:
        """Segmentação baseada em cor (verde)."""
        # Converter para HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir range de verde
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Criar máscara
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Operações morfológicas para limpar a máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remover objetos pequenos
        mask = remove_small_objects(mask.astype(bool), min_size=1000)
        mask = mask.astype(np.uint8) * 255
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segments = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filtrar áreas pequenas
                # Criar máscara do contorno
                contour_mask = np.zeros_like(mask)
                cv2.fillPoly(contour_mask, [contour], 255)
                
                # Calcular propriedades
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                moments = cv2.moments(contour)
                
                if moments['m00'] != 0:
                    centroid = (int(moments['m10']/moments['m00']), 
                              int(moments['m01']/moments['m00']))
                else:
                    centroid = (0, 0)
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                segment = PlantSegment(
                    mask=contour_mask,
                    bbox=(x, y, w, h),
                    area=area,
                    perimeter=perimeter,
                    centroid=centroid,
                    plant_type='plant',
                    confidence=0.8  # Confiança baseada em cor
                )
                segments.append(segment)
        
        return segments
    
    def _segment_by_watershed(self, image: np.ndarray) -> List[PlantSegment]:
        """Segmentação usando algoritmo watershed."""
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Operações morfológicas
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Área de fundo
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Área de primeiro plano
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        sure_fg = np.uint8(dist_transform > 0.3 * dist_transform.max())
        
        # Área desconhecida
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marcadores
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Aplicar watershed
        markers = cv2.watershed(image, markers)
        
        # Extrair segmentos
        segments = []
        for label in np.unique(markers):
            if label == 0 or label == 1:  # Pular fundo e marcador 0
                continue
                
            # Criar máscara para o label
            mask = (markers == label).astype(np.uint8) * 255
            
            # Encontrar contorno
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                area = cv2.contourArea(contour)
                
                if area > 1000:  # Filtrar áreas pequenas
                    # Calcular propriedades
                    perimeter = cv2.arcLength(contour, True)
                    moments = cv2.moments(contour)
                    
                    if moments['m00'] != 0:
                        centroid = (int(moments['m10']/moments['m00']), 
                                  int(moments['m01']/moments['m00']))
                    else:
                        centroid = (0, 0)
                    
                    # Bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    segment = PlantSegment(
                        mask=mask,
                        bbox=(x, y, w, h),
                        area=area,
                        perimeter=perimeter,
                        centroid=centroid,
                        plant_type='plant',
                        confidence=0.7  # Confiança baseada em watershed
                    )
                    segments.append(segment)
        
        return segments
    
    def _segment_by_deep_learning(self, image: np.ndarray) -> List[PlantSegment]:
        """Segmentação usando deep learning."""
        # Implementação básica - pode ser expandida com modelos específicos
        # Por enquanto, usar segmentação por cor como fallback
        return self._segment_by_color(image)
    
    def _segment_hybrid(self, image: np.ndarray) -> List[PlantSegment]:
        """Segmentação híbrida combinando múltiplos métodos."""
        # Segmentação por cor
        color_segments = self._segment_by_color(image)
        
        # Segmentação por watershed
        watershed_segments = self._segment_by_watershed(image)
        
        # Combinar resultados
        all_segments = color_segments + watershed_segments
        
        # Remover duplicatas baseado em sobreposição
        filtered_segments = self._filter_overlapping_segments(all_segments)
        
        return filtered_segments
    
    def _filter_overlapping_segments(self, segments: List[PlantSegment], iou_threshold: float = 0.5) -> List[PlantSegment]:
        """Remove segmentos sobrepostos."""
        if len(segments) <= 1:
            return segments
        
        filtered = []
        used = set()
        
        for i, seg1 in enumerate(segments):
            if i in used:
                continue
                
            best_segment = seg1
            best_confidence = seg1.confidence
            
            for j, seg2 in enumerate(segments[i+1:], i+1):
                if j in used:
                    continue
                    
                # Calcular IoU
                iou = self._calculate_iou(seg1.bbox, seg2.bbox)
                
                if iou > iou_threshold:
                    # Manter o segmento com maior confiança
                    if seg2.confidence > best_confidence:
                        best_segment = seg2
                        best_confidence = seg2.confidence
                    used.add(j)
            
            filtered.append(best_segment)
            used.add(i)
        
        return filtered
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calcula Intersection over Union entre dois bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calcular interseção
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcular união
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def crop_plant_segments(self, image: np.ndarray, segments: List[PlantSegment]) -> List[np.ndarray]:
        """
        Recorta segmentos de plantas.
        
        Args:
            image: Imagem original
            segments: Lista de segmentos
            
        Returns:
            Lista de imagens recortadas
        """
        cropped_images = []
        
        for segment in segments:
            x, y, w, h = segment.bbox
            
            # Garantir que as coordenadas estão dentro da imagem
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                crop = image[y:y+h, x:x+w]
                cropped_images.append(crop)
                
        return cropped_images
    
    def visualize_segments(self, image: np.ndarray, segments: List[PlantSegment]) -> np.ndarray:
        """
        Visualiza segmentos na imagem.
        
        Args:
            image: Imagem original
            segments: Lista de segmentos
            
        Returns:
            Imagem com segmentos visualizados
        """
        vis_image = image.copy()
        
        for i, segment in enumerate(segments):
            # Cor baseada no índice
            color = self._get_segment_color(i)
            
            # Desenhar máscara
            mask_colored = np.zeros_like(vis_image)
            mask_colored[segment.mask > 0] = color
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
            
            # Desenhar bounding box
            x, y, w, h = segment.bbox
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Desenhar centroide
            cx, cy = segment.centroid
            cv2.circle(vis_image, (int(cx), int(cy)), 5, color, -1)
            
            # Desenhar label
            label = f"Plant {i+1}: {segment.area:.0f}px"
            cv2.putText(vis_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def _get_segment_color(self, index: int) -> Tuple[int, int, int]:
        """Retorna cor para visualização baseada no índice."""
        colors = [
            (0, 255, 0),    # Verde
            (255, 0, 0),    # Azul
            (0, 0, 255),    # Vermelho
            (255, 255, 0),  # Ciano
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Amarelo
        ]
        return colors[index % len(colors)]

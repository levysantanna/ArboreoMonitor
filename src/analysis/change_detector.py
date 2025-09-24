"""
Detector de mudanças para o ArboreoMonitor.

Este módulo implementa a detecção de mudanças nas plantas,
incluindo cortes, podas e outras alterações significativas.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Tipos de mudanças detectadas."""
    CUT = "cut"  # Corte/remoção
    PRUNE = "prune"  # Poda
    GROWTH = "growth"  # Crescimento
    DISEASE = "disease"  # Doença
    DAMAGE = "damage"  # Danos
    UNKNOWN = "unknown"  # Mudança não identificada


@dataclass
class ChangeDetection:
    """Detecção de mudança em uma planta."""
    plant_id: str
    timestamp: datetime
    change_type: ChangeType
    confidence: float
    severity: str  # 'low', 'medium', 'high'
    description: str
    before_image: Optional[np.ndarray] = None
    after_image: Optional[np.ndarray] = None
    change_mask: Optional[np.ndarray] = None
    affected_area: float = 0.0
    metadata: Dict[str, any] = None


class ChangeDetector:
    """
    Detector de mudanças em plantas.
    
    Suporta:
    - Detecção de cortes e podas
    - Detecção de crescimento
    - Detecção de doenças
    - Análise de diferenças visuais
    """
    
    def __init__(self, config: dict = None):
        """
        Inicializa o detector de mudanças.
        
        Args:
            config: Configuração do detector
        """
        self.config = config or {}
        self.change_threshold = self.config.get('change_threshold', 0.3)
        self.motion_threshold = self.config.get('motion_threshold', 0.1)
        
    def detect_changes(self, before_image: np.ndarray, after_image: np.ndarray,
                      plant_mask: Optional[np.ndarray] = None) -> List[ChangeDetection]:
        """
        Detecta mudanças entre duas imagens.
        
        Args:
            before_image: Imagem anterior
            after_image: Imagem posterior
            plant_mask: Máscara da planta (opcional)
            
        Returns:
            Lista de detecções de mudança
        """
        try:
            # Pré-processar imagens
            before_processed = self._preprocess_image(before_image)
            after_processed = self._preprocess_image(after_image)
            
            # Calcular diferença
            difference = self._calculate_difference(before_processed, after_processed)
            
            # Aplicar máscara se fornecida
            if plant_mask is not None:
                difference = cv2.bitwise_and(difference, plant_mask)
            
            # Detectar mudanças
            changes = []
            
            # Detectar cortes/podas
            cut_changes = self._detect_cuts(before_processed, after_processed, difference)
            changes.extend(cut_changes)
            
            # Detectar crescimento
            growth_changes = self._detect_growth(before_processed, after_processed, difference)
            changes.extend(growth_changes)
            
            # Detectar doenças
            disease_changes = self._detect_diseases(before_processed, after_processed, difference)
            changes.extend(disease_changes)
            
            # Detectar danos
            damage_changes = self._detect_damage(before_processed, after_processed, difference)
            changes.extend(damage_changes)
            
            return changes
            
        except Exception as e:
            logger.error(f"Erro na detecção de mudanças: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Pré-processa imagem para análise."""
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar filtro gaussiano para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Aplicar equalização de histograma
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def _calculate_difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Calcula diferença entre duas imagens."""
        # Calcular diferença absoluta
        diff = cv2.absdiff(before, after)
        
        # Aplicar threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Operações morfológicas para limpar
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _detect_cuts(self, before: np.ndarray, after: np.ndarray, 
                    difference: np.ndarray) -> List[ChangeDetection]:
        """Detecta cortes e podas."""
        changes = []
        
        try:
            # Encontrar contornos na diferença
            contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 100:  # Área mínima para considerar
                    # Calcular características do contorno
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Analisar se parece com um corte
                    if self._is_cut_like(contour, aspect_ratio):
                        # Calcular confiança baseada na área e forma
                        confidence = min(1.0, area / 1000.0)
                        
                        # Determinar severidade
                        severity = 'high' if area > 5000 else 'medium' if area > 1000 else 'low'
                        
                        change = ChangeDetection(
                            plant_id="unknown",  # Será preenchido pelo sistema
                            timestamp=datetime.now(),
                            change_type=ChangeType.CUT,
                            confidence=confidence,
                            severity=severity,
                            description=f"Corte detectado - área: {area:.0f}px",
                            change_mask=self._create_mask_from_contour(contour, difference.shape),
                            affected_area=area,
                            metadata={
                                'contour_area': area,
                                'aspect_ratio': aspect_ratio,
                                'bounding_box': (x, y, w, h)
                            }
                        )
                        changes.append(change)
            
        except Exception as e:
            logger.error(f"Erro na detecção de cortes: {e}")
        
        return changes
    
    def _detect_growth(self, before: np.ndarray, after: np.ndarray, 
                      difference: np.ndarray) -> List[ChangeDetection]:
        """Detecta crescimento."""
        changes = []
        
        try:
            # Analisar diferenças que indicam crescimento
            # (áreas que apareceram na imagem posterior)
            growth_mask = self._detect_growth_areas(before, after)
            
            if np.sum(growth_mask) > 0:
                # Encontrar contornos de crescimento
                contours, _ = cv2.findContours(growth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > 200:  # Área mínima para crescimento
                        confidence = min(1.0, area / 2000.0)
                        
                        change = ChangeDetection(
                            plant_id="unknown",
                            timestamp=datetime.now(),
                            change_type=ChangeType.GROWTH,
                            confidence=confidence,
                            severity='low',  # Crescimento é geralmente positivo
                            description=f"Crescimento detectado - área: {area:.0f}px",
                            change_mask=self._create_mask_from_contour(contour, growth_mask.shape),
                            affected_area=area,
                            metadata={'growth_area': area}
                        )
                        changes.append(change)
            
        except Exception as e:
            logger.error(f"Erro na detecção de crescimento: {e}")
        
        return changes
    
    def _detect_diseases(self, before: np.ndarray, after: np.ndarray, 
                        difference: np.ndarray) -> List[ChangeDetection]:
        """Detecta doenças e problemas de saúde."""
        changes = []
        
        try:
            # Analisar mudanças de cor que podem indicar doença
            disease_mask = self._detect_disease_areas(before, after)
            
            if np.sum(disease_mask) > 0:
                contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > 100:
                        confidence = min(1.0, area / 1000.0)
                        severity = 'high' if area > 2000 else 'medium' if area > 500 else 'low'
                        
                        change = ChangeDetection(
                            plant_id="unknown",
                            timestamp=datetime.now(),
                            change_type=ChangeType.DISEASE,
                            confidence=confidence,
                            severity=severity,
                            description=f"Possível doença detectada - área: {area:.0f}px",
                            change_mask=self._create_mask_from_contour(contour, disease_mask.shape),
                            affected_area=area,
                            metadata={'disease_area': area}
                        )
                        changes.append(change)
            
        except Exception as e:
            logger.error(f"Erro na detecção de doenças: {e}")
        
        return changes
    
    def _detect_damage(self, before: np.ndarray, after: np.ndarray, 
                      difference: np.ndarray) -> List[ChangeDetection]:
        """Detecta danos físicos."""
        changes = []
        
        try:
            # Analisar padrões que indicam danos
            damage_mask = self._detect_damage_areas(before, after)
            
            if np.sum(damage_mask) > 0:
                contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > 150:
                        confidence = min(1.0, area / 1500.0)
                        severity = 'high' if area > 3000 else 'medium' if area > 800 else 'low'
                        
                        change = ChangeDetection(
                            plant_id="unknown",
                            timestamp=datetime.now(),
                            change_type=ChangeType.DAMAGE,
                            confidence=confidence,
                            severity=severity,
                            description=f"Danos detectados - área: {area:.0f}px",
                            change_mask=self._create_mask_from_contour(contour, damage_mask.shape),
                            affected_area=area,
                            metadata={'damage_area': area}
                        )
                        changes.append(change)
            
        except Exception as e:
            logger.error(f"Erro na detecção de danos: {e}")
        
        return changes
    
    def _is_cut_like(self, contour: np.ndarray, aspect_ratio: float) -> bool:
        """Verifica se um contorno parece com um corte."""
        # Cortes geralmente têm formas alongadas ou retangulares
        return aspect_ratio > 2.0 or aspect_ratio < 0.5
    
    def _detect_growth_areas(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Detecta áreas de crescimento."""
        # Usar subtração para encontrar áreas que apareceram
        growth = cv2.subtract(after, before)
        _, growth_mask = cv2.threshold(growth, 20, 255, cv2.THRESH_BINARY)
        return growth_mask
    
    def _detect_disease_areas(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Detecta áreas de doença."""
        # Converter para HSV para análise de cor
        before_hsv = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
        after_hsv = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
        
        before_hsv = cv2.cvtColor(before_hsv, cv2.COLOR_BGR2HSV)
        after_hsv = cv2.cvtColor(after_hsv, cv2.COLOR_BGR2HSV)
        
        # Detectar mudanças para cores que indicam doença (amarelo, marrom)
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([20, 255, 100])
        
        yellow_mask = cv2.inRange(after_hsv, yellow_lower, yellow_upper)
        brown_mask = cv2.inRange(after_hsv, brown_lower, brown_upper)
        
        disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)
        return disease_mask
    
    def _detect_damage_areas(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Detecta áreas de danos."""
        # Usar detecção de bordas para encontrar quebras ou rachaduras
        before_edges = cv2.Canny(before, 50, 150)
        after_edges = cv2.Canny(after, 50, 150)
        
        # Encontrar diferenças nas bordas
        edge_diff = cv2.absdiff(before_edges, after_edges)
        
        # Aplicar operações morfológicas
        kernel = np.ones((3, 3), np.uint8)
        damage_mask = cv2.morphologyEx(edge_diff, cv2.MORPH_CLOSE, kernel)
        
        return damage_mask
    
    def _create_mask_from_contour(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Cria máscara a partir de um contorno."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        return mask
    
    def analyze_change_severity(self, changes: List[ChangeDetection]) -> Dict[str, any]:
        """
        Analisa a severidade das mudanças detectadas.
        
        Args:
            changes: Lista de detecções de mudança
            
        Returns:
            Análise de severidade
        """
        if not changes:
            return {'total_changes': 0, 'severity_level': 'none'}
        
        # Contar mudanças por tipo e severidade
        change_counts = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for change in changes:
            change_type = change.change_type.value
            if change_type not in change_counts:
                change_counts[change_type] = 0
            change_counts[change_type] += 1
            
            severity_counts[change.severity] += 1
        
        # Determinar nível geral de severidade
        if severity_counts['high'] > 0:
            overall_severity = 'high'
        elif severity_counts['medium'] > 0:
            overall_severity = 'medium'
        else:
            overall_severity = 'low'
        
        # Calcular área total afetada
        total_affected_area = sum(change.affected_area for change in changes)
        
        return {
            'total_changes': len(changes),
            'change_types': change_counts,
            'severity_distribution': severity_counts,
            'overall_severity': overall_severity,
            'total_affected_area': total_affected_area,
            'requires_attention': overall_severity in ['high', 'medium']
        }
    
    def generate_change_report(self, changes: List[ChangeDetection]) -> Dict[str, any]:
        """
        Gera relatório de mudanças detectadas.
        
        Args:
            changes: Lista de detecções de mudança
            
        Returns:
            Relatório de mudanças
        """
        if not changes:
            return {'message': 'Nenhuma mudança detectada'}
        
        # Agrupar por tipo de mudança
        changes_by_type = {}
        for change in changes:
            change_type = change.change_type.value
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(change)
        
        # Gerar relatório
        report = {
            'summary': {
                'total_changes': len(changes),
                'change_types': len(changes_by_type),
                'timestamp': datetime.now().isoformat()
            },
            'changes_by_type': {},
            'recommendations': []
        }
        
        for change_type, type_changes in changes_by_type.items():
            report['changes_by_type'][change_type] = {
                'count': len(type_changes),
                'avg_confidence': np.mean([c.confidence for c in type_changes]),
                'severity_distribution': {
                    'low': len([c for c in type_changes if c.severity == 'low']),
                    'medium': len([c for c in type_changes if c.severity == 'medium']),
                    'high': len([c for c in type_changes if c.severity == 'high'])
                },
                'total_affected_area': sum(c.affected_area for c in type_changes)
            }
        
        # Gerar recomendações
        if ChangeType.CUT in changes_by_type:
            report['recommendations'].append("Cortes detectados - verificar se foram autorizados")
        
        if ChangeType.DISEASE in changes_by_type:
            report['recommendations'].append("Possíveis doenças detectadas - investigar tratamento")
        
        if ChangeType.DAMAGE in changes_by_type:
            report['recommendations'].append("Danos detectados - avaliar necessidade de reparo")
        
        return report

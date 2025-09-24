"""
Extrator de frames inteligente para o ArboreoMonitor.

Este módulo implementa a lógica de amostragem de frames baseada no tipo de fonte
e nas configurações do sistema.
"""

import cv2
import numpy as np
from typing import Generator, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Estratégias de amostragem de frames."""
    TIME_BASED = "time_based"  # Baseado em tempo
    FRAME_BASED = "frame_based"  # Baseado em número de frames
    ADAPTIVE = "adaptive"  # Adaptativo baseado no conteúdo


@dataclass
class SamplingConfig:
    """Configuração de amostragem de frames."""
    # Para vídeos normais (drones, arquivos)
    normal_interval_seconds: float = 300  # 5 minutos
    normal_min_interval: float = 60  # Mínimo 1 minuto
    
    # Para IP cams (streams longos)
    ipcam_interval_seconds: float = 3600  # 1 hora
    ipcam_min_interval: float = 300  # Mínimo 5 minutos
    
    # Para arquivos grandes (> 1GB)
    large_file_interval_seconds: float = 1800  # 30 minutos
    
    # Configurações adaptativas
    enable_adaptive: bool = True
    motion_threshold: float = 0.1  # Threshold para detecção de movimento
    quality_threshold: float = 0.7  # Threshold de qualidade mínima


class FrameExtractor:
    """
    Extrator inteligente de frames com diferentes estratégias de amostragem.
    """
    
    def __init__(self, config: SamplingConfig = None):
        """
        Inicializa o extrator de frames.
        
        Args:
            config: Configuração de amostragem
        """
        self.config = config or SamplingConfig()
        self.last_frame = None
        self.frame_count = 0
        
    def extract_frames(self, video_processor, source_type, file_size_mb: float = 0) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Extrai frames usando estratégia apropriada para o tipo de fonte.
        
        Args:
            video_processor: Processador de vídeo
            source_type: Tipo da fonte de vídeo
            file_size_mb: Tamanho do arquivo em MB (0 para streams)
            
        Yields:
            Tuplas (frame, metadata) dos frames extraídos
        """
        if source_type.value == "ip_cam":
            yield from self._extract_ipcam_frames(video_processor)
        elif source_type.value == "file" and file_size_mb > 1024:  # > 1GB
            yield from self._extract_large_file_frames(video_processor)
        else:
            yield from self._extract_normal_frames(video_processor)
    
    def _extract_normal_frames(self, video_processor) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Extrai frames de vídeos normais (drones, arquivos pequenos)."""
        interval = self.config.normal_interval_seconds
        
        for frame in video_processor.get_frames_at_intervals(interval):
            if self._should_extract_frame(frame):
                metadata = {
                    'timestamp': datetime.now(),
                    'frame_number': self.frame_count,
                    'extraction_method': 'normal',
                    'interval_seconds': interval
                }
                yield frame, metadata
                self.frame_count += 1
    
    def _extract_ipcam_frames(self, video_processor) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Extrai frames de IP cams com intervalos maiores."""
        interval = self.config.ipcam_interval_seconds
        
        for frame in video_processor.get_frames_at_intervals(interval):
            if self._should_extract_frame(frame):
                metadata = {
                    'timestamp': datetime.now(),
                    'frame_number': self.frame_count,
                    'extraction_method': 'ipcam',
                    'interval_seconds': interval
                }
                yield frame, metadata
                self.frame_count += 1
    
    def _extract_large_file_frames(self, video_processor) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Extrai frames de arquivos grandes com intervalos maiores."""
        interval = self.config.large_file_interval_seconds
        
        for frame in video_processor.get_frames_at_intervals(interval):
            if self._should_extract_frame(frame):
                metadata = {
                    'timestamp': datetime.now(),
                    'frame_number': self.frame_count,
                    'extraction_method': 'large_file',
                    'interval_seconds': interval
                }
                yield frame, metadata
                self.frame_count += 1
    
    def _should_extract_frame(self, frame: np.ndarray) -> bool:
        """
        Decide se um frame deve ser extraído baseado em critérios de qualidade.
        
        Args:
            frame: Frame a ser avaliado
            
        Returns:
            True se o frame deve ser extraído
        """
        if frame is None:
            return False
            
        # Verificar qualidade básica
        if not self._is_frame_quality_good(frame):
            return False
            
        # Verificar movimento (se adaptativo estiver habilitado)
        if self.config.enable_adaptive:
            if not self._has_significant_motion(frame):
                return False
                
        return True
    
    def _is_frame_quality_good(self, frame: np.ndarray) -> bool:
        """Verifica se a qualidade do frame é adequada."""
        # Verificar se o frame não está muito escuro
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if mean_brightness < 30:  # Muito escuro
            return False
            
        # Verificar se o frame não está muito borrado
        laplacian_var = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if laplacian_var < 100:  # Muito borrado
            return False
            
        return True
    
    def _has_significant_motion(self, frame: np.ndarray) -> bool:
        """Verifica se há movimento significativo no frame."""
        if self.last_frame is None:
            self.last_frame = frame
            return True
            
        # Calcular diferença entre frames
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_current, gray_last)
        motion_ratio = np.sum(diff > 30) / diff.size
        
        self.last_frame = frame
        
        return motion_ratio > self.config.motion_threshold
    
    def extract_frames_adaptive(self, video_processor) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Extrai frames usando estratégia adaptativa baseada no conteúdo.
        
        Args:
            video_processor: Processador de vídeo
            
        Yields:
            Tuplas (frame, metadata) dos frames extraídos
        """
        last_extraction_time = datetime.now()
        min_interval = timedelta(seconds=self.config.normal_min_interval)
        
        for frame in video_processor.get_frames_at_intervals(60):  # Verificar a cada minuto
            current_time = datetime.now()
            
            # Verificar se passou tempo mínimo
            if current_time - last_extraction_time < min_interval:
                continue
                
            # Verificar critérios adaptativos
            if (self._should_extract_frame(frame) and 
                self._has_significant_motion(frame)):
                
                metadata = {
                    'timestamp': current_time,
                    'frame_number': self.frame_count,
                    'extraction_method': 'adaptive',
                    'interval_seconds': (current_time - last_extraction_time).total_seconds()
                }
                
                yield frame, metadata
                last_extraction_time = current_time
                self.frame_count += 1

"""
Processador principal de vídeo para o ArboreoMonitor.

Este módulo implementa o processamento de diferentes tipos de streams de vídeo,
incluindo drones, arquivos locais e IP cams.
"""

import cv2
import numpy as np
from typing import Optional, Union, Generator, Tuple
from pathlib import Path
from enum import Enum
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VideoSource(Enum):
    """Tipos de fonte de vídeo suportados."""
    DRONE = "drone"
    FILE = "file"
    IP_CAM = "ip_cam"
    WEBCAM = "webcam"


@dataclass
class VideoMetadata:
    """Metadados de vídeo extraídos."""
    source_type: VideoSource
    width: int
    height: int
    fps: float
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    timestamp: datetime = None
    location: Optional[Tuple[float, float]] = None  # GPS coordinates
    camera_info: Optional[dict] = None


class VideoProcessor:
    """
    Processador principal de vídeo para diferentes fontes.
    
    Suporta:
    - Streams de drones (RTSP, UDP)
    - Arquivos de vídeo locais
    - IP cams (HTTP, RTSP)
    - Webcams locais
    """
    
    def __init__(self, config: dict = None):
        """
        Inicializa o processador de vídeo.
        
        Args:
            config: Configurações do processador
        """
        self.config = config or {}
        self.cap = None
        self.metadata = None
        
    def open_source(self, source: Union[str, int], source_type: VideoSource) -> bool:
        """
        Abre uma fonte de vídeo.
        
        Args:
            source: Caminho do arquivo, URL ou índice da câmera
            source_type: Tipo da fonte de vídeo
            
        Returns:
            True se a fonte foi aberta com sucesso
        """
        try:
            if source_type == VideoSource.FILE:
                self.cap = cv2.VideoCapture(str(source))
            elif source_type == VideoSource.IP_CAM:
                self.cap = cv2.VideoCapture(source)
            elif source_type == VideoSource.DRONE:
                # Para drones, assumimos stream RTSP
                self.cap = cv2.VideoCapture(source)
            elif source_type == VideoSource.WEBCAM:
                self.cap = cv2.VideoCapture(int(source))
            else:
                raise ValueError(f"Tipo de fonte não suportado: {source_type}")
                
            if not self.cap.isOpened():
                logger.error(f"Falha ao abrir fonte de vídeo: {source}")
                return False
                
            # Extrair metadados
            self.metadata = self._extract_metadata(source_type)
            logger.info(f"Fonte de vídeo aberta: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao abrir fonte de vídeo: {e}")
            return False
    
    def _extract_metadata(self, source_type: VideoSource) -> VideoMetadata:
        """Extrai metadados da fonte de vídeo."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Tentar obter duração e frame count
        frame_count = None
        duration = None
        if source_type == VideoSource.FILE:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0:
                duration = frame_count / fps
        
        return VideoMetadata(
            source_type=source_type,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            timestamp=datetime.now()
        )
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lê um frame da fonte de vídeo.
        
        Returns:
            Tupla (sucesso, frame)
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Obtém um frame em um timestamp específico.
        
        Args:
            timestamp: Timestamp em segundos
            
        Returns:
            Frame no timestamp especificado
        """
        if self.cap is None:
            return None
            
        # Definir posição no vídeo
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def get_frames_at_intervals(self, interval_seconds: float) -> Generator[np.ndarray, None, None]:
        """
        Gera frames em intervalos específicos.
        
        Args:
            interval_seconds: Intervalo entre frames em segundos
            
        Yields:
            Frames extraídos nos intervalos
        """
        if self.cap is None or self.metadata is None:
            return
            
        current_time = 0.0
        total_duration = self.metadata.duration or 3600  # Default 1 hora se não conhecido
        
        while current_time < total_duration:
            frame = self.get_frame_at_time(current_time)
            if frame is not None:
                yield frame
            current_time += interval_seconds
    
    def close(self):
        """Fecha a fonte de vídeo."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Fonte de vídeo fechada")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

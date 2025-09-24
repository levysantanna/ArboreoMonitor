"""
Módulo de processamento de vídeo para o ArboreoMonitor.

Este módulo contém todas as funcionalidades relacionadas ao processamento
de streams de vídeo, incluindo drones, arquivos de vídeo e IP cams.
"""

from .video_processor import VideoProcessor
from .stream_handler import StreamHandler
from .frame_extractor import FrameExtractor

__all__ = ['VideoProcessor', 'StreamHandler', 'FrameExtractor']

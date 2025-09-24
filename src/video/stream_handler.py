"""
Gerenciador de streams para o ArboreoMonitor.

Este módulo implementa o gerenciamento de diferentes tipos de streams de vídeo,
incluindo IP cams, drones e arquivos locais.
"""

import cv2
import requests
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from queue import Queue, Empty
import json

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuração de um stream."""
    name: str
    url: str
    stream_type: str  # 'rtsp', 'http', 'file'
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    buffer_size: int = 10


class StreamHandler:
    """
    Gerenciador de múltiplos streams de vídeo.
    
    Suporta:
    - IP cams (RTSP, HTTP)
    - Streams de drones
    - Arquivos de vídeo locais
    """
    
    def __init__(self):
        """Inicializa o gerenciador de streams."""
        self.streams: Dict[str, StreamConfig] = {}
        self.active_streams: Dict[str, cv2.VideoCapture] = {}
        self.frame_queues: Dict[str, Queue] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
        
    def add_stream(self, config: StreamConfig) -> bool:
        """
        Adiciona um novo stream.
        
        Args:
            config: Configuração do stream
            
        Returns:
            True se o stream foi adicionado com sucesso
        """
        try:
            self.streams[config.name] = config
            self.frame_queues[config.name] = Queue(maxsize=config.buffer_size)
            logger.info(f"Stream adicionado: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar stream {config.name}: {e}")
            return False
    
    def remove_stream(self, name: str) -> bool:
        """
        Remove um stream.
        
        Args:
            name: Nome do stream
            
        Returns:
            True se o stream foi removido com sucesso
        """
        try:
            if name in self.active_streams:
                self.stop_stream(name)
            
            if name in self.streams:
                del self.streams[name]
            if name in self.frame_queues:
                del self.frame_queues[name]
                
            logger.info(f"Stream removido: {name}")
            return True
        except Exception as e:
            logger.error(f"Erro ao remover stream {name}: {e}")
            return False
    
    def start_stream(self, name: str) -> bool:
        """
        Inicia um stream.
        
        Args:
            name: Nome do stream
            
        Returns:
            True se o stream foi iniciado com sucesso
        """
        if name not in self.streams:
            logger.error(f"Stream não encontrado: {name}")
            return False
            
        config = self.streams[name]
        if not config.enabled:
            logger.warning(f"Stream {name} está desabilitado")
            return False
            
        try:
            # Criar captura de vídeo
            cap = cv2.VideoCapture(config.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer
            
            if not cap.isOpened():
                logger.error(f"Falha ao abrir stream: {name}")
                return False
                
            self.active_streams[name] = cap
            
            # Iniciar thread de captura
            thread = threading.Thread(
                target=self._capture_frames,
                args=(name, config),
                daemon=True
            )
            thread.start()
            self.threads[name] = thread
            
            logger.info(f"Stream iniciado: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar stream {name}: {e}")
            return False
    
    def stop_stream(self, name: str) -> bool:
        """
        Para um stream.
        
        Args:
            name: Nome do stream
            
        Returns:
            True se o stream foi parado com sucesso
        """
        try:
            if name in self.active_streams:
                self.active_streams[name].release()
                del self.active_streams[name]
                
            if name in self.threads:
                self.threads[name].join(timeout=5)
                del self.threads[name]
                
            logger.info(f"Stream parado: {name}")
            return True
        except Exception as e:
            logger.error(f"Erro ao parar stream {name}: {e}")
            return False
    
    def start_all_streams(self) -> Dict[str, bool]:
        """
        Inicia todos os streams habilitados.
        
        Returns:
            Dicionário com status de cada stream
        """
        results = {}
        for name, config in self.streams.items():
            if config.enabled:
                results[name] = self.start_stream(name)
            else:
                results[name] = False
                logger.info(f"Stream {name} está desabilitado")
                
        return results
    
    def stop_all_streams(self) -> Dict[str, bool]:
        """
        Para todos os streams ativos.
        
        Returns:
            Dicionário com status de cada stream
        """
        results = {}
        for name in list(self.active_streams.keys()):
            results[name] = self.stop_stream(name)
        return results
    
    def get_frame(self, name: str, timeout: float = 1.0) -> Optional[tuple]:
        """
        Obtém um frame de um stream.
        
        Args:
            name: Nome do stream
            timeout: Timeout em segundos
            
        Returns:
            Tupla (frame, timestamp) ou None
        """
        if name not in self.frame_queues:
            return None
            
        try:
            frame_data = self.frame_queues[name].get(timeout=timeout)
            return frame_data
        except Empty:
            return None
    
    def _capture_frames(self, name: str, config: StreamConfig):
        """
        Captura frames de um stream em thread separada.
        
        Args:
            name: Nome do stream
            config: Configuração do stream
        """
        cap = self.active_streams[name]
        retry_count = 0
        
        while name in self.active_streams:
            try:
                ret, frame = cap.read()
                
                if not ret:
                    retry_count += 1
                    if retry_count >= config.retry_attempts:
                        logger.error(f"Falha ao capturar frame do stream {name}")
                        break
                    time.sleep(1)
                    continue
                
                retry_count = 0
                
                # Adicionar frame à fila
                frame_data = (frame, datetime.now())
                try:
                    self.frame_queues[name].put_nowait(frame_data)
                except:
                    # Fila cheia, remover frame mais antigo
                    try:
                        self.frame_queues[name].get_nowait()
                        self.frame_queues[name].put_nowait(frame_data)
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Erro ao capturar frame do stream {name}: {e}")
                time.sleep(1)
    
    def get_stream_status(self) -> Dict[str, dict]:
        """
        Obtém status de todos os streams.
        
        Returns:
            Dicionário com status de cada stream
        """
        status = {}
        
        for name, config in self.streams.items():
            is_active = name in self.active_streams
            queue_size = self.frame_queues[name].qsize() if name in self.frame_queues else 0
            
            status[name] = {
                'enabled': config.enabled,
                'active': is_active,
                'queue_size': queue_size,
                'url': config.url,
                'type': config.stream_type
            }
            
        return status
    
    def load_ipcam_list(self, file_path: str) -> bool:
        """
        Carrega lista de IP cams de um arquivo JSON.
        
        Args:
            file_path: Caminho do arquivo JSON
            
        Returns:
            True se a lista foi carregada com sucesso
        """
        try:
            with open(file_path, 'r') as f:
                ipcam_list = json.load(f)
                
            for ipcam in ipcam_list:
                config = StreamConfig(
                    name=ipcam['name'],
                    url=ipcam['url'],
                    stream_type=ipcam.get('type', 'rtsp'),
                    enabled=ipcam.get('enabled', True),
                    timeout=ipcam.get('timeout', 30)
                )
                self.add_stream(config)
                
            logger.info(f"Lista de IP cams carregada: {len(ipcam_list)} cams")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar lista de IP cams: {e}")
            return False

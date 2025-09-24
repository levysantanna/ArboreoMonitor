"""
Extrator de metadados para o ArboreoMonitor.

Este módulo implementa a extração de metadados de imagens e vídeos,
incluindo informações EXIF, GPS, timestamp e características técnicas.
"""

import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import hashlib

try:
    import cv2
    import numpy as np
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    # Define dummy classes for testing
    class Image:
        pass
    class cv2:
        pass
    class np:
        pass

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadados de uma imagem."""
    # Informações básicas
    filename: str
    file_size: int
    file_hash: str
    timestamp: datetime
    
    # Dimensões
    width: int
    height: int
    channels: int
    bit_depth: int
    
    # Informações de câmera
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[float] = None
    iso: Optional[int] = None
    
    # Localização GPS
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    gps_timestamp: Optional[datetime] = None
    
    # Características da imagem
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    dominant_colors: list = None
    
    # Metadados customizados
    custom_data: Dict[str, Any] = None


@dataclass
class VideoMetadata:
    """Metadados de um vídeo."""
    # Informações básicas
    filename: str
    file_size: int
    file_hash: str
    timestamp: datetime
    
    # Dimensões e qualidade
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    bitrate: int
    
    # Codec e formato
    codec: str
    format: str
    pixel_format: str
    
    # Localização GPS (se disponível)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    
    # Características do vídeo
    brightness: float = 0.0
    contrast: float = 0.0
    motion_level: float = 0.0
    
    # Metadados customizados
    custom_data: Dict[str, Any] = None


class MetadataExtractor:
    """
    Extrator de metadados de imagens e vídeos.
    
    Suporta:
    - Metadados EXIF de imagens
    - Metadados de vídeo (OpenCV)
    - Informações GPS
    - Características visuais
    - Metadados customizados
    """
    
    def __init__(self):
        """Inicializa o extrator de metadados."""
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    def extract_image_metadata(self, image_path: str) -> ImageMetadata:
        """
        Extrai metadados de uma imagem.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Metadados da imagem
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
            
            # Carregar imagem
            with Image.open(image_path) as img:
                # Informações básicas
                file_size = path.stat().st_size
                file_hash = self._calculate_file_hash(image_path)
                
                # Dimensões
                width, height = img.size
                channels = len(img.getbands())
                bit_depth = img.mode
                
                # Extrair EXIF
                exif_data = self._extract_exif_data(img)
                
                # Extrair GPS
                gps_data = self._extract_gps_data(exif_data)
                
                # Características visuais
                visual_features = self._extract_visual_features(img)
                
                # Metadados customizados
                custom_data = self._extract_custom_metadata(img, exif_data)
                
                return ImageMetadata(
                    filename=path.name,
                    file_size=file_size,
                    file_hash=file_hash,
                    timestamp=datetime.fromtimestamp(path.stat().st_mtime),
                    width=width,
                    height=height,
                    channels=channels,
                    bit_depth=bit_depth,
                    camera_make=exif_data.get('Make'),
                    camera_model=exif_data.get('Model'),
                    lens_model=exif_data.get('LensModel'),
                    focal_length=exif_data.get('FocalLength'),
                    aperture=exif_data.get('FNumber'),
                    shutter_speed=exif_data.get('ExposureTime'),
                    iso=exif_data.get('ISOSpeedRatings'),
                    latitude=gps_data.get('latitude'),
                    longitude=gps_data.get('longitude'),
                    altitude=gps_data.get('altitude'),
                    gps_timestamp=gps_data.get('timestamp'),
                    brightness=visual_features['brightness'],
                    contrast=visual_features['contrast'],
                    sharpness=visual_features['sharpness'],
                    dominant_colors=visual_features['dominant_colors'],
                    custom_data=custom_data
                )
                
        except Exception as e:
            logger.error(f"Erro ao extrair metadados da imagem {image_path}: {e}")
            raise
    
    def extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extrai metadados de um vídeo.
        
        Args:
            video_path: Caminho do vídeo
            
        Returns:
            Metadados do vídeo
        """
        try:
            path = Path(video_path)
            if not path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
            
            # Abrir vídeo com OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
            # Informações básicas
            file_size = path.stat().st_size
            file_hash = self._calculate_file_hash(video_path)
            
            # Propriedades do vídeo
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Codec e formato
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Características do vídeo
            visual_features = self._extract_video_visual_features(cap)
            
            # Metadados customizados
            custom_data = self._extract_video_custom_metadata(cap)
            
            cap.release()
            
            return VideoMetadata(
                filename=path.name,
                file_size=file_size,
                file_hash=file_hash,
                timestamp=datetime.fromtimestamp(path.stat().st_mtime),
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                frame_count=frame_count,
                bitrate=int(file_size * 8 / duration) if duration > 0 else 0,
                codec=codec,
                format=path.suffix.lower(),
                pixel_format="BGR",
                brightness=visual_features['brightness'],
                contrast=visual_features['contrast'],
                motion_level=visual_features['motion_level'],
                custom_data=custom_data
            )
            
        except Exception as e:
            logger.error(f"Erro ao extrair metadados do vídeo {video_path}: {e}")
            raise
    
    def _extract_exif_data(self, img) -> Dict[str, Any]:
        """Extrai dados EXIF da imagem."""
        exif_data = {}
        
        try:
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair EXIF: {e}")
            
        return exif_data
    
    def _extract_gps_data(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai dados GPS dos metadados EXIF."""
        gps_data = {}
        
        try:
            if 'GPSInfo' in exif_data:
                gps_info = exif_data['GPSInfo']
                
                # Latitude
                if 2 in gps_info and 3 in gps_info:
                    lat_deg = gps_info[2]
                    lat_min = gps_info[3]
                    lat_sec = gps_info[4] if 4 in gps_info else 0
                    gps_data['latitude'] = lat_deg[0] + lat_min[0]/60 + lat_sec[0]/3600
                    if gps_info[1] == 'S':
                        gps_data['latitude'] = -gps_data['latitude']
                
                # Longitude
                if 4 in gps_info and 5 in gps_info:
                    lon_deg = gps_info[4]
                    lon_min = gps_info[5]
                    lon_sec = gps_info[6] if 6 in gps_info else 0
                    gps_data['longitude'] = lon_deg[0] + lon_min[0]/60 + lon_sec[0]/3600
                    if gps_info[7] == 'W':
                        gps_data['longitude'] = -gps_data['longitude']
                
                # Altitude
                if 6 in gps_info:
                    gps_data['altitude'] = gps_info[6]
                
                # Timestamp GPS
                if 7 in gps_info:
                    gps_time = gps_info[7]
                    gps_date = gps_info[29] if 29 in gps_info else None
                    if gps_date and gps_time:
                        try:
                            gps_data['timestamp'] = datetime.strptime(
                                f"{gps_date} {gps_time}", "%Y:%m:%d %H:%M:%S"
                            )
                        except:
                            pass
                            
        except Exception as e:
            logger.warning(f"Erro ao extrair GPS: {e}")
            
        return gps_data
    
    def _extract_visual_features(self, img) -> Dict[str, Any]:
        """Extrai características visuais da imagem."""
        # Converter para array numpy
        img_array = np.array(img)
        
        # Converter para escala de cinza se necessário
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Brilho
        brightness = np.mean(gray)
        
        # Contraste
        contrast = np.std(gray)
        
        # Nitidez (Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Cores dominantes (simplificado)
        if len(img_array.shape) == 3:
            # Reduzir cores para análise
            img_small = cv2.resize(img_array, (150, 150))
            img_reshaped = img_small.reshape(-1, 3)
            
            # K-means para cores dominantes
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(img_reshaped)
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
        else:
            dominant_colors = []
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'dominant_colors': dominant_colors
        }
    
    def _extract_video_visual_features(self, cap) -> Dict[str, Any]:
        """Extrai características visuais do vídeo."""
        # Amostrar alguns frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(10, frame_count)
        
        brightness_values = []
        contrast_values = []
        motion_values = []
        
        prev_frame = None
        
        for i in range(0, frame_count, max(1, frame_count // sample_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brilho e contraste
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))
            
            # Movimento
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_values.append(np.mean(diff))
            
            prev_frame = gray.copy()
        
        return {
            'brightness': float(np.mean(brightness_values)) if brightness_values else 0,
            'contrast': float(np.mean(contrast_values)) if contrast_values else 0,
            'motion_level': float(np.mean(motion_values)) if motion_values else 0
        }
    
    def _extract_custom_metadata(self, img: Image.Image, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai metadados customizados."""
        custom_data = {}
        
        # Informações de orientação
        if 'Orientation' in exif_data:
            custom_data['orientation'] = exif_data['Orientation']
        
        # Informações de flash
        if 'Flash' in exif_data:
            custom_data['flash'] = exif_data['Flash']
        
        # Informações de white balance
        if 'WhiteBalance' in exif_data:
            custom_data['white_balance'] = exif_data['WhiteBalance']
        
        return custom_data
    
    def _extract_video_custom_metadata(self, cap: cv2.VideoCapture) -> Dict[str, Any]:
        """Extrai metadados customizados do vídeo."""
        custom_data = {}
        
        # Propriedades específicas do vídeo
        custom_data['fourcc'] = int(cap.get(cv2.CAP_PROP_FOURCC))
        custom_data['backend'] = cap.getBackendName()
        
        return custom_data
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash MD5 do arquivo."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_metadata(self, metadata: Any, output_path: str) -> bool:
        """
        Salva metadados em arquivo JSON.
        
        Args:
            metadata: Metadados a serem salvos
            output_path: Caminho do arquivo de saída
            
        Returns:
            True se salvo com sucesso
        """
        try:
            # Converter para dicionário
            if hasattr(metadata, '__dict__'):
                data = asdict(metadata)
            else:
                data = metadata
            
            # Converter datetime para string
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            # Salvar JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=datetime_converter, ensure_ascii=False)
            
            logger.info(f"Metadados salvos em: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {e}")
            return False
    
    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Carrega metadados de arquivo JSON.
        
        Args:
            metadata_path: Caminho do arquivo de metadados
            
        Returns:
            Metadados carregados
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Metadados carregados de: {metadata_path}")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao carregar metadados: {e}")
            return {}

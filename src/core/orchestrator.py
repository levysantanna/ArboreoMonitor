"""
Orquestrador principal do ArboreoMonitor.

Este módulo coordena todos os componentes do sistema,
gerenciando o fluxo de dados e processamento.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Warning: OpenCV not available")
    cv2 = None
    np = None

from .data_manager import DataManager
from ..video.stream_handler import StreamHandler
from ..detection.plant_detector import PlantDetector
from ..analysis.model3d_generator import Model3DGenerator
from ..analysis.growth_analyzer import GrowthAnalyzer
from ..analysis.change_detector import ChangeDetector
from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class ArboreoOrchestrator:
    """
    Orquestrador principal do sistema ArboreoMonitor.
    
    Coordena todos os componentes e gerencia o fluxo de processamento
    de dados de monitoramento de plantas.
    """
    
    def __init__(self, data_manager: DataManager, stream_handler: StreamHandler,
                 plant_detector: PlantDetector, model3d_generator: Model3DGenerator,
                 growth_analyzer: GrowthAnalyzer, change_detector: ChangeDetector,
                 metadata_extractor: MetadataExtractor, config: Dict[str, Any]):
        """
        Inicializa o orquestrador.
        
        Args:
            data_manager: Gerenciador de dados
            stream_handler: Gerenciador de streams
            plant_detector: Detector de plantas
            model3d_generator: Gerador de modelos 3D
            growth_analyzer: Analisador de crescimento
            change_detector: Detector de mudanças
            metadata_extractor: Extrator de metadados
            config: Configuração do sistema
        """
        self.data_manager = data_manager
        self.stream_handler = stream_handler
        self.plant_detector = plant_detector
        self.model3d_generator = model3d_generator
        self.growth_analyzer = growth_analyzer
        self.change_detector = change_detector
        self.metadata_extractor = metadata_extractor
        self.config = config
        
        self.processing_tasks = []
        self.running = False
        
    async def initialize(self):
        """Inicializa o orquestrador."""
        try:
            # Inicializar gerenciador de dados
            await self.data_manager.initialize()
            
            # Inicializar gerenciador de streams
            await self._initialize_streams()
            
            logger.info("Orquestrador inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar orquestrador: {e}")
            raise
    
    async def _initialize_streams(self):
        """Inicializa streams de vídeo."""
        try:
            # Iniciar todos os streams habilitados
            results = self.stream_handler.start_all_streams()
            
            active_streams = sum(1 for success in results.values() if success)
            total_streams = len(results)
            
            logger.info(f"Streams iniciados: {active_streams}/{total_streams}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar streams: {e}")
    
    async def start_processing(self):
        """Inicia o processamento de dados."""
        try:
            self.running = True
            
            # Criar tasks de processamento
            self.processing_tasks = [
                asyncio.create_task(self._process_video_streams()),
                asyncio.create_task(self._process_detected_plants()),
                asyncio.create_task(self._generate_3d_models()),
                asyncio.create_task(self._analyze_growth()),
                asyncio.create_task(self._detect_changes()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            logger.info("Processamento iniciado")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar processamento: {e}")
            raise
    
    async def stop_processing(self):
        """Para o processamento de dados."""
        try:
            self.running = False
            
            # Cancelar tasks de processamento
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            # Aguardar cancelamento
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            # Parar streams
            self.stream_handler.stop_all_streams()
            
            logger.info("Processamento parado")
            
        except Exception as e:
            logger.error(f"Erro ao parar processamento: {e}")
    
    async def _process_video_streams(self):
        """Processa streams de vídeo."""
        while self.running:
            try:
                # Obter status dos streams
                stream_status = self.stream_handler.get_stream_status()
                
                for stream_name, status in stream_status.items():
                    if status['active']:
                        # Obter frame do stream
                        frame_data = self.stream_handler.get_frame(stream_name, timeout=1.0)
                        
                        if frame_data:
                            frame, timestamp = frame_data
                            
                            # Processar frame
                            await self._process_frame(frame, timestamp, stream_name)
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro no processamento de streams: {e}")
                await asyncio.sleep(5)
    
    async def _process_frame(self, frame, timestamp: datetime, stream_name: str):
        """Processa um frame de vídeo."""
        try:
            # Extrair metadados
            metadata = self.metadata_extractor.extract_image_metadata_from_array(frame)
            
            # Detectar plantas
            detections = self.plant_detector.detect_plants(frame)
            
            if detections:
                # Salvar frame e detecções
                await self._save_frame_data(frame, detections, metadata, stream_name, timestamp)
                
                # Processar cada detecção
                for detection in detections:
                    await self._process_plant_detection(detection, frame, timestamp, stream_name)
            
        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
    
    async def _process_plant_detection(self, detection, frame, timestamp: datetime, stream_name: str):
        """Processa uma detecção de planta."""
        try:
            # Recortar região da planta
            x, y, w, h = detection.bbox
            plant_crop = frame[y:y+h, x:x+w]
            
            # Salvar imagem da planta
            plant_id = f"plant_{timestamp.strftime('%Y%m%d_%H%M%S')}_{detection.plant_type.value}"
            await self._save_plant_image(plant_crop, plant_id, timestamp)
            
            # Adicionar ao banco de dados
            await self.data_manager.add_plant_detection(
                plant_id=plant_id,
                detection=detection,
                timestamp=timestamp,
                stream_name=stream_name
            )
            
        except Exception as e:
            logger.error(f"Erro ao processar detecção de planta: {e}")
    
    async def _process_detected_plants(self):
        """Processa plantas detectadas."""
        while self.running:
            try:
                # Obter plantas não processadas
                unprocessed_plants = await self.data_manager.get_unprocessed_plants()
                
                for plant in unprocessed_plants:
                    await self._process_plant(plant)
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erro no processamento de plantas: {e}")
                await asyncio.sleep(60)
    
    async def _process_plant(self, plant: Dict[str, Any]):
        """Processa uma planta específica."""
        try:
            plant_id = plant['id']
            
            # Obter imagens da planta
            images = await self.data_manager.get_plant_images(plant_id)
            
            if len(images) >= 2:
                # Gerar modelo 3D
                await self._generate_plant_3d_model(plant_id, images)
                
                # Analisar crescimento
                await self._analyze_plant_growth(plant_id, images)
                
                # Detectar mudanças
                await self._detect_plant_changes(plant_id, images)
            
            # Marcar como processada
            await self.data_manager.mark_plant_processed(plant_id)
            
        except Exception as e:
            logger.error(f"Erro ao processar planta {plant_id}: {e}")
    
    async def _generate_plant_3d_model(self, plant_id: str, images: List[np.ndarray]):
        """Gera modelo 3D de uma planta."""
        try:
            # Gerar modelo 3D
            model = self.model3d_generator.generate_plant_model(images)
            
            # Salvar modelo
            model_path = f"data/3d_models/{plant_id}.obj"
            self.model3d_generator.export_model(model, model_path, 'obj')
            
            # Salvar metadados
            metadata_path = f"data/3d_models/{plant_id}_metadata.json"
            self.model3d_generator.save_model_metadata(model, metadata_path)
            
            # Atualizar banco de dados
            await self.data_manager.update_plant_3d_model(plant_id, model_path, model)
            
            logger.info(f"Modelo 3D gerado para planta {plant_id}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar modelo 3D para planta {plant_id}: {e}")
    
    async def _analyze_plant_growth(self, plant_id: str, images: List[np.ndarray]):
        """Analisa crescimento de uma planta."""
        try:
            # Obter medições anteriores
            previous_measurements = await self.data_manager.get_plant_measurements(plant_id)
            
            # Calcular medições atuais
            current_measurements = self._calculate_plant_measurements(images[-1])
            
            # Adicionar medição atual
            await self.data_manager.add_plant_measurement(plant_id, current_measurements)
            
            # Analisar crescimento
            all_measurements = previous_measurements + [current_measurements]
            growth_analysis = self.growth_analyzer.analyze_growth(plant_id, all_measurements)
            
            if growth_analysis:
                # Salvar análise
                await self.data_manager.save_growth_analysis(plant_id, growth_analysis)
                
                # Verificar anomalias
                if growth_analysis.anomalies:
                    await self._handle_growth_anomalies(plant_id, growth_analysis.anomalies)
            
            logger.info(f"Análise de crescimento concluída para planta {plant_id}")
            
        except Exception as e:
            logger.error(f"Erro na análise de crescimento da planta {plant_id}: {e}")
    
    async def _detect_plant_changes(self, plant_id: str, images: List[np.ndarray]):
        """Detecta mudanças em uma planta."""
        try:
            if len(images) >= 2:
                # Comparar imagens
                before_image = images[-2]
                after_image = images[-1]
                
                # Detectar mudanças
                changes = self.change_detector.detect_changes(before_image, after_image)
                
                if changes:
                    # Salvar mudanças
                    await self.data_manager.save_plant_changes(plant_id, changes)
                    
                    # Verificar severidade
                    severity_analysis = self.change_detector.analyze_change_severity(changes)
                    
                    if severity_analysis['requires_attention']:
                        await self._handle_plant_changes(plant_id, changes, severity_analysis)
            
            logger.info(f"Detecção de mudanças concluída para planta {plant_id}")
            
        except Exception as e:
            logger.error(f"Erro na detecção de mudanças da planta {plant_id}: {e}")
    
    async def _generate_3d_models(self):
        """Gera modelos 3D em lote."""
        while self.running:
            try:
                # Obter plantas que precisam de modelos 3D
                plants_needing_models = await self.data_manager.get_plants_needing_3d_models()
                
                for plant in plants_needing_models:
                    await self._generate_plant_3d_model(plant['id'], plant['images'])
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(300)  # 5 minutos
                
            except Exception as e:
                logger.error(f"Erro na geração de modelos 3D: {e}")
                await asyncio.sleep(600)
    
    async def _analyze_growth(self):
        """Analisa crescimento em lote."""
        while self.running:
            try:
                # Obter plantas que precisam de análise de crescimento
                plants_needing_analysis = await self.data_manager.get_plants_needing_growth_analysis()
                
                for plant in plants_needing_analysis:
                    await self._analyze_plant_growth(plant['id'], plant['images'])
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(600)  # 10 minutos
                
            except Exception as e:
                logger.error(f"Erro na análise de crescimento: {e}")
                await asyncio.sleep(1200)
    
    async def _detect_changes(self):
        """Detecta mudanças em lote."""
        while self.running:
            try:
                # Obter plantas que precisam de detecção de mudanças
                plants_needing_change_detection = await self.data_manager.get_plants_needing_change_detection()
                
                for plant in plants_needing_change_detection:
                    await self._detect_plant_changes(plant['id'], plant['images'])
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(1800)  # 30 minutos
                
            except Exception as e:
                logger.error(f"Erro na detecção de mudanças: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Limpa dados antigos."""
        while self.running:
            try:
                # Limpar dados antigos
                await self.data_manager.cleanup_old_data()
                
                # Aguardar antes da próxima iteração
                await asyncio.sleep(86400)  # 24 horas
                
            except Exception as e:
                logger.error(f"Erro na limpeza de dados: {e}")
                await asyncio.sleep(86400)
    
    async def _save_frame_data(self, frame, detections, metadata, stream_name: str, timestamp: datetime):
        """Salva dados do frame."""
        try:
            # Salvar frame
            frame_path = f"data/input/frames/{stream_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(frame_path, frame)
            
            # Salvar metadados
            metadata_path = f"data/metadata/{stream_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            self.metadata_extractor.save_metadata(metadata, metadata_path)
            
            # Salvar detecções
            detections_data = {
                'timestamp': timestamp.isoformat(),
                'stream_name': stream_name,
                'detections': [
                    {
                        'bbox': detection.bbox,
                        'confidence': detection.confidence,
                        'plant_type': detection.plant_type.value,
                        'area': detection.area
                    }
                    for detection in detections
                ]
            }
            
            detections_path = f"data/output/detections/{stream_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(detections_path, 'w') as f:
                json.dump(detections_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados do frame: {e}")
    
    async def _save_plant_image(self, plant_crop, plant_id: str, timestamp: datetime):
        """Salva imagem de uma planta."""
        try:
            # Salvar imagem
            image_path = f"data/output/plants/{plant_id}.jpg"
            cv2.imwrite(image_path, plant_crop)
            
            # Salvar metadados
            metadata = {
                'plant_id': plant_id,
                'timestamp': timestamp.isoformat(),
                'image_path': image_path
            }
            
            metadata_path = f"data/metadata/plants/{plant_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Erro ao salvar imagem da planta: {e}")
    
    def _calculate_plant_measurements(self, image: np.ndarray) -> Dict[str, Any]:
        """Calcula medições de uma planta."""
        try:
            # Detectar plantas na imagem
            detections = self.plant_detector.detect_plants(image)
            
            if detections:
                # Usar a maior detecção
                largest_detection = max(detections, key=lambda d: d.area)
                
                # Calcular medições
                height = largest_detection.bbox[3]  # altura em pixels
                width = largest_detection.bbox[2]   # largura em pixels
                area = largest_detection.area
                
                # Estimar volume (simplificado)
                volume = area * 0.1  # cm³
                
                # Calcular score de saúde (simplificado)
                health_score = min(1.0, largest_detection.confidence)
                
                return {
                    'timestamp': datetime.now(),
                    'height': height,
                    'width': width,
                    'area': area,
                    'volume': volume,
                    'health_score': health_score,
                    'growth_stage': 'mature'  # Simplificado
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao calcular medições da planta: {e}")
            return None
    
    async def _handle_growth_anomalies(self, plant_id: str, anomalies: List[Dict[str, Any]]):
        """Lida com anomalias de crescimento."""
        try:
            for anomaly in anomalies:
                if anomaly['severity'] == 'high':
                    # Enviar alerta
                    await self._send_alert(
                        f"Anomalia de crescimento detectada na planta {plant_id}",
                        anomaly
                    )
            
        except Exception as e:
            logger.error(f"Erro ao lidar com anomalias de crescimento: {e}")
    
    async def _handle_plant_changes(self, plant_id: str, changes: List[Dict[str, Any]], severity_analysis: Dict[str, Any]):
        """Lida com mudanças em plantas."""
        try:
            # Enviar alerta se necessário
            if severity_analysis['requires_attention']:
                await self._send_alert(
                    f"Mudanças detectadas na planta {plant_id}",
                    {
                        'changes': changes,
                        'severity': severity_analysis
                    }
                )
            
        except Exception as e:
            logger.error(f"Erro ao lidar com mudanças da planta: {e}")
    
    async def _send_alert(self, message: str, data: Dict[str, Any]):
        """Envia alerta."""
        try:
            # Log do alerta
            logger.warning(f"ALERTA: {message}")
            logger.warning(f"Dados: {data}")
            
            # Aqui você pode implementar envio de email, SMS, webhook, etc.
            # Por exemplo:
            # await self._send_email_alert(message, data)
            # await self._send_webhook_alert(message, data)
            
        except Exception as e:
            logger.error(f"Erro ao enviar alerta: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtém status do sistema."""
        try:
            return {
                'running': self.running,
                'streams': self.stream_handler.get_stream_status(),
                'processing_tasks': len(self.processing_tasks),
                'database_status': await self.data_manager.get_status(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter status do sistema: {e}")
            return {'error': str(e)}

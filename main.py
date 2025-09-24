#!/usr/bin/env python3
"""
ArboreoMonitor - Sistema de Monitoramento de Seres Arbóreos

Este é o arquivo principal do sistema ArboreoMonitor que orquestra
todas as funcionalidades de monitoramento de plantas e árvores.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from src.core.orchestrator import ArboreoOrchestrator
from src.core.data_manager import DataManager
from src.video.stream_handler import StreamHandler
from src.detection.plant_detector import PlantDetector
from src.analysis.model3d_generator import Model3DGenerator
from src.analysis.growth_analyzer import GrowthAnalyzer
from src.analysis.change_detector import ChangeDetector
from src.core.metadata_extractor import MetadataExtractor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/arboreo_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ArboreoMonitor:
    """
    Classe principal do ArboreoMonitor.
    
    Orquestra todos os componentes do sistema de monitoramento
    de plantas e árvores.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o ArboreoMonitor.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config_path = config_path or "config/config.yaml"
        self.orchestrator = None
        self.running = False
        
        # Criar diretórios necessários
        self._create_directories()
        
        # Carregar configuração
        self._load_config()
        
        # Inicializar componentes
        self._initialize_components()
    
    def _create_directories(self):
        """Cria diretórios necessários."""
        directories = [
            "logs",
            "data/input",
            "data/output",
            "data/models",
            "data/3d_models",
            "data/metadata",
            "data/analysis"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Carrega configuração do sistema."""
        try:
            import yaml
            
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Configuração padrão
                self.config = self._get_default_config()
                
            logger.info("Configuração carregada com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Retorna configuração padrão."""
        return {
            'video_processing': {
                'normal_interval': 300,      # 5 minutos
                'ipcam_interval': 3600,     # 1 hora
                'large_file_interval': 1800  # 30 minutos
            },
            'detection': {
                'min_confidence': 0.5,
                'min_plant_size': 60,       # cm
                'iou_threshold': 0.5
            },
            'analysis': {
                'growth_threshold': 0.1,    # cm/dia
                'health_threshold': 0.3,
                'change_threshold': 0.3
            },
            'database': {
                'url': 'sqlite:///data/arboreo.db'
            },
            'redis': {
                'url': 'redis://localhost:6379'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'key': 'your_api_key_here'
            }
        }
    
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema."""
        try:
            # Inicializar gerenciador de dados
            self.data_manager = DataManager(self.config)
            
            # Inicializar gerenciador de streams
            self.stream_handler = StreamHandler()
            
            # Inicializar detector de plantas
            self.plant_detector = PlantDetector(self.config.get('detection', {}))
            
            # Inicializar gerador de modelos 3D
            self.model3d_generator = Model3DGenerator(self.config.get('analysis', {}))
            
            # Inicializar analisador de crescimento
            self.growth_analyzer = GrowthAnalyzer(self.config.get('analysis', {}))
            
            # Inicializar detector de mudanças
            self.change_detector = ChangeDetector(self.config.get('analysis', {}))
            
            # Inicializar extrator de metadados
            self.metadata_extractor = MetadataExtractor()
            
            # Inicializar orquestrador
            self.orchestrator = ArboreoOrchestrator(
                data_manager=self.data_manager,
                stream_handler=self.stream_handler,
                plant_detector=self.plant_detector,
                model3d_generator=self.model3d_generator,
                growth_analyzer=self.growth_analyzer,
                change_detector=self.change_detector,
                metadata_extractor=self.metadata_extractor,
                config=self.config
            )
            
            logger.info("Componentes inicializados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar componentes: {e}")
            raise
    
    async def start(self):
        """Inicia o sistema ArboreoMonitor."""
        try:
            logger.info("Iniciando ArboreoMonitor...")
            
            # Inicializar orquestrador
            await self.orchestrator.initialize()
            
            # Carregar lista de IP cams
            await self._load_ipcam_list()
            
            # Iniciar processamento
            await self.orchestrator.start_processing()
            
            self.running = True
            logger.info("ArboreoMonitor iniciado com sucesso")
            
            # Manter sistema rodando
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Erro ao iniciar sistema: {e}")
            raise
    
    async def stop(self):
        """Para o sistema ArboreoMonitor."""
        try:
            logger.info("Parando ArboreoMonitor...")
            
            self.running = False
            
            if self.orchestrator:
                await self.orchestrator.stop_processing()
            
            logger.info("ArboreoMonitor parado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao parar sistema: {e}")
    
    async def _load_ipcam_list(self):
        """Carrega lista de IP cams para monitoramento."""
        try:
            ipcam_list_path = "data/ipcam_list.json"
            if Path(ipcam_list_path).exists():
                await self.stream_handler.load_ipcam_list(ipcam_list_path)
                logger.info("Lista de IP cams carregada")
            else:
                logger.warning("Lista de IP cams não encontrada")
                
        except Exception as e:
            logger.error(f"Erro ao carregar lista de IP cams: {e}")
    
    def setup_signal_handlers(self):
        """Configura handlers para sinais do sistema."""
        def signal_handler(signum, frame):
            logger.info(f"Sinal {signum} recebido, parando sistema...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Função principal."""
    try:
        # Criar instância do ArboreoMonitor
        monitor = ArboreoMonitor()
        
        # Configurar handlers de sinal
        monitor.setup_signal_handlers()
        
        # Iniciar sistema
        await monitor.start()
        
    except KeyboardInterrupt:
        logger.info("Interrupção do usuário, parando sistema...")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Verificar se está rodando como script principal
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None
    
    # Executar sistema
    asyncio.run(main())

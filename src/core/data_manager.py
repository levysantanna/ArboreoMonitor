"""
Gerenciador de dados para o ArboreoMonitor.

Este módulo implementa o gerenciamento de dados do sistema,
incluindo banco de dados, cache e armazenamento de arquivos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import hashlib
import shutil

try:
    import aiofiles
    import cv2
    import numpy as np
except ImportError:
    print("Warning: Some dependencies not available")
    aiofiles = None
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


class DataManager:
    """
    Gerenciador de dados do ArboreoMonitor.
    
    Gerencia:
    - Banco de dados SQLite
    - Cache Redis
    - Armazenamento de arquivos
    - Metadados
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o gerenciador de dados.
        
        Args:
            config: Configuração do sistema
        """
        self.config = config
        self.db_path = config.get('database', {}).get('url', 'sqlite:///data/arboreo.db')
        self.redis_url = config.get('redis', {}).get('url', 'redis://localhost:6379')
        
        # Converter URL do banco para caminho de arquivo
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path.replace('sqlite:///', '')
        
        self.connection = None
        self.redis_client = None
        
    async def initialize(self):
        """Inicializa o gerenciador de dados."""
        try:
            # Inicializar banco de dados
            await self._initialize_database()
            
            # Inicializar Redis
            await self._initialize_redis()
            
            logger.info("Gerenciador de dados inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar gerenciador de dados: {e}")
            raise
    
    async def _initialize_database(self):
        """Inicializa o banco de dados SQLite."""
        try:
            # Criar diretório do banco se não existir
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Conectar ao banco
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Criar tabelas
            await self._create_tables()
            
            logger.info("Banco de dados inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise
    
    async def _initialize_redis(self):
        """Inicializa o cliente Redis."""
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.from_url(self.redis_url)
            
            # Testar conexão
            await self.redis_client.ping()
            
            logger.info("Redis inicializado")
            
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}")
            self.redis_client = None
    
    async def _create_tables(self):
        """Cria tabelas do banco de dados."""
        try:
            cursor = self.connection.cursor()
            
            # Tabela de plantas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plants (
                    id TEXT PRIMARY KEY,
                    species TEXT,
                    location_lat REAL,
                    location_lon REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Tabela de detecções
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plant_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT,
                    timestamp TIMESTAMP,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    confidence REAL,
                    plant_type TEXT,
                    area REAL,
                    stream_name TEXT,
                    image_path TEXT,
                    FOREIGN KEY (plant_id) REFERENCES plants (id)
                )
            ''')
            
            # Tabela de medições
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plant_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT,
                    timestamp TIMESTAMP,
                    height REAL,
                    width REAL,
                    area REAL,
                    volume REAL,
                    health_score REAL,
                    growth_stage TEXT,
                    FOREIGN KEY (plant_id) REFERENCES plants (id)
                )
            ''')
            
            # Tabela de modelos 3D
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plant_3d_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT,
                    model_path TEXT,
                    model_format TEXT,
                    volume REAL,
                    surface_area REAL,
                    height REAL,
                    width REAL,
                    depth REAL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (plant_id) REFERENCES plants (id)
                )
            ''')
            
            # Tabela de análises de crescimento
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS growth_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT,
                    analysis_date TIMESTAMP,
                    growth_rate REAL,
                    volume_growth_rate REAL,
                    health_trend REAL,
                    anomaly_count INTEGER,
                    recommendations TEXT,
                    FOREIGN KEY (plant_id) REFERENCES plants (id)
                )
            ''')
            
            # Tabela de mudanças
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plant_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT,
                    timestamp TIMESTAMP,
                    change_type TEXT,
                    confidence REAL,
                    severity TEXT,
                    description TEXT,
                    affected_area REAL,
                    before_image_path TEXT,
                    after_image_path TEXT,
                    FOREIGN KEY (plant_id) REFERENCES plants (id)
                )
            ''')
            
            # Tabela de streams
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_streams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    url TEXT,
                    stream_type TEXT,
                    status TEXT DEFAULT 'inactive',
                    location_lat REAL,
                    location_lon REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_frame_at TIMESTAMP
                )
            ''')
            
            # Índices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_detections_plant_id ON plant_detections (plant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_detections_timestamp ON plant_detections (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_measurements_plant_id ON plant_measurements (plant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_measurements_timestamp ON plant_measurements (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_changes_plant_id ON plant_changes (plant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plant_changes_timestamp ON plant_changes (timestamp)')
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao criar tabelas: {e}")
            raise
    
    async def add_plant_detection(self, plant_id: str, detection, timestamp: datetime, stream_name: str):
        """Adiciona uma detecção de planta."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO plant_detections 
                (plant_id, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, 
                 confidence, plant_type, area, stream_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant_id,
                timestamp,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3],
                detection.confidence,
                detection.plant_type.value,
                detection.area,
                stream_name
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao adicionar detecção de planta: {e}")
    
    async def add_plant_measurement(self, plant_id: str, measurement: Dict[str, Any]):
        """Adiciona uma medição de planta."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO plant_measurements 
                (plant_id, timestamp, height, width, area, volume, health_score, growth_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant_id,
                measurement['timestamp'],
                measurement['height'],
                measurement['width'],
                measurement['area'],
                measurement['volume'],
                measurement['health_score'],
                measurement['growth_stage']
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao adicionar medição de planta: {e}")
    
    async def update_plant_3d_model(self, plant_id: str, model_path: str, model):
        """Atualiza modelo 3D de uma planta."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO plant_3d_models 
                (plant_id, model_path, model_format, volume, surface_area, height, width, depth)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant_id,
                model_path,
                'obj',
                model.volume,
                model.surface_area,
                model.height,
                model.width,
                model.depth
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar modelo 3D: {e}")
    
    async def save_growth_analysis(self, plant_id: str, analysis):
        """Salva análise de crescimento."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO growth_analyses 
                (plant_id, analysis_date, growth_rate, volume_growth_rate, 
                 health_trend, anomaly_count, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant_id,
                datetime.now(),
                analysis.growth_rate,
                analysis.volume_growth_rate,
                analysis.health_trend,
                len(analysis.anomalies),
                json.dumps(analysis.recommendations)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao salvar análise de crescimento: {e}")
    
    async def save_plant_changes(self, plant_id: str, changes: List[Dict[str, Any]]):
        """Salva mudanças de uma planta."""
        try:
            cursor = self.connection.cursor()
            
            for change in changes:
                cursor.execute('''
                    INSERT INTO plant_changes 
                    (plant_id, timestamp, change_type, confidence, severity, 
                     description, affected_area, before_image_path, after_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    plant_id,
                    change['timestamp'],
                    change['change_type'],
                    change['confidence'],
                    change['severity'],
                    change['description'],
                    change['affected_area'],
                    change.get('before_image_path'),
                    change.get('after_image_path')
                ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao salvar mudanças da planta: {e}")
    
    async def get_unprocessed_plants(self) -> List[Dict[str, Any]]:
        """Obtém plantas não processadas."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT DISTINCT p.id, p.species, p.location_lat, p.location_lon
                FROM plants p
                LEFT JOIN plant_3d_models m ON p.id = m.plant_id
                WHERE m.plant_id IS NULL
                ORDER BY p.created_at DESC
                LIMIT 10
            ''')
            
            plants = []
            for row in cursor.fetchall():
                plants.append({
                    'id': row['id'],
                    'species': row['species'],
                    'location': {
                        'latitude': row['location_lat'],
                        'longitude': row['location_lon']
                    }
                })
            
            return plants
            
        except Exception as e:
            logger.error(f"Erro ao obter plantas não processadas: {e}")
            return []
    
    async def get_plant_images(self, plant_id: str) -> List[np.ndarray]:
        """Obtém imagens de uma planta."""
        try:
            # Implementação simplificada - retornar imagens do diretório
            images_dir = Path(f"data/output/plants/{plant_id}")
            
            if not images_dir.exists():
                return []
            
            images = []
            for image_path in images_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                if image is not None:
                    images.append(image)
            
            return images
            
        except Exception as e:
            logger.error(f"Erro ao obter imagens da planta: {e}")
            return []
    
    async def get_plant_measurements(self, plant_id: str) -> List[Dict[str, Any]]:
        """Obtém medições de uma planta."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT timestamp, height, width, area, volume, health_score, growth_stage
                FROM plant_measurements
                WHERE plant_id = ?
                ORDER BY timestamp ASC
            ''', (plant_id,))
            
            measurements = []
            for row in cursor.fetchall():
                measurements.append({
                    'timestamp': datetime.fromisoformat(row['timestamp']),
                    'height': row['height'],
                    'width': row['width'],
                    'area': row['area'],
                    'volume': row['volume'],
                    'health_score': row['health_score'],
                    'growth_stage': row['growth_stage']
                })
            
            return measurements
            
        except Exception as e:
            logger.error(f"Erro ao obter medições da planta: {e}")
            return []
    
    async def mark_plant_processed(self, plant_id: str):
        """Marca planta como processada."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                UPDATE plants 
                SET updated_at = CURRENT_TIMESTAMP, status = 'processed'
                WHERE id = ?
            ''', (plant_id,))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erro ao marcar planta como processada: {e}")
    
    async def get_plants_needing_3d_models(self) -> List[Dict[str, Any]]:
        """Obtém plantas que precisam de modelos 3D."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT p.id, p.species
                FROM plants p
                LEFT JOIN plant_3d_models m ON p.id = m.plant_id
                WHERE m.plant_id IS NULL
                AND p.status = 'active'
                ORDER BY p.created_at ASC
                LIMIT 5
            ''')
            
            plants = []
            for row in cursor.fetchall():
                plant_id = row['id']
                images = await self.get_plant_images(plant_id)
                
                if len(images) >= 2:
                    plants.append({
                        'id': plant_id,
                        'species': row['species'],
                        'images': images
                    })
            
            return plants
            
        except Exception as e:
            logger.error(f"Erro ao obter plantas que precisam de modelos 3D: {e}")
            return []
    
    async def get_plants_needing_growth_analysis(self) -> List[Dict[str, Any]]:
        """Obtém plantas que precisam de análise de crescimento."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT p.id, p.species
                FROM plants p
                LEFT JOIN growth_analyses g ON p.id = g.plant_id
                WHERE g.plant_id IS NULL
                AND p.status = 'active'
                ORDER BY p.created_at ASC
                LIMIT 5
            ''')
            
            plants = []
            for row in cursor.fetchall():
                plant_id = row['id']
                images = await self.get_plant_images(plant_id)
                
                if len(images) >= 2:
                    plants.append({
                        'id': plant_id,
                        'species': row['species'],
                        'images': images
                    })
            
            return plants
            
        except Exception as e:
            logger.error(f"Erro ao obter plantas que precisam de análise de crescimento: {e}")
            return []
    
    async def get_plants_needing_change_detection(self) -> List[Dict[str, Any]]:
        """Obtém plantas que precisam de detecção de mudanças."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT p.id, p.species
                FROM plants p
                WHERE p.status = 'active'
                ORDER BY p.updated_at ASC
                LIMIT 5
            ''')
            
            plants = []
            for row in cursor.fetchall():
                plant_id = row['id']
                images = await self.get_plant_images(plant_id)
                
                if len(images) >= 2:
                    plants.append({
                        'id': plant_id,
                        'species': row['species'],
                        'images': images
                    })
            
            return plants
            
        except Exception as e:
            logger.error(f"Erro ao obter plantas que precisam de detecção de mudanças: {e}")
            return []
    
    async def cleanup_old_data(self):
        """Limpa dados antigos."""
        try:
            cursor = self.connection.cursor()
            
            # Limpar detecções antigas (mais de 30 dias)
            cursor.execute('''
                DELETE FROM plant_detections 
                WHERE timestamp < datetime('now', '-30 days')
            ''')
            
            # Limpar medições antigas (mais de 90 dias)
            cursor.execute('''
                DELETE FROM plant_measurements 
                WHERE timestamp < datetime('now', '-90 days')
            ''')
            
            # Limpar mudanças antigas (mais de 180 dias)
            cursor.execute('''
                DELETE FROM plant_changes 
                WHERE timestamp < datetime('now', '-180 days')
            ''')
            
            self.connection.commit()
            
            logger.info("Limpeza de dados antigos concluída")
            
        except Exception as e:
            logger.error(f"Erro na limpeza de dados antigos: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Obtém status do gerenciador de dados."""
        try:
            cursor = self.connection.cursor()
            
            # Contar registros
            cursor.execute('SELECT COUNT(*) as count FROM plants')
            plant_count = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM plant_detections')
            detection_count = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM plant_measurements')
            measurement_count = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM plant_3d_models')
            model_count = cursor.fetchone()['count']
            
            return {
                'database_status': 'connected',
                'plant_count': plant_count,
                'detection_count': detection_count,
                'measurement_count': measurement_count,
                'model_count': model_count,
                'redis_status': 'connected' if self.redis_client else 'disconnected'
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Fecha conexões."""
        try:
            if self.connection:
                self.connection.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Conexões fechadas")
            
        except Exception as e:
            logger.error(f"Erro ao fechar conexões: {e}")

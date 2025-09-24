"""
Analisador de crescimento para o ArboreoMonitor.

Este módulo implementa a análise de velocidade de crescimento das plantas
e detecção de mudanças ao longo do tempo.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class GrowthMeasurement:
    """Medição de crescimento de uma planta."""
    plant_id: str
    timestamp: datetime
    height: float
    width: float
    area: float
    volume: float
    health_score: float
    growth_stage: str


@dataclass
class GrowthAnalysis:
    """Análise de crescimento de uma planta."""
    plant_id: str
    measurements: List[GrowthMeasurement]
    growth_rate: float  # cm/dia
    volume_growth_rate: float  # cm³/dia
    health_trend: float  # Tendência de saúde (-1 a 1)
    growth_stage_transitions: List[Tuple[datetime, str, str]]  # (timestamp, from, to)
    anomalies: List[Dict[str, any]]  # Anomalias detectadas


class GrowthAnalyzer:
    """
    Analisador de crescimento de plantas.
    
    Suporta:
    - Análise de velocidade de crescimento
    - Detecção de mudanças de estágio
    - Análise de tendências de saúde
    - Detecção de anomalias
    """
    
    def __init__(self, config: dict = None):
        """
        Inicializa o analisador de crescimento.
        
        Args:
            config: Configuração do analisador
        """
        self.config = config or {}
        self.measurements_db = {}  # plant_id -> List[GrowthMeasurement]
        
    def add_measurement(self, measurement: GrowthMeasurement) -> bool:
        """
        Adiciona uma medição de crescimento.
        
        Args:
            measurement: Medição de crescimento
            
        Returns:
            True se adicionada com sucesso
        """
        try:
            if measurement.plant_id not in self.measurements_db:
                self.measurements_db[measurement.plant_id] = []
            
            self.measurements_db[measurement.plant_id].append(measurement)
            
            # Ordenar por timestamp
            self.measurements_db[measurement.plant_id].sort(key=lambda x: x.timestamp)
            
            logger.info(f"Medição adicionada para planta {measurement.plant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar medição: {e}")
            return False
    
    def analyze_growth(self, plant_id: str) -> Optional[GrowthAnalysis]:
        """
        Analisa o crescimento de uma planta.
        
        Args:
            plant_id: ID da planta
            
        Returns:
            Análise de crescimento ou None se não há dados suficientes
        """
        if plant_id not in self.measurements_db:
            logger.warning(f"Nenhuma medição encontrada para planta {plant_id}")
            return None
        
        measurements = self.measurements_db[plant_id]
        
        if len(measurements) < 2:
            logger.warning(f"Dados insuficientes para análise da planta {plant_id}")
            return None
        
        try:
            # Calcular taxa de crescimento
            growth_rate = self._calculate_growth_rate(measurements)
            volume_growth_rate = self._calculate_volume_growth_rate(measurements)
            
            # Analisar tendência de saúde
            health_trend = self._analyze_health_trend(measurements)
            
            # Detectar transições de estágio
            stage_transitions = self._detect_stage_transitions(measurements)
            
            # Detectar anomalias
            anomalies = self._detect_anomalies(measurements)
            
            return GrowthAnalysis(
                plant_id=plant_id,
                measurements=measurements,
                growth_rate=growth_rate,
                volume_growth_rate=volume_growth_rate,
                health_trend=health_trend,
                growth_stage_transitions=stage_transitions,
                anomalies=anomalies
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de crescimento: {e}")
            return None
    
    def _calculate_growth_rate(self, measurements: List[GrowthMeasurement]) -> float:
        """Calcula taxa de crescimento em cm/dia."""
        if len(measurements) < 2:
            return 0.0
        
        # Usar regressão linear para calcular taxa
        heights = [m.height for m in measurements]
        timestamps = [m.timestamp for m in measurements]
        
        # Converter timestamps para dias
        start_time = timestamps[0]
        days = [(t - start_time).total_seconds() / 86400 for t in timestamps]
        
        # Calcular inclinação (taxa de crescimento)
        if len(days) > 1:
            slope = np.polyfit(days, heights, 1)[0]
            return slope
        else:
            return 0.0
    
    def _calculate_volume_growth_rate(self, measurements: List[GrowthMeasurement]) -> float:
        """Calcula taxa de crescimento de volume em cm³/dia."""
        if len(measurements) < 2:
            return 0.0
        
        volumes = [m.volume for m in measurements]
        timestamps = [m.timestamp for m in measurements]
        
        # Converter timestamps para dias
        start_time = timestamps[0]
        days = [(t - start_time).total_seconds() / 86400 for t in timestamps]
        
        # Calcular inclinação
        if len(days) > 1:
            slope = np.polyfit(days, volumes, 1)[0]
            return slope
        else:
            return 0.0
    
    def _analyze_health_trend(self, measurements: List[GrowthMeasurement]) -> float:
        """Analisa tendência de saúde (-1 a 1)."""
        if len(measurements) < 2:
            return 0.0
        
        health_scores = [m.health_score for m in measurements]
        timestamps = [m.timestamp for m in measurements]
        
        # Converter timestamps para dias
        start_time = timestamps[0]
        days = [(t - start_time).total_seconds() / 86400 for t in timestamps]
        
        # Calcular tendência
        if len(days) > 1:
            slope = np.polyfit(days, health_scores, 1)[0]
            # Normalizar para -1 a 1
            return np.clip(slope, -1, 1)
        else:
            return 0.0
    
    def _detect_stage_transitions(self, measurements: List[GrowthMeasurement]) -> List[Tuple[datetime, str, str]]:
        """Detecta transições de estágio de crescimento."""
        transitions = []
        
        for i in range(1, len(measurements)):
            prev_stage = measurements[i-1].growth_stage
            curr_stage = measurements[i].growth_stage
            
            if prev_stage != curr_stage:
                transitions.append((measurements[i].timestamp, prev_stage, curr_stage))
        
        return transitions
    
    def _detect_anomalies(self, measurements: List[GrowthMeasurement]) -> List[Dict[str, any]]:
        """Detecta anomalias no crescimento."""
        anomalies = []
        
        if len(measurements) < 3:
            return anomalies
        
        # Análise de altura
        heights = [m.height for m in measurements]
        height_mean = np.mean(heights)
        height_std = np.std(heights)
        
        for i, measurement in enumerate(measurements):
            # Detectar altura anômala
            if abs(measurement.height - height_mean) > 2 * height_std:
                anomalies.append({
                    'type': 'height_anomaly',
                    'timestamp': measurement.timestamp,
                    'value': measurement.height,
                    'expected_range': (height_mean - 2*height_std, height_mean + 2*height_std),
                    'severity': 'high' if abs(measurement.height - height_mean) > 3 * height_std else 'medium'
                })
            
            # Detectar saúde muito baixa
            if measurement.health_score < 0.3:
                anomalies.append({
                    'type': 'health_anomaly',
                    'timestamp': measurement.timestamp,
                    'value': measurement.health_score,
                    'expected_range': (0.3, 1.0),
                    'severity': 'high'
                })
            
            # Detectar crescimento negativo (possível poda/corte)
            if i > 0:
                prev_measurement = measurements[i-1]
                height_change = measurement.height - prev_measurement.height
                time_diff = (measurement.timestamp - prev_measurement.timestamp).total_seconds() / 86400  # dias
                
                if height_change < -0.1 and time_diff < 7:  # Redução significativa em menos de uma semana
                    anomalies.append({
                        'type': 'negative_growth',
                        'timestamp': measurement.timestamp,
                        'value': height_change,
                        'time_diff_days': time_diff,
                        'severity': 'high',
                        'description': 'Possível poda ou corte detectado'
                    })
        
        return anomalies
    
    def compare_plants(self, plant_ids: List[str]) -> Dict[str, any]:
        """
        Compara crescimento entre múltiplas plantas.
        
        Args:
            plant_ids: Lista de IDs das plantas
            
        Returns:
            Comparação entre plantas
        """
        comparisons = {}
        
        for plant_id in plant_ids:
            analysis = self.analyze_growth(plant_id)
            if analysis:
                comparisons[plant_id] = {
                    'growth_rate': analysis.growth_rate,
                    'volume_growth_rate': analysis.volume_growth_rate,
                    'health_trend': analysis.health_trend,
                    'anomaly_count': len(analysis.anomalies),
                    'measurement_count': len(analysis.measurements)
                }
        
        # Calcular estatísticas comparativas
        if comparisons:
            growth_rates = [comp['growth_rate'] for comp in comparisons.values()]
            health_trends = [comp['health_trend'] for comp in comparisons.values()]
            
            comparisons['_statistics'] = {
                'avg_growth_rate': np.mean(growth_rates),
                'std_growth_rate': np.std(growth_rates),
                'avg_health_trend': np.mean(health_trends),
                'std_health_trend': np.std(health_trends),
                'fastest_growing': max(comparisons.keys(), key=lambda x: comparisons[x]['growth_rate']),
                'healthiest': max(comparisons.keys(), key=lambda x: comparisons[x]['health_trend'])
            }
        
        return comparisons
    
    def generate_growth_report(self, plant_id: str) -> Dict[str, any]:
        """
        Gera relatório de crescimento de uma planta.
        
        Args:
            plant_id: ID da planta
            
        Returns:
            Relatório de crescimento
        """
        analysis = self.analyze_growth(plant_id)
        if not analysis:
            return {'error': 'Dados insuficientes para análise'}
        
        # Calcular estatísticas
        measurements = analysis.measurements
        heights = [m.height for m in measurements]
        volumes = [m.volume for m in measurements]
        health_scores = [m.health_score for m in measurements]
        
        # Período de análise
        start_date = measurements[0].timestamp
        end_date = measurements[-1].timestamp
        total_days = (end_date - start_date).days
        
        report = {
            'plant_id': plant_id,
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_days': total_days
            },
            'growth_metrics': {
                'current_height': heights[-1],
                'height_growth': heights[-1] - heights[0],
                'growth_rate_cm_per_day': analysis.growth_rate,
                'current_volume': volumes[-1],
                'volume_growth': volumes[-1] - volumes[0],
                'volume_growth_rate_cm3_per_day': analysis.volume_growth_rate
            },
            'health_metrics': {
                'current_health': health_scores[-1],
                'health_trend': analysis.health_trend,
                'avg_health': np.mean(health_scores)
            },
            'stage_transitions': [
                {
                    'timestamp': t.isoformat(),
                    'from_stage': from_stage,
                    'to_stage': to_stage
                }
                for t, from_stage, to_stage in analysis.growth_stage_transitions
            ],
            'anomalies': analysis.anomalies,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        return report
    
    def _generate_recommendations(self, analysis: GrowthAnalysis) -> List[str]:
        """Gera recomendações baseadas na análise."""
        recommendations = []
        
        # Recomendações baseadas na taxa de crescimento
        if analysis.growth_rate < 0.1:
            recommendations.append("Taxa de crescimento baixa - verificar nutrientes e água")
        elif analysis.growth_rate > 2.0:
            recommendations.append("Taxa de crescimento alta - monitorar estabilidade")
        
        # Recomendações baseadas na saúde
        if analysis.health_trend < -0.3:
            recommendations.append("Tendência de saúde declinando - verificar doenças ou pragas")
        
        # Recomendações baseadas em anomalias
        high_severity_anomalies = [a for a in analysis.anomalies if a.get('severity') == 'high']
        if high_severity_anomalies:
            recommendations.append("Anomalias de alta severidade detectadas - investigação imediata recomendada")
        
        # Recomendações baseadas no estágio
        if analysis.growth_stage_transitions:
            latest_transition = analysis.growth_stage_transitions[-1]
            if latest_transition[2] == 'mature':
                recommendations.append("Planta atingiu estágio maduro - considerar poda de manutenção")
        
        return recommendations
    
    def save_analysis(self, analysis: GrowthAnalysis, output_path: str) -> bool:
        """
        Salva análise de crescimento em arquivo.
        
        Args:
            analysis: Análise de crescimento
            output_path: Caminho do arquivo de saída
            
        Returns:
            True se salvo com sucesso
        """
        try:
            # Converter para dicionário serializável
            data = {
                'plant_id': analysis.plant_id,
                'measurements': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'height': m.height,
                        'width': m.width,
                        'area': m.area,
                        'volume': m.volume,
                        'health_score': m.health_score,
                        'growth_stage': m.growth_stage
                    }
                    for m in analysis.measurements
                ],
                'growth_rate': analysis.growth_rate,
                'volume_growth_rate': analysis.volume_growth_rate,
                'health_trend': analysis.health_trend,
                'growth_stage_transitions': [
                    {
                        'timestamp': t.isoformat(),
                        'from_stage': from_stage,
                        'to_stage': to_stage
                    }
                    for t, from_stage, to_stage in analysis.growth_stage_transitions
                ],
                'anomalies': analysis.anomalies
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Análise de crescimento salva: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar análise: {e}")
            return False

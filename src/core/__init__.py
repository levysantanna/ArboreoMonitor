"""
Módulo principal do ArboreoMonitor.

Este módulo contém as funcionalidades centrais do sistema,
incluindo extração de metadados, gerenciamento de dados e orquestração.
"""

from .metadata_extractor import MetadataExtractor
from .data_manager import DataManager
from .orchestrator import ArboreoOrchestrator

__all__ = ['MetadataExtractor', 'DataManager', 'ArboreoOrchestrator']

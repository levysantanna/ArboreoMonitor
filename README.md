# ArboreoMonitor

Sistema de monitoramento inteligente de seres arbóreos a partir de 60 centímetros.

## Visão Geral

O ArboreoMonitor é um sistema avançado de monitoramento de plantas e árvores que:

- Processa streams de vídeo de drones, arquivos de vídeo e IP cams
- Detecta e segmenta plantas/árvores automaticamente
- Extrai metadados de imagens e vídeos
- Amostra frames inteligentemente (5 minutos para vídeos normais, 1 hora para IP cams)
- Recorta imagens mantendo apenas as plantas detectadas
- Gera modelos 3D das plantas
- Analisa velocidade de crescimento
- Detecta cortes, podas e mudanças nas plantas
- Monitora IP cams públicas para testes contínuos

## Arquitetura

```
src/
├── core/                    # Núcleo do sistema
├── video/                   # Processamento de vídeo
├── detection/               # Detecção de plantas
├── analysis/                # Análise e modelagem 3D
├── storage/                 # Armazenamento de dados
├── monitoring/              # Monitoramento de IP cams
└── utils/                   # Utilitários
```

## Tecnologias

- Python 3.9+
- OpenCV para processamento de vídeo
- YOLO/Detectron2 para detecção de plantas
- Open3D para modelagem 3D
- FastAPI para API REST
- PostgreSQL para banco de dados
- Redis para cache
- Docker para containerização

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

## Documentação

Consulte a pasta `docs/` para documentação detalhada.

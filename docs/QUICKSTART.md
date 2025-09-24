# Guia de Início Rápido - ArboreoMonitor

## Visão Geral

O ArboreoMonitor é um sistema inteligente de monitoramento de seres arbóreos que utiliza processamento de vídeo, deep learning e análise 3D para monitorar plantas e árvores a partir de 60 centímetros de altura.

## Instalação Rápida

### 1. Pré-requisitos

- **Python 3.9+**
- **CUDA 11.8+** (opcional, para GPU)
- **Docker** (opcional)
- **PostgreSQL** (opcional)
- **Redis** (opcional)

### 2. Instalação Automática

```bash
# Clone o repositório
git clone https://github.com/levysantanna/ArboreoMonitor.git
cd ArboreoMonitor

# Execute o script de instalação
chmod +x scripts/install.sh
./scripts/install.sh
```

### 3. Instalação Manual

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Criar diretórios
mkdir -p data/{input,output,models,3d_models,metadata,analysis}
mkdir -p logs

# Copiar configuração
cp config/config.yaml config/config.yaml
```

## Uso Básico

### 1. Iniciar o Sistema

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Iniciar sistema
python main.py
```

### 2. Usar Docker

```bash
# Construir e executar
docker-compose up -d

# Verificar status
docker-compose ps

# Ver logs
docker-compose logs -f arboreo-monitor
```

### 3. Acessar API

```bash
# Verificar saúde do sistema
curl http://localhost:8000/health

# Listar plantas
curl -H "Authorization: Bearer your_api_key" \
     http://localhost:8000/api/v1/plants
```

## Configuração

### 1. Arquivo de Configuração

Edite `config/config.yaml` para personalizar:

```yaml
# Configurações de vídeo
video_processing:
  normal_interval: 300        # 5 minutos
  ipcam_interval: 3600       # 1 hora

# Configurações de detecção
detection:
  min_confidence: 0.5
  min_plant_size: 60         # cm

# Configurações de análise
analysis:
  growth_threshold: 0.1      # cm/dia
  health_threshold: 0.3
```

### 2. IP Cams

Adicione IP cams em `data/ipcam_list.json`:

```json
[
  {
    "name": "Central Park NYC",
    "url": "http://207.251.86.238/cctv/centralpark.jpg",
    "type": "http",
    "location": "New York, USA",
    "enabled": true
  }
]
```

## Funcionalidades Principais

### 1. Detecção de Plantas

- **YOLO**: Detecção rápida de objetos
- **Detectron2**: Segmentação precisa
- **Classificação**: Identificação de espécies

### 2. Análise de Crescimento

- **Taxa de crescimento**: cm/dia
- **Volume**: cm³
- **Saúde**: Score 0-1
- **Estágios**: seedling, young, mature, old

### 3. Modelagem 3D

- **Fotogrametria**: Múltiplas imagens
- **Deep Learning**: Estimativa de profundidade
- **Exportação**: OBJ, PLY, STL

### 4. Detecção de Mudanças

- **Cortes**: Detecção de cortes/podas
- **Crescimento**: Mudanças positivas
- **Doenças**: Problemas de saúde
- **Danos**: Danos físicos

## Exemplos de Uso

### 1. Monitorar IP Cam

```python
from src.video.stream_handler import StreamHandler
from src.detection.plant_detector import PlantDetector

# Configurar stream
stream_handler = StreamHandler()
stream_handler.add_stream({
    'name': 'test_cam',
    'url': 'http://camera.example.com/stream',
    'type': 'http',
    'enabled': True
})

# Iniciar stream
stream_handler.start_stream('test_cam')

# Obter frame
frame_data = stream_handler.get_frame('test_cam')
if frame_data:
    frame, timestamp = frame_data
    
    # Detectar plantas
    detector = PlantDetector()
    detections = detector.detect_plants(frame)
    
    print(f"Plantas detectadas: {len(detections)}")
```

### 2. Analisar Crescimento

```python
from src.analysis.growth_analyzer import GrowthAnalyzer

# Criar analisador
analyzer = GrowthAnalyzer()

# Adicionar medições
analyzer.add_measurement({
    'plant_id': 'plant_001',
    'timestamp': datetime.now(),
    'height': 2.5,
    'width': 1.8,
    'area': 4.5,
    'volume': 8.1,
    'health_score': 0.85
})

# Analisar crescimento
analysis = analyzer.analyze_growth('plant_001')
if analysis:
    print(f"Taxa de crescimento: {analysis.growth_rate} cm/dia")
    print(f"Tendência de saúde: {analysis.health_trend}")
```

### 3. Gerar Modelo 3D

```python
from src.analysis.model3d_generator import Model3DGenerator

# Criar gerador
generator = Model3DGenerator()

# Gerar modelo
model = generator.generate_plant_model(images, method='photogrammetry')

# Exportar modelo
generator.export_model(model, 'plant_001.obj', 'obj')

# Salvar metadados
generator.save_model_metadata(model, 'plant_001_metadata.json')
```

### 4. Detectar Mudanças

```python
from src.analysis.change_detector import ChangeDetector

# Criar detector
detector = ChangeDetector()

# Detectar mudanças
changes = detector.detect_changes(before_image, after_image)

# Analisar severidade
severity = detector.analyze_change_severity(changes)

if severity['requires_attention']:
    print("Atenção necessária!")
    for change in changes:
        print(f"- {change['change_type']}: {change['description']}")
```

## API REST

### 1. Endpoints Principais

```bash
# Plantas
GET    /api/v1/plants                    # Listar plantas
GET    /api/v1/plants/{id}               # Obter planta
POST   /api/v1/plants                     # Criar planta

# Análise de crescimento
GET    /api/v1/plants/{id}/growth         # Análise de crescimento
GET    /api/v1/plants/compare             # Comparar plantas

# Detecção de mudanças
GET    /api/v1/plants/{id}/changes        # Mudanças da planta
POST   /api/v1/plants/{id}/changes        # Reportar mudança

# Modelos 3D
GET    /api/v1/plants/{id}/model3d        # Modelo 3D
GET    /api/v1/plants/{id}/model3d/metadata # Metadados do modelo

# Streams
GET    /api/v1/streams                    # Listar streams
POST   /api/v1/streams                    # Adicionar stream
GET    /api/v1/streams/{id}/status        # Status do stream

# Relatórios
POST   /api/v1/reports                    # Gerar relatório
GET    /api/v1/reports/{id}               # Obter relatório
```

### 2. Exemplos de Uso da API

```bash
# Obter análise de crescimento
curl -H "Authorization: Bearer your_api_key" \
     "http://localhost:8000/api/v1/plants/plant_001/growth"

# Gerar relatório
curl -X POST \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"type": "growth_analysis", "plant_ids": ["plant_001"], "format": "pdf"}' \
     "http://localhost:8000/api/v1/reports"

# Adicionar stream
curl -X POST \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"name": "New Stream", "url": "rtsp://camera.example.com/stream"}' \
     "http://localhost:8000/api/v1/streams"
```

## Monitoramento

### 1. Métricas

```bash
# Prometheus
curl http://localhost:9090/metrics

# Grafana
http://localhost:3000
```

### 2. Logs

```bash
# Logs da aplicação
tail -f logs/arboreo_monitor.log

# Logs do Docker
docker-compose logs -f arboreo-monitor
```

### 3. Health Check

```bash
# Verificar saúde
curl http://localhost:8000/health

# Status do sistema
curl -H "Authorization: Bearer your_api_key" \
     http://localhost:8000/api/v1/status
```

## Troubleshooting

### 1. Problemas Comuns

#### Erro de GPU
```bash
# Verificar CUDA
nvidia-smi
nvcc --version

# Reinstalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Erro de Memória
```bash
# Verificar uso de memória
free -h
ps aux --sort=-%mem | head

# Aumentar swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Erro de Banco de Dados
```bash
# Verificar conexão
psql -h localhost -U arboreo_user -d arboreo

# Verificar logs
tail -f /var/log/postgresql/postgresql-13-main.log
```

### 2. Logs de Debug

```bash
# Habilitar debug
export DEBUG=true
export LOG_LEVEL=DEBUG

# Verificar logs
tail -f logs/application.log
```

## Próximos Passos

1. **Configurar IP Cams**: Adicione suas câmeras em `data/ipcam_list.json`
2. **Personalizar Detecção**: Ajuste parâmetros em `config/config.yaml`
3. **Monitorar Resultados**: Use a API para acompanhar o progresso
4. **Configurar Alertas**: Configure notificações para mudanças importantes
5. **Expandir Funcionalidades**: Adicione novos tipos de análise

## Suporte

- **Documentação**: Consulte `docs/` para documentação completa
- **Issues**: Reporte problemas no GitHub
- **Discussões**: Participe das discussões da comunidade

## Licença

Este projeto está licenciado sob a MIT License. Veja o arquivo `LICENSE` para mais detalhes.

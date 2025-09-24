# API do ArboreoMonitor

## Visão Geral

A API do ArboreoMonitor fornece endpoints REST para interagir com o sistema de monitoramento de plantas. Todos os endpoints retornam JSON e seguem padrões REST.

## Base URL

```
http://localhost:8000/api/v1
```

## Autenticação

### API Key
```http
Authorization: Bearer <api_key>
```

### Exemplo
```http
curl -H "Authorization: Bearer your_api_key" \
     http://localhost:8000/api/v1/plants
```

## Endpoints

### 1. Plantas

#### Listar Plantas
```http
GET /plants
```

**Parâmetros de Query:**
- `page`: Número da página (padrão: 1)
- `limit`: Limite de resultados (padrão: 20)
- `species`: Filtrar por espécie
- `location`: Filtrar por localização
- `status`: Filtrar por status

**Resposta:**
```json
{
  "plants": [
    {
      "id": "plant_001",
      "species": "eucalyptus",
      "location": {
        "latitude": -23.5505,
        "longitude": -46.6333
      },
      "height": 2.5,
      "width": 1.8,
      "health_score": 0.85,
      "growth_stage": "mature",
      "last_updated": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

#### Obter Planta por ID
```http
GET /plants/{plant_id}
```

**Resposta:**
```json
{
  "id": "plant_001",
  "species": "eucalyptus",
  "location": {
    "latitude": -23.5505,
    "longitude": -46.6333
  },
  "measurements": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "height": 2.5,
      "width": 1.8,
      "area": 4.5,
      "volume": 8.1,
      "health_score": 0.85
    }
  ],
  "growth_analysis": {
    "growth_rate": 0.15,
    "volume_growth_rate": 0.8,
    "health_trend": 0.1,
    "anomalies": []
  },
  "3d_model": {
    "available": true,
    "url": "/api/v1/plants/plant_001/model3d",
    "format": "obj"
  }
}
```

#### Criar Nova Planta
```http
POST /plants
```

**Body:**
```json
{
  "species": "eucalyptus",
  "location": {
    "latitude": -23.5505,
    "longitude": -46.6333
  },
  "initial_measurements": {
    "height": 2.0,
    "width": 1.5,
    "health_score": 0.9
  }
}
```

**Resposta:**
```json
{
  "id": "plant_002",
  "status": "created",
  "message": "Planta criada com sucesso"
}
```

### 2. Análise de Crescimento

#### Obter Análise de Crescimento
```http
GET /plants/{plant_id}/growth
```

**Parâmetros de Query:**
- `start_date`: Data de início (ISO 8601)
- `end_date`: Data de fim (ISO 8601)
- `include_predictions`: Incluir previsões (boolean)

**Resposta:**
```json
{
  "plant_id": "plant_001",
  "analysis_period": {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-15T23:59:59Z",
    "total_days": 15
  },
  "growth_metrics": {
    "current_height": 2.5,
    "height_growth": 0.3,
    "growth_rate_cm_per_day": 0.15,
    "current_volume": 8.1,
    "volume_growth": 1.2,
    "volume_growth_rate_cm3_per_day": 0.8
  },
  "health_metrics": {
    "current_health": 0.85,
    "health_trend": 0.1,
    "avg_health": 0.82
  },
  "predictions": {
    "height_30_days": 2.8,
    "volume_30_days": 9.5,
    "health_30_days": 0.88
  },
  "recommendations": [
    "Taxa de crescimento normal",
    "Planta saudável"
  ]
}
```

#### Comparar Plantas
```http
GET /plants/compare
```

**Parâmetros de Query:**
- `plant_ids`: Lista de IDs separados por vírgula
- `metric`: Métrica para comparação (height, volume, health)

**Resposta:**
```json
{
  "comparison": {
    "plant_001": {
      "growth_rate": 0.15,
      "health_trend": 0.1,
      "anomaly_count": 0
    },
    "plant_002": {
      "growth_rate": 0.12,
      "health_trend": -0.05,
      "anomaly_count": 1
    }
  },
  "statistics": {
    "avg_growth_rate": 0.135,
    "fastest_growing": "plant_001",
    "healthiest": "plant_001"
  }
}
```

### 3. Detecção de Mudanças

#### Obter Mudanças
```http
GET /plants/{plant_id}/changes
```

**Parâmetros de Query:**
- `start_date`: Data de início
- `end_date`: Data de fim
- `change_type`: Tipo de mudança (cut, prune, growth, disease, damage)
- `severity`: Severidade (low, medium, high)

**Resposta:**
```json
{
  "plant_id": "plant_001",
  "changes": [
    {
      "id": "change_001",
      "timestamp": "2024-01-10T14:30:00Z",
      "change_type": "growth",
      "confidence": 0.85,
      "severity": "low",
      "description": "Crescimento detectado - área: 450px",
      "affected_area": 450.0,
      "before_image": "/api/v1/images/before_001.jpg",
      "after_image": "/api/v1/images/after_001.jpg"
    }
  ],
  "summary": {
    "total_changes": 1,
    "severity_distribution": {
      "low": 1,
      "medium": 0,
      "high": 0
    },
    "requires_attention": false
  }
}
```

#### Reportar Mudança
```http
POST /plants/{plant_id}/changes
```

**Body:**
```json
{
  "change_type": "cut",
  "description": "Corte detectado manualmente",
  "severity": "high",
  "timestamp": "2024-01-15T10:30:00Z",
  "images": ["image_001.jpg", "image_002.jpg"]
}
```

### 4. Modelos 3D

#### Obter Modelo 3D
```http
GET /plants/{plant_id}/model3d
```

**Parâmetros de Query:**
- `format`: Formato do modelo (obj, ply, stl)
- `quality`: Qualidade (low, medium, high)

**Resposta:**
```http
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="plant_001.obj"
```

#### Obter Metadados do Modelo 3D
```http
GET /plants/{plant_id}/model3d/metadata
```

**Resposta:**
```json
{
  "plant_id": "plant_001",
  "model_info": {
    "format": "obj",
    "file_size": 2048576,
    "vertex_count": 1024,
    "face_count": 2048,
    "bounding_box": {
      "width": 1.8,
      "height": 2.5,
      "depth": 1.2
    },
    "volume": 8.1,
    "surface_area": 15.6
  },
  "generation_info": {
    "method": "photogrammetry",
    "images_used": 12,
    "confidence": 0.85,
    "generated_at": "2024-01-15T10:30:00Z"
  }
}
```

### 5. Streams de Vídeo

#### Listar Streams
```http
GET /streams
```

**Resposta:**
```json
{
  "streams": [
    {
      "id": "stream_001",
      "name": "Central Park NYC",
      "url": "http://207.251.86.238/cctv/centralpark.jpg",
      "type": "http",
      "status": "active",
      "location": {
        "latitude": 40.7829,
        "longitude": -73.9654
      },
      "last_frame": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### Adicionar Stream
```http
POST /streams
```

**Body:**
```json
{
  "name": "New Stream",
  "url": "rtsp://camera.example.com/stream",
  "type": "rtsp",
  "location": {
    "latitude": -23.5505,
    "longitude": -46.6333
  }
}
```

#### Obter Status do Stream
```http
GET /streams/{stream_id}/status
```

**Resposta:**
```json
{
  "stream_id": "stream_001",
  "status": "active",
  "uptime": "2d 5h 30m",
  "frames_processed": 1250,
  "last_frame": "2024-01-15T10:30:00Z",
  "performance": {
    "fps": 1.0,
    "latency": 2.5,
    "error_rate": 0.01
  }
}
```

### 6. Relatórios

#### Gerar Relatório
```http
POST /reports
```

**Body:**
```json
{
  "type": "growth_analysis",
  "plant_ids": ["plant_001", "plant_002"],
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-15T23:59:59Z",
  "format": "pdf"
}
```

**Resposta:**
```json
{
  "report_id": "report_001",
  "status": "generating",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

#### Obter Relatório
```http
GET /reports/{report_id}
```

**Resposta:**
```json
{
  "report_id": "report_001",
  "status": "completed",
  "download_url": "/api/v1/reports/report_001/download",
  "generated_at": "2024-01-15T10:35:00Z",
  "file_size": 2048576
}
```

### 7. Configuração

#### Obter Configuração
```http
GET /config
```

**Resposta:**
```json
{
  "video_processing": {
    "normal_interval": 300,
    "ipcam_interval": 3600,
    "large_file_interval": 1800
  },
  "detection": {
    "min_confidence": 0.5,
    "min_plant_size": 60,
    "iou_threshold": 0.5
  },
  "analysis": {
    "growth_threshold": 0.1,
    "health_threshold": 0.3,
    "change_threshold": 0.3
  }
}
```

#### Atualizar Configuração
```http
PUT /config
```

**Body:**
```json
{
  "video_processing": {
    "normal_interval": 600
  },
  "detection": {
    "min_confidence": 0.6
  }
}
```

## Códigos de Status HTTP

- `200 OK`: Sucesso
- `201 Created`: Recurso criado
- `400 Bad Request`: Requisição inválida
- `401 Unauthorized`: Não autorizado
- `403 Forbidden`: Acesso negado
- `404 Not Found`: Recurso não encontrado
- `422 Unprocessable Entity`: Dados inválidos
- `500 Internal Server Error`: Erro interno do servidor

## Limites e Rate Limiting

### Rate Limits
- **Requests por minuto**: 100
- **Requests por hora**: 1000
- **Requests por dia**: 10000

### Headers de Rate Limiting
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Paginação

### Parâmetros
- `page`: Número da página (padrão: 1)
- `limit`: Limite de resultados (padrão: 20, máximo: 100)

### Headers de Paginação
```http
X-Total-Count: 150
X-Page-Count: 8
X-Current-Page: 1
X-Per-Page: 20
```

## Filtros e Ordenação

### Filtros
- `filter[field]=value`: Filtrar por campo
- `filter[field][gte]=value`: Maior ou igual
- `filter[field][lte]=value`: Menor ou igual
- `filter[field][in]=value1,value2`: Lista de valores

### Ordenação
- `sort=field`: Ordenar por campo
- `sort=-field`: Ordenar decrescente
- `sort=field1,field2`: Múltiplos campos

### Exemplo
```http
GET /plants?filter[species]=eucalyptus&filter[health_score][gte]=0.8&sort=-last_updated&page=1&limit=10
```

## Webhooks

### Configurar Webhook
```http
POST /webhooks
```

**Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["plant.growth", "plant.change", "plant.anomaly"],
  "secret": "your_webhook_secret"
}
```

### Eventos Disponíveis
- `plant.created`: Planta criada
- `plant.updated`: Planta atualizada
- `plant.growth`: Crescimento detectado
- `plant.change`: Mudança detectada
- `plant.anomaly`: Anomalia detectada
- `stream.connected`: Stream conectado
- `stream.disconnected`: Stream desconectado

## SDKs e Bibliotecas

### Python
```python
from arboreo_monitor import ArboreoClient

client = ArboreoClient(api_key="your_api_key")
plants = client.plants.list()
```

### JavaScript
```javascript
import { ArboreoClient } from '@arboreo/monitor-client';

const client = new ArboreoClient({ apiKey: 'your_api_key' });
const plants = await client.plants.list();
```

## Exemplos de Uso

### Monitorar Crescimento
```python
import requests

# Obter análise de crescimento
response = requests.get(
    'http://localhost:8000/api/v1/plants/plant_001/growth',
    headers={'Authorization': 'Bearer your_api_key'}
)

growth_data = response.json()
print(f"Taxa de crescimento: {growth_data['growth_metrics']['growth_rate_cm_per_day']} cm/dia")
```

### Detectar Mudanças
```python
# Verificar mudanças recentes
response = requests.get(
    'http://localhost:8000/api/v1/plants/plant_001/changes',
    params={'start_date': '2024-01-01T00:00:00Z'},
    headers={'Authorization': 'Bearer your_api_key'}
)

changes = response.json()
for change in changes['changes']:
    if change['severity'] == 'high':
        print(f"Alerta: {change['description']}")
```

### Gerar Relatório
```python
# Gerar relatório de crescimento
response = requests.post(
    'http://localhost:8000/api/v1/reports',
    json={
        'type': 'growth_analysis',
        'plant_ids': ['plant_001', 'plant_002'],
        'format': 'pdf'
    },
    headers={'Authorization': 'Bearer your_api_key'}
)

report = response.json()
print(f"Relatório gerado: {report['report_id']}")
```

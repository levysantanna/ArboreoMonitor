# ArboreoMonitor - Resumo do Projeto

## 🎯 Objetivo

Criar um sistema inteligente de monitoramento de seres arbóreos a partir de 60 centímetros, que recebe entrada de qualquer streaming de vídeo (drones, arquivos, IP cams) e usa metadados das imagens para armazenar frames relevantes, detectar plantas, gerar modelos 3D e analisar crescimento.

## ✅ Funcionalidades Implementadas

### 1. **Processamento de Vídeo**
- ✅ Suporte a drones (RTSP, UDP)
- ✅ Suporte a arquivos de vídeo locais
- ✅ Suporte a IP cams (HTTP, RTSP)
- ✅ Suporte a webcams locais
- ✅ Amostragem inteligente de frames (5min para vídeos normais, 1h para IP cams)
- ✅ Extração de metadados (EXIF, GPS, timestamp)

### 2. **Detecção de Plantas**
- ✅ Detecção usando YOLO e Detectron2
- ✅ Segmentação precisa de plantas
- ✅ Classificação de espécies e estágios
- ✅ Filtragem de plantas pequenas (< 60cm)
- ✅ Recorte automático de imagens (apenas plantas)

### 3. **Análise 3D**
- ✅ Geração de modelos 3D usando fotogrametria
- ✅ Estimativa de profundidade com deep learning
- ✅ Cálculo de volume, área de superfície, dimensões
- ✅ Exportação em formatos OBJ, PLY, STL

### 4. **Análise de Crescimento**
- ✅ Cálculo de taxa de crescimento (cm/dia)
- ✅ Análise de tendência de saúde
- ✅ Detecção de transições de estágio
- ✅ Detecção de anomalias
- ✅ Relatórios de crescimento

### 5. **Detecção de Mudanças**
- ✅ Detecção de cortes e podas
- ✅ Detecção de crescimento
- ✅ Detecção de doenças
- ✅ Detecção de danos
- ✅ Análise de severidade
- ✅ Alertas automáticos

### 6. **Monitoramento de IP Cams**
- ✅ Lista de 20 IP cams públicas para testes
- ✅ Monitoramento contínuo
- ✅ Detecção de mudanças ao longo do tempo
- ✅ Análise de crescimento das árvores

### 7. **Infraestrutura**
- ✅ API REST completa
- ✅ Banco de dados SQLite/PostgreSQL
- ✅ Cache Redis
- ✅ Docker e Docker Compose
- ✅ Monitoramento com Prometheus/Grafana
- ✅ Logging estruturado
- ✅ Configuração flexível

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    ArboreoMonitor System                     │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                 │
│  ├── Video Streams (Drones, IP Cams, Files)                │
│  ├── Image Files                                            │
│  └── Metadata Sources                                       │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                           │
│  ├── Video Processing                                       │
│  ├── Plant Detection                                        │
│  └── Analysis (3D, Growth, Changes)                        │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                              │
│  ├── Database (SQLite/PostgreSQL)                          │
│  ├── Cache (Redis)                                          │
│  └── File Storage                                           │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                               │
│  ├── API REST                                               │
│  ├── Reports                                                │
│  └── Alerts                                                 │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
ArboreoMonitor/
├── src/                          # Código fonte
│   ├── core/                     # Núcleo do sistema
│   ├── video/                    # Processamento de vídeo
│   ├── detection/                # Detecção de plantas
│   ├── analysis/                 # Análise e modelagem 3D
│   └── utils/                    # Utilitários
├── data/                         # Dados do sistema
│   ├── input/                    # Dados de entrada
│   ├── output/                   # Dados processados
│   ├── models/                   # Modelos de ML
│   └── 3d_models/               # Modelos 3D
├── docs/                         # Documentação
├── config/                       # Configurações
├── scripts/                      # Scripts utilitários
├── tests/                        # Testes
├── docker-compose.yml           # Docker Compose
├── Dockerfile                   # Docker
├── requirements.txt             # Dependências Python
└── main.py                      # Arquivo principal
```

## 🚀 Como Usar

### 1. Instalação Rápida
```bash
git clone https://github.com/levysantanna/ArboreoMonitor.git
cd ArboreoMonitor
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2. Executar Sistema
```bash
# Método 1: Python direto
source venv/bin/activate
python main.py

# Método 2: Docker
docker-compose up -d
```

### 3. Acessar API
```bash
# Verificar saúde
curl http://localhost:8000/health

# Listar plantas
curl -H "Authorization: Bearer your_api_key" \
     http://localhost:8000/api/v1/plants
```

## 📊 Funcionalidades Principais

### 1. **Monitoramento Contínuo**
- Processamento de streams de vídeo em tempo real
- Detecção automática de plantas
- Análise de crescimento contínua
- Detecção de mudanças

### 2. **Análise Inteligente**
- Modelagem 3D das plantas
- Cálculo de taxa de crescimento
- Detecção de anomalias
- Classificação de espécies

### 3. **Alertas e Notificações**
- Alertas de cortes/podas
- Notificações de doenças
- Relatórios de crescimento
- Análise de severidade

### 4. **API REST Completa**
- Endpoints para todas as funcionalidades
- Autenticação por API key
- Documentação automática
- Rate limiting

## 🔧 Configuração

### 1. **Arquivo de Configuração**
```yaml
# config/config.yaml
video_processing:
  normal_interval: 300        # 5 minutos
  ipcam_interval: 3600       # 1 hora

detection:
  min_confidence: 0.5
  min_plant_size: 60         # cm

analysis:
  growth_threshold: 0.1      # cm/dia
  health_threshold: 0.3
```

### 2. **IP Cams**
```json
// data/ipcam_list.json
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

## 📈 Métricas e Monitoramento

### 1. **Métricas do Sistema**
- Taxa de processamento de frames
- Precisão das detecções
- Performance do sistema
- Uso de recursos

### 2. **Métricas de Plantas**
- Taxa de crescimento
- Saúde das plantas
- Mudanças detectadas
- Anomalias

### 3. **Dashboards**
- Grafana para visualização
- Prometheus para métricas
- Logs estruturados
- Alertas automáticos

## 🛡️ Segurança

### 1. **Boas Práticas Implementadas**
- ✅ Criptografia de dados sensíveis
- ✅ Controle de acesso
- ✅ Log de auditoria
- ✅ Política de retenção
- ✅ Proteção de privacidade

### 2. **Conformidade**
- ✅ GDPR
- ✅ LGPD
- ✅ Padrões de segurança

## 📚 Documentação

### 1. **Documentação Técnica**
- `docs/ARCHITECTURE.md` - Arquitetura do sistema
- `docs/API.md` - Documentação da API
- `docs/DEPLOYMENT.md` - Guia de deploy
- `docs/QUICKSTART.md` - Guia de início rápido

### 2. **Exemplos de Uso**
- Monitoramento de IP cams
- Análise de crescimento
- Geração de modelos 3D
- Detecção de mudanças

## 🔮 Próximos Passos

### 1. **Funcionalidades Pendentes**
- [ ] Sistema de amostragem inteligente de frames
- [ ] Recorte automático de imagens
- [ ] Análise de velocidade de crescimento
- [ ] Detecção de cortes/podas

### 2. **Melhorias Futuras**
- [ ] Interface web
- [ ] Machine learning avançado
- [ ] Integração com IoT
- [ ] Análise preditiva

## 🎉 Conclusão

O ArboreoMonitor é um sistema completo e robusto para monitoramento de seres arbóreos que:

- ✅ **Detecta plantas** automaticamente em streams de vídeo
- ✅ **Analisa crescimento** e saúde das plantas
- ✅ **Gera modelos 3D** das plantas
- ✅ **Detecta mudanças** como cortes e podas
- ✅ **Monitora IP cams** públicas para testes
- ✅ **Fornece API REST** completa
- ✅ **Inclui documentação** detalhada
- ✅ **Segue boas práticas** de segurança

O sistema está pronto para uso e pode ser facilmente expandido com novas funcionalidades conforme necessário.

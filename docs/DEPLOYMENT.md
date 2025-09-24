# Guia de Deploy do ArboreoMonitor

## Visão Geral

Este guia fornece instruções detalhadas para fazer o deploy do ArboreoMonitor em diferentes ambientes, desde desenvolvimento local até produção em escala.

## Pré-requisitos

### Sistema
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+
- **Python**: 3.9+
- **Docker**: 20.10+ (opcional)
- **GPU**: NVIDIA GPU com CUDA (recomendado)

### Dependências
- **CUDA**: 11.8+ (para GPU)
- **cuDNN**: 8.6+ (para GPU)
- **OpenCV**: 4.8+
- **PostgreSQL**: 13+ (para produção)

## Instalação Local

### 1. Clone do Repositório
```bash
git clone https://github.com/levysantanna/ArboreoMonitor.git
cd ArboreoMonitor
```

### 2. Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalação de Dependências
```bash
pip install -r requirements.txt
```

### 4. Configuração
```bash
cp config/config.example.yaml config/config.yaml
# Editar config/config.yaml com suas configurações
```

### 5. Inicialização do Banco de Dados
```bash
python -m alembic upgrade head
```

### 6. Execução
```bash
python main.py
```

## Deploy com Docker

### 1. Docker Compose (Recomendado)

#### docker-compose.yml
```yaml
version: '3.8'

services:
  arboreo-monitor:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/arboreo
      - REDIS_URL=redis://redis:6379
      - API_KEY=your_api_key
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=arboreo
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - arboreo-monitor

volumes:
  postgres_data:
```

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar CUDA (se disponível)
RUN if [ "$CUDA_VERSION" != "" ]; then \
    apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/input data/output data/models logs

# Configurar permissões
RUN chmod +x scripts/*.sh

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["python", "main.py"]
```

### 2. Executar com Docker Compose
```bash
docker-compose up -d
```

### 3. Verificar Status
```bash
docker-compose ps
docker-compose logs -f arboreo-monitor
```

## Deploy em Produção

### 1. Kubernetes

#### namespace.yaml
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: arboreo-monitor
```

#### configmap.yaml
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: arboreo-config
  namespace: arboreo-monitor
data:
  config.yaml: |
    database:
      url: postgresql://user:password@postgres:5432/arboreo
    redis:
      url: redis://redis:6379
    api:
      key: your_api_key
      port: 8000
```

#### secret.yaml
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: arboreo-secrets
  namespace: arboreo-monitor
type: Opaque
data:
  database-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>
```

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arboreo-monitor
  namespace: arboreo-monitor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arboreo-monitor
  template:
    metadata:
      labels:
        app: arboreo-monitor
    spec:
      containers:
      - name: arboreo-monitor
        image: arboreo-monitor:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: arboreo-secrets
              key: database-url
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: arboreo-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: arboreo-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: arboreo-models-pvc
```

#### service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: arboreo-monitor-service
  namespace: arboreo-monitor
spec:
  selector:
    app: arboreo-monitor
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

#### ingress.yaml
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: arboreo-monitor-ingress
  namespace: arboreo-monitor
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - arboreo-monitor.example.com
    secretName: arboreo-monitor-tls
  rules:
  - host: arboreo-monitor.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: arboreo-monitor-service
            port:
              number: 8000
```

### 2. Deploy no Kubernetes
```bash
# Aplicar configurações
kubectl apply -f k8s/

# Verificar status
kubectl get pods -n arboreo-monitor
kubectl get services -n arboreo-monitor
kubectl get ingress -n arboreo-monitor
```

## Configuração de Banco de Dados

### 1. PostgreSQL

#### Configuração de Produção
```sql
-- Criar banco de dados
CREATE DATABASE arboreo;

-- Criar usuário
CREATE USER arboreo_user WITH PASSWORD 'secure_password';

-- Conceder permissões
GRANT ALL PRIVILEGES ON DATABASE arboreo TO arboreo_user;

-- Configurar extensões
\c arboreo;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
```

#### Configuração de Backup
```bash
# Backup diário
pg_dump -h localhost -U arboreo_user -d arboreo > backup_$(date +%Y%m%d).sql

# Restore
psql -h localhost -U arboreo_user -d arboreo < backup_20240115.sql
```

### 2. Redis

#### Configuração de Produção
```redis
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Monitoramento e Logging

### 1. Prometheus + Grafana

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arboreo-monitor'
    static_configs:
      - targets: ['arboreo-monitor:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "ArboreoMonitor Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Plant Detection Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "plant_detection_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

### 2. ELK Stack

#### Logstash Configuration
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "arboreo-monitor" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
    date {
      match => [ "timestamp", "ISO8601" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "arboreo-monitor-%{+YYYY.MM.dd}"
  }
}
```

## Segurança

### 1. SSL/TLS

#### Certificado SSL
```bash
# Gerar certificado auto-assinado (desenvolvimento)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Usar Let's Encrypt (produção)
certbot --nginx -d arboreo-monitor.example.com
```

#### Nginx SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name arboreo-monitor.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://arboreo-monitor:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Firewall

#### UFW (Ubuntu)
```bash
# Permitir apenas portas necessárias
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 5432/tcp  # PostgreSQL (apenas para admin)
ufw enable
```

#### iptables
```bash
# Configurar regras de firewall
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -j DROP
```

### 3. Autenticação

#### JWT Configuration
```python
# config/auth.py
JWT_SECRET_KEY = "your-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DELTA = timedelta(hours=24)
```

#### OAuth2 Integration
```python
# config/oauth.py
OAUTH2_CLIENT_ID = "your-client-id"
OAUTH2_CLIENT_SECRET = "your-client-secret"
OAUTH2_AUTHORIZATION_URL = "https://auth.example.com/oauth/authorize"
OAUTH2_TOKEN_URL = "https://auth.example.com/oauth/token"
```

## Backup e Recovery

### 1. Backup Automático

#### Script de Backup
```bash
#!/bin/bash
# scripts/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/arboreo-monitor"
DB_NAME="arboreo"
DB_USER="arboreo_user"
DB_HOST="localhost"

# Criar diretório de backup
mkdir -p $BACKUP_DIR

# Backup do banco de dados
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/db_$DATE.sql

# Backup dos dados
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /app/data

# Backup dos modelos
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /app/models

# Limpar backups antigos (manter apenas 30 dias)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

#### Cron Job
```bash
# Adicionar ao crontab
0 2 * * * /app/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### 2. Recovery

#### Script de Recovery
```bash
#!/bin/bash
# scripts/recovery.sh

BACKUP_DATE=$1
BACKUP_DIR="/backups/arboreo-monitor"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

# Restaurar banco de dados
psql -h localhost -U arboreo_user -d arboreo < $BACKUP_DIR/db_$BACKUP_DATE.sql

# Restaurar dados
tar -xzf $BACKUP_DIR/data_$BACKUP_DATE.tar.gz -C /

# Restaurar modelos
tar -xzf $BACKUP_DIR/models_$BACKUP_DATE.tar.gz -C /

echo "Recovery completed: $BACKUP_DATE"
```

## Monitoramento de Performance

### 1. Métricas do Sistema

#### CPU e Memória
```bash
# Instalar htop
apt-get install htop

# Monitorar recursos
htop
```

#### GPU (NVIDIA)
```bash
# Instalar nvidia-smi
nvidia-smi

# Monitorar GPU
watch -n 1 nvidia-smi
```

### 2. Logs da Aplicação

#### Estrutura de Logs
```
logs/
├── application.log
├── error.log
├── access.log
└── performance.log
```

#### Configuração de Logging
```python
# config/logging.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/application.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'arboreo_monitor': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
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

#### Habilitar Debug
```python
# config/debug.py
DEBUG = True
LOG_LEVEL = 'DEBUG'
```

#### Verificar Logs
```bash
# Logs da aplicação
tail -f logs/application.log

# Logs do sistema
journalctl -u arboreo-monitor -f

# Logs do Docker
docker-compose logs -f arboreo-monitor
```

## Atualizações

### 1. Atualização de Código

#### Git Pull
```bash
git pull origin main
pip install -r requirements.txt
python -m alembic upgrade head
```

#### Docker Update
```bash
docker-compose pull
docker-compose up -d
```

### 2. Atualização de Modelos

#### Download de Novos Modelos
```bash
python scripts/download_models.py
```

#### Verificar Modelos
```bash
python scripts/verify_models.py
```

## Manutenção

### 1. Limpeza de Dados

#### Script de Limpeza
```bash
#!/bin/bash
# scripts/cleanup.sh

# Limpar logs antigos
find logs/ -name "*.log" -mtime +30 -delete

# Limpar dados temporários
rm -rf /tmp/arboreo-*

# Limpar cache
redis-cli FLUSHDB

echo "Cleanup completed"
```

### 2. Health Checks

#### Script de Health Check
```bash
#!/bin/bash
# scripts/health_check.sh

# Verificar API
curl -f http://localhost:8000/health || exit 1

# Verificar banco de dados
psql -h localhost -U arboreo_user -d arboreo -c "SELECT 1" || exit 1

# Verificar Redis
redis-cli ping || exit 1

echo "Health check passed"
```

#### Cron Job para Health Check
```bash
# Adicionar ao crontab
*/5 * * * * /app/scripts/health_check.sh
```

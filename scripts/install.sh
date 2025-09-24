#!/bin/bash

# Script de instalação do ArboreoMonitor

set -e

echo "🌱 Instalando ArboreoMonitor..."

# Verificar se Python 3.9+ está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale Python 3.9+ primeiro."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.9+ é necessário. Versão atual: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION encontrado"

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
echo "🔄 Atualizando pip..."
pip install --upgrade pip

# Instalar dependências
echo "📚 Instalando dependências..."
pip install -r requirements.txt

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p data/{input,output,models,3d_models,metadata,analysis}
mkdir -p logs
mkdir -p config

# Copiar arquivo de configuração se não existir
if [ ! -f "config/config.yaml" ]; then
    echo "⚙️  Criando arquivo de configuração..."
    cp config/config.example.yaml config/config.yaml
fi

# Verificar se CUDA está disponível
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA detectado:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  CUDA não detectado. O sistema funcionará apenas com CPU."
fi

# Verificar se Docker está disponível
if command -v docker &> /dev/null; then
    echo "🐳 Docker detectado"
    if command -v docker-compose &> /dev/null; then
        echo "🐳 Docker Compose detectado"
    else
        echo "⚠️  Docker Compose não encontrado. Instale para usar containers."
    fi
else
    echo "⚠️  Docker não encontrado. Instale para usar containers."
fi

# Verificar se PostgreSQL está disponível
if command -v psql &> /dev/null; then
    echo "🐘 PostgreSQL detectado"
else
    echo "⚠️  PostgreSQL não encontrado. O sistema usará SQLite por padrão."
fi

# Verificar se Redis está disponível
if command -v redis-cli &> /dev/null; then
    echo "🔴 Redis detectado"
else
    echo "⚠️  Redis não encontrado. O sistema funcionará sem cache."
fi

echo ""
echo "🎉 Instalação concluída!"
echo ""
echo "Para iniciar o sistema:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Para usar Docker:"
echo "  docker-compose up -d"
echo ""
echo "Para mais informações, consulte a documentação em docs/"

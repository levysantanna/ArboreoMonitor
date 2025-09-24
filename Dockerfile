# Dockerfile para ArboreoMonitor

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
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Instalar CUDA (se disponível)
ARG CUDA_VERSION=11.8
RUN if [ "$CUDA_VERSION" != "" ]; then \
    apt-get update && apt-get install -y \
    cuda-toolkit-${CUDA_VERSION} \
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
RUN mkdir -p data/input data/output data/models data/3d_models data/metadata data/analysis logs

# Configurar permissões
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["python", "main.py"]

# 🚀 Guía de Despliegue para Diferentes Entornos

**Versión**: 3.0 | **Lily AI Virtual Companion** | **Fecha**: Febrero 2025

---

## 📖 Índice

1. [Introducción](#introducción)
2. [Despliegue en Windows (Entorno Local)](#despliegue-en-windows-entorno-local)
3. [Despliegue en Linux](#despliegue-en-linux)
4. [Despliegue en macOS](#despliegue-en-macos)
5. [Despliegue en Contenedores (Docker)](#despliegue-en-contenedores-docker)
6. [Despliegue en Nube](#despliegue-en-nube)
7. [Despliegue en Producción](#despliegue-en-producción)
8. [Configuración de Seguridad](#configuración-de-seguridad)
9. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)
10. [Consideraciones de Escalabilidad](#consideraciones-de-escalabilidad)

---

## Introducción

Esta guía proporciona instrucciones detalladas para desplegar Lily AI Virtual Companion en diferentes entornos y configuraciones. Cubre desde instalaciones locales hasta despliegues en producción en la nube.

### Tipos de Despliegue

- **Entorno Local**: Instalación en una sola máquina para uso personal
- **Entorno de Desarrollo**: Configuración para desarrollo y pruebas
- **Entorno de Pruebas**: Configuración para pruebas de integración
- **Entorno de Producción**: Configuración para uso en producción
- **Entorno en la Nube**: Despliegue en servicios cloud

---

## Despliegue en Windows (Entorno Local)

### Requisitos Previos

- **Sistema Operativo**: Windows 10 (v1809) o Windows 11
- **Python**: 3.11 o superior
- **Ollama**: Instalado y funcionando
- **Espacio en disco**: 6 GB mínimo
- **RAM**: 4 GB mínimo (8 GB recomendado)

### Método 1: Instalador Automático (Recomendado)

```cmd
# 1. Descargar el proyecto
git clone https://github.com/Mijin-VT/LILY-VIRTUAL-3.0.git
cd LILY-VIRTUAL-3.0

# 2. Ejecutar instalador automático
Lily_Setup.bat
```

### Método 2: Instalación Manual

```cmd
# 1. Clonar el repositorio
git clone https://github.com/Mijin-VT/LILY-VIRTUAL-3.0.git
cd LILY-VIRTUAL-3.0

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\\Scripts\\activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Descargar modelo Vosk
# Descargar desde: https://alphacephei.com/vosk/models
# Extraer en: models/vosk-model-small-es-0.42/

# 5. Instalar modelo Ollama
ollama pull mistral

# 6. Iniciar el sistema
INICIAR_LILY.bat
```

### Configuración Específica de Windows

#### Variables de Entorno

```cmd
# Configurar variables de entorno (opcional)
set OLLAMA_HOST=127.0.0.1
set OLLAMA_PORT=11434
set LILY_PORT=8000
```

#### Configuración de Seguridad de Windows

- **Permisos de micrófono**: Asegurar que Python tenga acceso al micrófono
- **Firewall**: Permitir conexiones locales en los puertos 8000 y 11434
- **Antivirus**: Añadir excepciones para los scripts .bat y archivos de Python

---

## Despliegue en Linux

### Requisitos Previos

- **Distribución**: Ubuntu 20.04+, Debian 11+, Fedora 35+ o equivalente
- **Python**: 3.11 o superior
- **Ollama**: Instalado y funcionando
- **Dependencias del sistema**: PortAudio, FFmpeg, etc.

### Instalación de Dependencias del Sistema

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv portaudio19-dev ffmpeg

# Fedora
sudo dnf install -y python3 python3-pip python3-devel portaudio-devel ffmpeg

# CentOS/RHEL
sudo yum install -y python3 python3-pip python3-devel portaudio-devel ffmpeg
```

### Método de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Mijin-VT/LILY-VIRTUAL-3.0.git
cd LILY-VIRTUAL-3.0

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Instalar PyAudio (puede requerir dependencias del sistema)
pip install pyaudio

# 5. Iniciar Ollama
ollama serve &

# 6. Descargar modelo
ollama pull mistral

# 7. Iniciar Lily
python main.py
```

### Solución de Problemas Comunes en Linux

#### PyAudio no instala

```bash
# Instalar dependencias del sistema primero
sudo apt install portaudio19-dev python3-pyaudio

# O instalar desde wheel
pip install PyAudio
```

#### Problemas de audio

```bash
# Verificar dispositivos de audio
arecord -l
aplay -l

# Configurar PulseAudio si es necesario
pulseaudio --start
```

---

## Despliegue en macOS

### Requisitos Previos

- **macOS**: 10.15 (Catalina) o superior
- **Python**: 3.11 o superior
- **Ollama**: Instalado y funcionando
- **Xcode Command Line Tools**: Para compilación de extensiones

### Instalación de Herramientas

```bash
# 1. Instalar Xcode Command Line Tools
xcode-select --install

# 2. Instalar Python (si no está instalado)
# Descargar desde python.org o usar Homebrew
brew install python@3.12
```

### Método de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Mijin-VT/LILY-VIRTUAL-3.0.git
cd LILY-VIRTUAL-3.0

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Instalar PyAudio con dependencias
brew install portaudio
pip install pyaudio

# 5. Iniciar Ollama
ollama serve &

# 6. Descargar modelo
ollama pull mistral

# 7. Iniciar Lily
python main.py
```

### Configuración de Seguridad en macOS

- **Acceso al micrófono**: Asegurar que Terminal/Terminal.app tenga acceso al micrófono
- **Acceso a carpetas**: Permitir acceso a la carpeta del proyecto
- **Gatekeeper**: Puede ser necesario permitir aplicaciones descargadas

---

## Despliegue en Contenedores (Docker)

### Dockerfile para Lily AI

```dockerfile
FROM python:3.12-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorio para modelos Vosk
RUN mkdir -p models

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  lily-ai:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./audio_samples:/app/audio_samples
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
    devices:
      - /dev/snd:/dev/snd  # Para acceso a audio (Linux)
    privileged: true  # Para acceso a dispositivos de audio

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Comandos de Despliegue con Docker

```bash
# 1. Construir y levantar servicios
docker-compose up --build

# 2. Iniciar en modo detached
docker-compose up -d --build

# 3. Ver logs
docker-compose logs -f lily-ai

# 4. Ejecutar comandos en contenedor
docker-compose exec lily-ai bash

# 5. Detener servicios
docker-compose down
```

### Consideraciones de Docker

#### Acceso al Audio

En Linux, es posible que necesites configurar el acceso al audio:

```bash
# Ejecutar Docker con acceso al audio
docker run --device=/dev/snd lily-ai:latest
```

#### GPU Support

Para aceleración GPU:

```yaml
# En docker-compose.yml
services:
  ollama:
    # ... otras configuraciones
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Despliegue en Nube

### AWS (Amazon Web Services)

#### EC2 Instance Setup

```bash
# 1. Crear instancia EC2 (recomendado: t3.medium o superior)
# Sistema: Ubuntu Server 22.04 LTS

# 2. Conectar a la instancia
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 4. Instalar dependencias
sudo apt install -y python3 python3-pip python3-venv git portaudio19-dev

# 5. Clonar y configurar Lily
git clone https://github.com/Mijin-VT/LILY-VIRTUAL-3.0.git
cd LILY-VIRTUAL-3.0

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Configurar firewall
sudo ufw allow 8000
sudo ufw enable

# 7. Iniciar Lily (sin audio en servidor cloud)
python main.py --host 0.0.0.0 --port 8000
```

#### Configuración de Seguridad en AWS

- **Security Groups**: Permitir tráfico en el puerto 8000
- **IAM Roles**: Configurar roles si se usan servicios adicionales
- **VPC**: Configurar correctamente para acceso externo

### Google Cloud Platform (GCP)

#### Compute Engine Setup

```bash
# Similar al setup de AWS, pero usando gcloud CLI
gcloud compute instances create lily-ai \
    --image-family ubuntu-2204-lts \
    --image-project ubuntu-os-cloud \
    --machine-type e2-medium \
    --zone us-central1-a
```

### Azure

#### Virtual Machine Setup

```bash
# Usando Azure CLI
az vm create \
    --resource-group lily-ai-rg \
    --name lily-ai-vm \
    --image Ubuntu2204 \
    --size Standard_D2s_v3 \
    --admin-username azureuser
```

---

## Despliegue en Producción

### Configuración de Producción

#### Variables de Entorno

```bash
# .env.production
OLLAMA_HOST=ollama-service
OLLAMA_PORT=11434
LILY_HOST=0.0.0.0
LILY_PORT=8000
LOG_LEVEL=INFO
DEBUG_MODE=false
MAX_WORKERS=4
TIMEOUT=300
```

#### Configuración de Servidor

```python
# main.py (producción)
import uvicorn
from main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Número de workers para producción
        log_level="info",
        timeout_keep_alive=300,
        forwarded_allow_ips="*"  # Si detrás de proxy
    )
```

### Supervisión y Logging

#### Configuración de Logging

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    # Crear directorio de logs si no existe
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar logger principal
    logger = logging.getLogger("lily_ai")
    logger.setLevel(logging.INFO)
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        "logs/lily_ai.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formateador
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Añadir handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### Balanceo de Carga

Para entornos de alta disponibilidad:

```nginx
# nginx.conf ejemplo
upstream lily_ai_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://lily_ai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Configuración de Seguridad

### HTTPS y SSL

#### Configuración con Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Autenticación (Opcional)

```python
# auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
```

### CORS Configuration

```python
# main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Opciones adicionales
    # expose_headers=["Access-Control-Allow-Origin"]
)
```

---

## Monitoreo y Mantenimiento

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Endpoint para verificación de salud del sistema"""
    import psutil
    import time
    
    # Verificar recursos del sistema
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    # Verificar servicios dependientes
    ollama_status = check_ollama_connection()
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "timestamp": time.time(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        },
        "services": {
            "ollama": ollama_status,
            "database": True,  # Si aplica
            "storage": True    # Si aplica
        }
    }
```

### Scripts de Mantenimiento

#### Limpieza de Archivos Temporales

```bash
#!/bin/bash
# cleanup.sh
# Script para limpiar archivos temporales

echo "Limpiando archivos temporales..."

# Eliminar archivos de audio antiguos (> 24 horas)
find static/audio -name "*.wav" -mmin +1440 -delete
find static/audio -name "*.mp3" -mmin +1440 -delete

# Limpiar logs antiguos (> 7 días)
find logs -name "*.log" -mtime +7 -delete

# Reiniciar servicios si es necesario
# systemctl restart lily-ai

echo "Limpieza completada."
```

#### Backup de Datos

```bash
#!/bin/bash
# backup.sh
# Script para respaldar datos importantes

BACKUP_DIR="/backup/lily-ai"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/$DATE

# Respaldo de memoria de conversación
cp -r data/conversation_memory.json $BACKUP_DIR/$DATE/

# Respaldo de configuraciones
cp -r config/ $BACKUP_DIR/$DATE/config_backup_$DATE/

# Crear archivo de verificación
echo "Backup realizado: $DATE" > $BACKUP_DIR/$DATE/backup_info.txt

# Comprimir backup
tar -czf $BACKUP_DIR/lily_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE

# Eliminar directorio temporal
rm -rf $BACKUP_DIR/$DATE

echo "Backup completado: lily_backup_$DATE.tar.gz"
```

---

## Consideraciones de Escalabilidad

### Horizontal Scaling

Para manejar más usuarios concurrentes:

1. **Multiples instancias**: Ejecutar múltiples instancias de Lily
2. **Balanceador de carga**: Distribuir tráfico entre instancias
3. **Base de datos compartida**: Usar base de datos externa para persistencia
4. **Cola de mensajes**: Para procesamiento asíncrono

### Optimización de Recursos

#### Caching

```python
# cache.py
from functools import lru_cache
import hashlib

class ResponseCache:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Eliminar el primer elemento (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
    
    def hash_input(self, text, user_id):
        return hashlib.md5(f"{text}_{user_id}".encode()).hexdigest()

# Uso en el motor de IA
cache = ResponseCache(maxsize=256)

def generate_cached_response(text, user_id):
    cache_key = cache.hash_input(text, user_id)
    
    cached_response = cache.get(cache_key)
    if cached_response:
        return cached_response
    
    # Generar respuesta normalmente
    response = ai_engine.generate_response(text, user_id)
    
    # Guardar en cache
    cache.set(cache_key, response)
    
    return response
```

#### Pool de Conexiones

```python
# connection_pool.py
import asyncio
from asyncio import Queue

class ConnectionPool:
    def __init__(self, max_connections=10):
        self.pool = Queue(maxsize=max_connections)
        self.max_connections = max_connections
        
        # Inicializar conexiones
        for _ in range(max_connections):
            self.pool.put_nowait(self.create_connection())
    
    async def acquire(self):
        return await self.pool.get()
    
    def release(self, conn):
        if self.pool.qsize() < self.max_connections:
            self.pool.put_nowait(conn)
    
    def create_connection(self):
        # Crear nueva conexión
        return {}  # Placeholder
```

### Monitoreo de Recursos

```python
# monitoring.py
import psutil
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_users: int
    response_time_avg: float
    error_rate: float

class SystemMonitor:
    def __init__(self):
        self.metrics_history = []
        self.active_requests = 0
    
    def collect_metrics(self) -> SystemMetrics:
        """Recopila métricas del sistema"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            active_users=self.get_active_users(),
            response_time_avg=self.get_avg_response_time(),
            error_rate=self.get_error_rate()
        )
    
    def should_scale_up(self, metrics: SystemMetrics) -> bool:
        """Determina si se debe escalar hacia arriba"""
        return (
            metrics.cpu_percent > 80 or
            metrics.memory_percent > 85 or
            metrics.response_time_avg > 3.0  # Más de 3 segundos
        )
    
    def get_active_users(self) -> int:
        # Implementación específica
        return 0
    
    def get_avg_response_time(self) -> float:
        # Implementación específica
        return 0.0
    
    def get_error_rate(self) -> float:
        # Implementación específica
        return 0.0
```

---

## Estrategias de Despliegue

### Blue-Green Deployment

```bash
# Ejemplo de script de blue-green deployment
#!/bin/bash

BLUE_PORT=8000
GREEN_PORT=8001
CURRENT_COLOR=blue

deploy_new_version() {
    local new_color=$1
    local new_port=$2
    
    echo "Desplegando nueva versión en $new_color..."
    
    # Detener versión antigua
    pkill -f "main.py.*$new_port" || true
    
    # Iniciar nueva versión
    cd /path/to/lily-$new_color
    source venv/bin/activate
    nohup python main.py --port $new_port > /var/log/lily-$new_color.log 2>&1 &
    
    # Esperar a que esté listo
    sleep 10
    
    # Verificar salud
    if curl -f http://localhost:$new_port/health > /dev/null 2>&1; then
        echo "Versión $new_color lista"
        update_load_balancer $new_port
        CURRENT_COLOR=$new_color
        return 0
    else
        echo "Error en despliegue de $new_color"
        return 1
    fi
}

update_load_balancer() {
    local port=$1
    # Actualizar configuración del balanceador
    sed -i "s/port [0-9]\+/port $port/" /etc/nginx/sites-available/lily
    nginx -s reload
}
```

### Canary Release

```python
# canary.py
import random
from enum import Enum

class ReleaseStrategy(Enum):
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"

class CanaryDeployment:
    def __init__(self, canary_percentage=0.1):  # 10% de tráfico
        self.canary_percentage = canary_percentage
        self.current_version = "stable"
        self.canary_version = "canary"
    
    def route_request(self, user_id: str):
        """Decide si enrutar a versión canary o estable"""
        user_hash = hash(user_id) % 100
        
        if user_hash < (self.canary_percentage * 100):
            return self.canary_version
        else:
            return self.current_version
    
    def promote_canary(self):
        """Promueve la versión canary a estable"""
        if self.evaluate_canary_performance():
            self.current_version = self.canary_version
            self.canary_version = self.generate_new_version()
            return True
        return False
    
    def evaluate_canary_performance(self) -> bool:
        # Evaluar métricas de la versión canary
        # Comparar con umbrales
        return True  # Placeholder
    
    def generate_new_version(self):
        # Generar nueva versión canary
        return f"canary_{int(time.time())}"
```

---

## Checklist de Despliegue

### Pre-Despliegue

- [ ] Pruebas unitarias completadas
- [ ] Pruebas de integración pasadas
- [ ] Pruebas de rendimiento realizadas
- [ ] Revisión de seguridad completada
- [ ] Documentación actualizada
- [ ] Copia de seguridad realizada

### Durante Despliegue

- [ ] Monitoreo activo
- [ ] Rollback planificado
- [ ] Comunicación con stakeholders
- [ ] Validación de funcionalidad

### Post-Despliegue

- [ ] Verificación de salud del sistema
- [ ] Monitoreo de métricas
- [ ] Validación de usuarios
- [ ] Documentación de cambios

---

## Recursos Adicionales

- **Documentación de FastAPI**: https://fastapi.tiangolo.com/
- **Guía de Docker**: https://docs.docker.com/
- **Ollama Documentation**: https://github.com/jmorganca/ollama
- **AWS Deployment Guide**: https://aws.amazon.com/
- **Repositorio Lily AI**: https://github.com/Mijin-VT/LILY-VIRTUAL-3.0

---

*Guía de despliegue actualizada para Lily AI Virtual 3.0*
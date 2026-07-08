# 🖥️ Requisitos del Sistema y Compatibilidad

**Versión**: 3.0 | **Lily AI Virtual Companion** | **Fecha**: Febrero 2025

---

## 📖 Índice

1. [Requisitos Mínimos del Sistema](#requisitos-mínimos-del-sistema)
2. [Requisitos Recomendados](#requisitos-recomendados)
3. [Compatibilidad de Software](#compatibilidad-de-software)
4. [Dependencias del Sistema](#dependencias-del-sistema)
5. [Requisitos de Hardware Específicos](#requisitos-de-hardware-específicos)
6. [Requisitos de Red](#requisitos-de-red)
7. [Plataformas Soportadas](#plataformas-soportadas)
8. [Limitaciones y Consideraciones](#limitaciones-y-consideraciones)

---

## Requisitos Mínimos del Sistema

### Hardware Mínimo

| Componente | Requisito Mínimo | Notas |
|------------|------------------|-------|
| **Procesador** | CPU de 64 bits Quad-core (Intel i3 / AMD Ryzen 3) | Arquitectura x86_64 requerida, 2.0 GHz o superior |
| **Memoria RAM** | 8 GB | Mínimo para ejecución de modelo Mistral 7B |
| **Almacenamiento** | 8 GB libres | Espacio para instalación, modelos Ollama y caché |
| **Micrófono** | Sí | Requerido para reconocimiento de voz |
| **Altavoces** | Sí | Para salida de audio TTS |
| **Conectividad** | Conexión a internet | Para descargas iniciales de modelos y TTS |

### Software Mínimo

| Componente | Versión Mínima | Notas |
|------------|----------------|-------|
| **Sistema Operativo** | Windows 10 (v1809) | Windows 11 también soportado |
| **Python** | 3.11 | Versión mínima requerida |
| **Ollama** | v0.1.0 | Para ejecución del modelo IA |
| **Navegador Web** | Microsoft Edge (v90+) | Otros navegadores modernos también compatibles |

---

## Requisitos Recomendados

### Hardware Recomendado

| Componente | Requisito Recomendado | Notas |
|------------|------------------------|-------|
| **Procesador** | Intel i5 / AMD Ryzen 5 o superior | Octa-core recomendado para mejor rendimiento |
| **Memoria RAM** | 16 GB o más | Para manejar modelos grandes y multitarea |
| **Almacenamiento** | 20 GB libres SSD | Para modelos más grandes, caché y mejor rendimiento |
| **GPU** | No requerida | Beneficioso para modelos más grandes si se añaden en el futuro |
| **Micrófono** | Con cancelación de ruido | Para mejor reconocimiento de voz |
| **Sistema de Audio** | Calidad decente | Para mejor experiencia TTS |

### Software Recomendado

| Componente | Versión Recomendada | Notas |
|------------|----------------------|-------|
| **Sistema Operativo** | Windows 11 | Últimas características y seguridad |
| **Python** | 3.12 o 3.13 | Últimas optimizaciones |
| **Ollama** | Versión más reciente | Mejor compatibilidad y rendimiento |
| **Navegador Web** | Microsoft Edge (última versión) | Mejor integración con la aplicación |

---

## Compatibilidad de Software

### Versiones de Python Soportadas

| Versión | Estado | Notas |
|---------|--------|-------|
| **Python 3.11** | ✅ Soportado | Versión mínima soportada |
| **Python 3.12** | ✅ Soportado | Recomendada |
| **Python 3.13** | ✅ Soportado | Recomendada |
| **Python 3.14+** | ❓ Experimental | No probado oficialmente |
| **Python < 3.11** | ❌ No soportado | No compatible con dependencias |

### Sistemas Operativos Soportados

| Sistema Operativo | Versión | Estado | Notas |
|-------------------|---------|--------|-------|
| **Windows 10** | v1809+ | ✅ Soportado | Soporte completo |
| **Windows 11** | v21H2+ | ✅ Soportado | Recomendado |
| **macOS** | 10.15+ | ⚠️ Limitado | Funcionalidad parcial |
| **Linux** | Ubuntu 20.04+, Debian 11+ | ⚠️ Limitado | Requiere ajustes manuales |

### Navegadores Web Compatibles

| Navegador | Versión Mínima | Estado | Notas |
|-----------|----------------|--------|-------|
| **Microsoft Edge** | v90 | ✅ Recomendado | Mejor integración |
| **Google Chrome** | v90 | ✅ Soportado | Totalmente funcional |
| **Mozilla Firefox** | v88 | ✅ Soportado | Funcionalidad completa |
| **Safari** | v14 | ⚠️ Limitado | Posibles limitaciones de audio |
| **Opera** | v85 | ✅ Soportado | Totalmente funcional |

---

## Dependencias del Sistema

### Dependencias de Python

| Dependencia | Versión | Propósito | Notas |
|-------------|---------|-----------|-------|
| **fastapi** | 0.115.6 | Framework web | Requerido |
| **uvicorn** | 0.34.0 | Servidor ASGI | Requerido |
| **pydantic** | 2.10.5 | Validación de datos | Requerido |
| **vosk** | 0.3.45 | Reconocimiento de voz offline | Requerido |
| **pyaudio** | 0.2.14 | Captura de audio del micrófono | Requerido |
| **pyautogui** | 0.9.54 | Control de teclado (YouTube/medios) | Requerido |
| **qwen-tts** | latest | Texto a voz neuronal | Requerido |
| **torch** | latest | Motor de inferencia de IA | Requerido |
| **soundfile** | latest | Procesamiento de audio | Requerido |
| **chromadb** | 0.4.22 | Base de datos vectorial | Requerido |
| **textblob** | 0.19.0 | Análisis de sentimientos | Requerido |
| **requests** | 2.32.3 | Solicitudes HTTP | Requerido |
| **faster-whisper** | 1.2.1 | Transcripción avanzada | Opcional |
| **sentence-transformers** | - | Embeddings semánticos | Opcional |
| **numpy** | 1.26.4 | Operaciones numéricas | Requerido |
| **pvporcupine** | 3.0.0 | Detección de wake word alternativa | Opcional |
| **pydub** | 0.25.1 | Procesamiento de audio | Requerido |

### Dependencias del Sistema

| Componente | Versión Mínima | Propósito | Notas |
|------------|----------------|-----------|-------|
| **Ollama** | v0.1.0 | Ejecución de modelos IA | Requerido |
| **Microsoft Edge** | v90 | Interfaz web | Recomendado |
| **FFmpeg** | 4.0+ | Procesamiento de audio/video | Opcional |
| **PortAudio** | 19.7+ | Soporte de audio | Requerido por PyAudio |
| **Transformers** | 4.0+ | Personalización avanzada del modelo | Opcional |
| **PyTorch** | 1.0+ | Motor de inferencia para Qwen3-TTS | Requerido |
| **FlashAttention 2** | latest | Optimización de memoria GPU | Opcional |

---

## Requisitos de Hardware Específicos

### Micrófono

**Requisitos:**
- **Tipo**: Entrada de audio (micrófono) o combo jack
- **Calidad**: CD quality (44.1kHz) o superior
- **Canal**: Mono o estéreo
- **Latencia**: Preferiblemente baja (<20ms)

**Recomendaciones:**
- Micrófonos USB con calidad decente
- Auriculares con micrófono integrado
- Micrófonos de condensador para mejor calidad
- Dispositivos con cancelación de ruido

### Tarjeta de Sonido

**Requisitos:**
- **Salida**: Altavoces o auriculares
- **Calidad**: CD quality (44.1kHz) o superior
- **Formato**: WAV, MP3, PCM

**Recomendaciones:**
- Controladores de audio actualizados
- Dispositivos de alta fidelidad para mejor experiencia TTS
- Salida de audio dedicada preferiblemente

### Conexión de Red

**Requisitos Mínimos:**
- **Velocidad**: 1 Mbps de bajada (para TTS)
- **Estabilidad**: Conexión estable para TTS
- **Protocolo**: TCP/IP, HTTP/HTTPS

**Recomendaciones:**
- 10+ Mbps para descargas rápidas de modelos
- Conexión cableada preferiblemente para estabilidad
- Sin restricciones de firewall para puertos locales

---

## Requisitos de Red

### Puertos Utilizados

| Puerto | Protocolo | Uso | Notas |
|--------|-----------|-----|-------|
| **8000** | TCP | Servidor web principal | Predeterminado |
| **11434** | TCP | API de Ollama | Requerido |
| **80/443** | TCP | Descarga inicial de modelo TTS | Solo primera vez |

### Configuración de Firewall

**Reglas Requeridas:**
- Permitir conexiones entrantes en el puerto 8000 (opcional, para acceso remoto)
- Permitir conexiones salientes para TTS y Ollama
- Permitir conexiones locales entre componentes

**Notas de Seguridad:**
- Por defecto, el servidor solo escucha en localhost (127.0.0.1)
- No se exponen servicios innecesarios a la red
- Toda la IA opera localmente

---

## Plataformas Soportadas

### Windows

| Versión | Soporte | Notas |
|---------|---------|-------|
| **Windows 10 Home** | ✅ Completo | Soporte completo |
| **Windows 10 Pro** | ✅ Completo | Soporte completo |
| **Windows 11 Home** | ✅ Completo | Recomendado |
| **Windows 11 Pro** | ✅ Completo | Recomendado |
| **Windows Server 2019** | ⚠️ Parcial | Posible con ajustes |
| **Windows Server 2022** | ⚠️ Parcial | Posible con ajustes |

### Distribuciones Linux Compatibles

| Distribución | Versión | Estado | Notas |
|--------------|---------|--------|-------|
| **Ubuntu** | 20.04+ | ⚠️ Funcional | Requiere instalación manual |
| **Debian** | 11+ | ⚠️ Funcional | Requiere instalación manual |
| **Fedora** | 35+ | ⚠️ Funcional | Requiere instalación manual |
| **CentOS** | 8+ | ⚠️ Limitado | Soporte limitado |
| **Arch Linux** | Rolling | ⚠️ Experimental | Puede requerir ajustes |

### macOS

| Versión | Soporte | Notas |
|---------|---------|-------|
| **macOS Catalina** | ⚠️ Limitado | Funcionalidad parcial |
| **macOS Big Sur** | ⚠️ Limitado | Funcionalidad parcial |
| **macOS Monterey** | ⚠️ Limitado | Funcionalidad parcial |
| **macOS Ventura** | ⚠️ Limitado | Funcionalidad parcial |
| **macOS Sonoma** | ⚠️ Limitado | Funcionalidad parcial |

---

## Limitaciones y Consideraciones

### Limitaciones de Hardware

- **Memoria RAM**: Menos de 8GB puede causar lentitud o imposibilidad de ejecutar el modelo
- **CPU**: CPUs con menos de 4 núcleos pueden tener rendimiento deficiente
- **Almacenamiento**: Espacio insuficiente impide descarga de modelos
- **Audio**: Dispositivos de audio defectuosos afectan reconocimiento de voz

### Limitaciones de Software

- **Versiones antiguas de Python**: No compatibles con dependencias modernas
- **Controladores de audio**: Pueden afectar funcionamiento de PyAudio
- **Firewall corporativo**: Puede bloquear componentes locales
- **Software antivirus**: Puede interferir con scripts .bat

### Consideraciones de Rendimiento

- **Modelos grandes**: Requieren más RAM y CPU
- **Reconocimiento de voz**: Usa CPU constantemente cuando está activo
- **TTS**: Funciona completamente offline después de descarga inicial
- **Procesamiento en segundo plano**: Puede afectar otros procesos

### Compatibilidad con Otros Software

- **Otros asistentes de voz**: Pueden interferir con el micrófono
- **Software de grabación**: Puede interferir con el reconocimiento de voz
- **Controladores de audio especiales**: Pueden requerir configuración adicional
- **Software de automatización**: Puede interferir con PyAutoGUI

---

## Verificación de Requisitos

### Script de Verificación

Puedes usar el siguiente script para verificar los requisitos básicos:

```bash
# Verificar Python
python --version

# Verificar pip
pip --version

# Verificar Ollama
ollama --version

# Verificar modelos Ollama
ollama list

# Verificar dependencias
python -c "import fastapi, vosk, pyaudio, pyautogui, soundfile; print('✓ Todas las dependencias están instaladas')"
```

### Verificación Manual

1. **Python**: Ejecuta `python --version` en CMD
2. **Ollama**: Ejecuta `ollama serve` y verifica que inicie
3. **Micrófono**: Prueba en otra aplicación
4. **Navegador**: Asegúrate de que Microsoft Edge esté instalado
5. **Conexión**: Verifica acceso a internet para descarga inicial del modelo Qwen3-TTS

---

## Solución de Problemas Comunes

### Problemas de Compatibilidad

#### ❌ Python no encontrado
- **Causa**: Python no está en el PATH
- **Solución**: Reinstalar Python marcando "Add to PATH"

#### ❌ PyAudio no instala
- **Causa**: Requiere compilación nativa en Windows
- **Solución**: Instalar desde archivo .whl precompilado

#### ❌ Vosk no funciona
- **Causa**: Modelo no descargado o ruta incorrecta
- **Solución**: Descargar modelo desde sitio oficial

#### ❌ Acceso denegado a puertos
- **Causa**: Firewall bloqueando acceso
- **Solución**: Configurar excepciones de firewall

---

## Recomendaciones de Instalación

### Para Usuarios Finales
1. Asegúrate de tener Windows 10/11 con Python 3.11+
2. Verifica tener al menos 6 GB libres de almacenamiento
3. Asegúrate de que tu micrófono funcione correctamente
4. Ejecuta el instalador automático `Lily_Setup.bat`

### Para Desarrolladores
1. Configura un entorno virtual de Python
2. Instala dependencias en orden específico
3. Configura rutas de modelos manualmente si es necesario
4. Prueba componentes individualmente antes de integración

---

## Actualizaciones y Mantenimiento

### Requisitos para Actualizaciones
- Conexión a internet para descargar nuevas versiones
- Suficiente espacio de almacenamiento
- Permisos de administrador para ciertas operaciones

### Mantenimiento Requerido
- Limpieza periódica de archivos de audio temporales
- Actualización de modelos de IA
- Mantenimiento de dependencias de Python

---

*Documentación actualizada para Lily AI Virtual 3.0*
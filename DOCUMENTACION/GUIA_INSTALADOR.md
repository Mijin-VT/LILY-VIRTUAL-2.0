# 🚀 Guía del Instalador Automático

**Versión**: 3.0 | **Archivo**: `Lily_Setup.bat`

---

## 📋 ¿Qué hace Lily_Setup.bat?

Este script de Windows automatiza completamente la instalación de Lily AI Virtual 3.0, instalando todo lo necesario:

1. ✅ **Descarga e instalación de Python** si no está presente
2. ✅ **Dependencias de Python** (FastAPI, Vosk, PyAudio, PyAutoGUI, Qwen3-TTS, etc.)
3. ✅ **Modelo Ollama Mistral 7B** (~4GB)
4. ✅ **Modelo Vosk para reconocimiento de voz** (~50MB)
5. ✅ **Configuración inicial** del entorno

---

## ⚠️ REQUISITOS PREVIOS

Antes de ejecutar el instalador, asegúrate de tener:

### 1. Python 3.11 o superior
- **Descargar**: https://www.python.org/downloads/
- **IMPORTANTE**: Durante instalación, marcar **"Add Python to PATH"**
- **Verificar**: Abrir CMD y ejecutar `python --version`

### 2. Ollama
- **Descargar**: https://ollama.ai/
- **Instalar** siguiendo las instrucciones del instalador
- **Verificar**: Abrir CMD y ejecutar `ollama --version`

### 3. Conexión a Internet
- Necesaria para descargar dependencias y modelos (~4.5GB total)
- Conexión estable recomendada (mínimo 10 Mbps)

### 4. Espacio en Disco
- **Mínimo**: 5 GB libres
- **Recomendado**: 10 GB libres

---

## 🎯 CÓMO USAR EL INSTALADOR

### Paso 1: Preparación
1. Asegúrate de tener **Python** y **Ollama** instalados
2. Cierra aplicaciones que usen mucho ancho de banda
3. Ten paciencia: el proceso puede tardar 15-30 minutos según tu conexión

### Paso 2: Ejecutar el Instalador

1. **Haz doble clic** en `Lily_Setup.bat`
2. Lee la información inicial que aparece en la pantalla
3. Presiona cualquier tecla para continuar

### Paso 3: Proceso Automático

El script ejecutará automáticamente estos pasos:

```
[PASO 1/5] Verificando Python...
[PASO 2/5] Instalando dependencias de Python...
[PASO 3/5] Verificando Ollama...
[PASO 4/5] Descargando modelo Mistral 7B... (~4GB)
[PASO 5/5] Verificando instalación...
```

### Novedades de la Versión 3.0

La versión actualizada incluye nuevas características avanzadas:

- **Sistema emocional más avanzado**: Con memoria emocional jerárquica y análisis de contexto emocional
- **Personalidades emocionales avanzadas**: Múltiples perfiles emocionales (Original, Cariñosa, Juguetona, Profesional)
- **Personalización del modelo de lenguaje**: Soporte para diferentes estilos de comunicación
- **Soporte multilingüe**: Capacidad de comunicarse en español e inglés
- **Análisis de tono de voz simulado**: Para detectar emociones en la entonación
- **Sistema de predicción emocional**: Para anticipar estados emocionales

### Paso 4: Verificación Final

Al finalizar, verás un resumen:

```
✅ Python instalado
✅ Dependencias de Python instaladas
✅ Ollama detectado
✅ Modelo Mistral instalado
✅ Instalación completada exitosamente
```

---

## ⏱️ TIEMPOS ESTIMADOS

| Componente | Tamaño | Tiempo (100 Mbps) | Tiempo (10 Mbps) |
|------------|--------|-------------------|------------------|
| Dependencias Python | ~200 MB | 1-2 minutos | 3-5 minutos |
| Modelo Mistral 7B | ~4 GB | 5-7 minutos | 50-60 minutos |
| Configuración | - | 1-2 minutos | 1-2 minutos |
| **TOTAL** | **~4.2 GB** | **7-10 minutos** | **55-70 minutos** |

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### ❌ "Python no está instalado o no está en PATH"

**Causa**: Python no se instaló correctamente o no se agregó al PATH.

**Solución**:
1. Reinstalar Python desde https://www.python.org/
2. Durante instalación, **MARCAR** "Add Python to PATH"
3. Reiniciar la computadora
4. Ejecutar instalador nuevamente

---

### ❌ "Ollama no está instalado"

**Causa**: Ollama no se encuentra instalado en el sistema.

**Solución**:
1. Descargar Ollama desde https://ollama.ai/
2. Instalar Ollama (requiere reinicio)
3. Ejecutar instalador nuevamente

---

### ❌ "No se pudo descargar el modelo Mistral"

**Causa**: Problemas de conexión o Ollama no está ejecutándose.

**Solución manual**:
1. Abrir CMD como administrador
2. Ejecutar:
   ```cmd
   ollama pull mistral
   ```
3. Esperar a que termine la descarga (~4GB)
4. Verificar con: `ollama list`

---

### ❌ Error al instalar dependencias de Python

**Causa**: Permisos insuficientes o pip desactualizado.

**Solución**:
1. Abrir CMD como **Administrador**
2. Navegar a la carpeta del proyecto:
   ```cmd
   cd "C:\Ruta\A\LILY-VIRTUAL-3.0"
   ```
3. Actualizar pip:
   ```cmd
   python -m pip install --upgrade pip
   ```
4. Instalar dependencias manualmente:
   ```cmd
   pip install -r requirements.txt
   ```

---

### ❌ "Acceso denegado" o "Permission denied"

**Causa**: Windows bloquea la ejecución de scripts.

**Solución**:
1. Clic derecho en `Lily_Setup.bat`
2. Seleccionar **"Ejecutar como administrador"**
3. Aceptar el mensaje de Control de Cuentas de Usuario (UAC)

---

### ❌ Error con PyAudio

**Causa**: PyAudio requiere compilación nativa y puede fallar en Windows.

**Solución**:
1. Descargar el archivo .whl adecuado desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Instalar manualmente:
   ```cmd
   pip install PyAudio-X.X.X-cpxx-cpxxm-win_amd64.whl
   ```
   (Reemplazar X.X.X y cpxx con la versión correcta)

---

### ❌ Error con Vosk

**Causa**: El modelo Vosk no está instalado o no se encuentra.

**Solución**:
1. Descargar modelo desde: https://alphacephei.com/vosk/models
2. Descargar `vosk-model-small-es-0.42.zip`
3. Extraer en la carpeta `models/vosk-model-small-es-0.42/`

---

### ❌ Error de puerto ocupado

**Causa**: El puerto 8000 ya está en uso por otra aplicación.

**Solución**:
1. Encontrar proceso usando el puerto:
   ```cmd
   netstat -ano | findstr :8000
   ```
2. Terminar el proceso:
   ```cmd
   taskkill /PID [PID] /F
   ```
   (Reemplazar [PID] con el número del proceso)
3. Alternativamente, cambiar puerto en `main.py`:
   ```python
   uvicorn.run("main:app", host="0.0.0.0", port=8001)  # Cambiar a 8001
   ```

---

### ❌ Error de permisos de firewall

**Causa**: Windows Firewall bloquea la aplicación.

**Solución**:
1. Ir a Configuración > Actualización y seguridad > Windows Defender Firewall
2. Permitir una aplicación a través del firewall
3. Buscar `python.exe` y marcar como permitido

---

### ❌ Error de modelo Vosk no encontrado

**Causa**: El modelo Vosk no está en la ubicación esperada.

**Solución**:
1. Verificar que el modelo esté en `models/vosk-model-small-es-0.42/`
2. Asegurarse de que el modelo tenga el nombre correcto de carpeta
3. Descargar nuevamente si es necesario desde: https://alphacephei.com/vosk/models

---

### ❌ Error de micrófono no disponible

**Causa**: El sistema no puede acceder al micrófono.

**Solución**:
1. Verificar permisos de micrófono en Configuración de Windows
2. Asegurarse de que no haya otras aplicaciones usando el micrófono
3. Probar el micrófono en otra aplicación para verificar funcionamiento

---

### ❌ Error de audio no reproduce

**Causa**: Problemas con la reproducción de audio TTS.

**Solución**:
1. Verificar volumen del sistema
2. Probar conexión a internet (requerida solo para descarga inicial del modelo TTS)
3. Verificar dispositivos de audio en Configuración de Windows

---

## ✅ DESPUÉS DE LA INSTALACIÓN

Una vez completada la instalación:

1. **Ejecuta** `INICIAR_LILY.bat` (hacer doble clic)
2. **Espera** a que se abra Microsoft Edge automáticamente
3. **Di "LILY"** para probar el wake word
4. **¡Disfruta** de tu compañera virtual!

---

## 📝 NOTAS IMPORTANTES

### Descargas Grandes
- **Modelo Mistral**: ~4GB (puede tardar en conexiones lentas)
- **Dependencias**: ~200MB
- Asegúrate de tener suficiente espacio en disco

### Primera Ejecución
- La primera vez que uses Lily, puede tardar unos segundos en cargar los modelos
- Esto es normal y solo ocurre en la primera ejecución

### Actualizaciones
- Si actualizas Lily a una nueva versión, puedes ejecutar este instalador nuevamente
- Solo descargará e instalará lo que falte o esté desactualizado

### Antivirus
- Algunos antivirus pueden detectar scripts .bat como sospechosos
- Si ocurre, añade una excepción para la carpeta de Lily
- El código es completamente seguro y de código abierto

---

## 🔄 REINSTALACIÓN COMPLETA

Si necesitas reinstalar todo desde cero:

### 1. Eliminar modelos
```cmd
# Eliminar modelo Mistral
ollama rm mistral

# Eliminar modelo Vosk (si existe)
# Borrar carpeta: models\vosk-model-small-es-0.42\
```

### 2. Desinstalar dependencias
```cmd
pip uninstall -r requirements.txt -y
```

### 3. Ejecutar instalador nuevamente
- Doble clic en `Lily_Setup.bat`

---

## 📊 RESUMEN DE COMPONENTES INSTALADOS

| Componente | Versión | Propósito |
|------------|---------|-----------|
| FastAPI | 0.115.6 | Framework web |
| Uvicorn | 0.34.0 | Servidor ASGI |
| Vosk | 0.3.45 | Reconocimiento de voz offline |
| PyAudio | 0.2.14 | Captura de audio del micrófono |
| PyAutoGUI | 0.9.54 | Control de teclado (YouTube/medios) |
| Qwen3-TTS | latest | Texto a voz neuronal |
| ChromaDB | 0.4.22 | Base de datos vectorial |
| TextBlob | 0.19.0 | Análisis de sentimientos |
| Pydantic | 2.10.5 | Validación de datos |
| Mistral 7B | Latest | Modelo de lenguaje vía Ollama |

---

## 🎉 ¡Listo!

Después de ejecutar `Lily_Setup.bat` exitosamente, tu sistema estará completamente configurado para ejecutar Lily AI con todas sus funcionalidades:

- 🎤 **Reconocimiento de voz offline** (Vosk)
- 🎵 **Control de YouTube y medios**
- 💬 **Chat con IA** (Mistral 7B)
- ❤️ **Inteligencia emocional**
- 🧠 **Memoria persistente**

**¡Disfruta de Lily AI Virtual 3.0!** 💕

---

## 📞 Soporte

Si encuentras problemas durante la instalación:

1. Revisa la sección de **Solución de Problemas** arriba
2. Consulta `DOCUMENTACION/SOLUCION_BAT_SE_CIERRA.md`
3. Reporta issues en: https://github.com/Mijin-VT/LILY-VIRTUAL-3.0/issues

---

*Documentación actualizada para LILY-VIRTUAL-3.0*

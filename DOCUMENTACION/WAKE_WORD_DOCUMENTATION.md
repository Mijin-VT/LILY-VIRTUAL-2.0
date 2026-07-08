# 🎤 Sistema de Wake Word - Documentación Técnica

**Versión**: 3.0 | **Componente**: Vosk Wake Word Engine | **Fecha**: Febrero 2025

---

## 📖 Índice

1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Configuración y Personalización](#configuración-y-personalización)
4. [API Endpoints](#api-endpoints)
5. [Implementación Técnica](#implementación-técnica)
6. [Troubleshooting](#troubleshooting)
7. [Consideraciones de Rendimiento](#consideraciones-de-rendimiento)

---

## Introducción

El sistema de wake word de Lily AI permite activar la aplicación mediante una palabra clave específica ("LILY") detectada por voz, sin necesidad de conexión a internet. Utiliza el motor Vosk para procesamiento de voz 100% offline.

### Características Principales

- **100% Offline**: Todo el procesamiento ocurre localmente
- **Detección en tiempo real**: Escucha continuamente la palabra clave
- **Bajo consumo de recursos**: Uso eficiente de CPU y memoria
- **Personalizable**: Configurable para diferentes palabras clave
- **Multiplataforma**: Compatible con Windows, macOS y Linux

---

## Arquitectura del Sistema

### Componentes del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│  Vosk Wake Word │───▶│  Callback       │
│   Input         │    │  Engine         │    │  Handler        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Audio Stream    │
                       │  Processing      │
                       └──────────────────┘
```

### Flujo de Ejecución

1. **Inicio**: El motor de wake word se inicializa con un modelo Vosk y una palabra clave
2. **Escucha**: El sistema abre un stream de audio del micrófono
3. **Procesamiento**: Los datos de audio se procesan continuamente con Vosk
4. **Detección**: Cuando se detecta la palabra clave, se activa un callback
5. **Respuesta**: El sistema puede iniciar una acción (como activar el modo de chat)

---

## Configuración y Personalización

### Parámetros de Configuración

| Parámetro | Tipo | Valor Predeterminado | Descripción |
|-----------|------|---------------------|-------------|
| `wake_word` | string | "LILY" | Palabra clave para activar el sistema |
| `model_path` | string | "models/vosk-model-small-es-0.42" | Ruta al modelo Vosk |
| `sample_rate` | integer | 16000 | Frecuencia de muestreo del audio |
| `wake_word_callback` | function | None | Función a llamar cuando se detecta la palabra clave |

### Personalización de la Palabra Clave

Para cambiar la palabra clave predeterminada:

```python
from models.vosk_wake_word_engine import VoskWakeWordEngine

def on_wake_word_detected():
    print("¡Wake word detectada!")

# Crear instancia con palabra clave personalizada
engine = VoskWakeWordEngine(
    wake_word_callback=on_wake_word_detected,
    wake_word="ALEXA",  # Cambiar a tu palabra deseada
    model_path="models/vosk-model-small-es-0.42"
)
```

### Selección de Modelo

El sistema soporta diferentes modelos Vosk:

- **`vosk-model-small-es-0.42`**: Modelo pequeño (~50MB), ideal para wake word
- **`vosk-model-es-0.42`**: Modelo grande (~1.4GB), mayor precisión
- **`vosk-model-small-en-us-0.15`**: Modelo en inglés

---

## API Endpoints

### Endpoints Disponibles

#### POST `/api/wake_word/enable`

Habilita la detección de wake word.

**Respuesta Exitosa (200 OK)**:
```json
{
  "status": "success",
  "message": "Detección de palabra clave habilitada"
}
```

**Ejemplo de Uso**:
```bash
curl -X POST http://127.0.0.1:8000/api/wake_word/enable
```

#### POST `/api/wake_word/disable`

Deshabilita la detección de wake word.

**Respuesta Exitosa (200 OK)**:
```json
{
  "status": "success",
  "message": "Detección de palabra clave deshabilitada"
}
```

**Ejemplo de Uso**:
```bash
curl -X POST http://127.0.0.1:8000/api/wake_word/disable
```

#### GET `/api/wake_word/status`

Obtiene el estado actual del sistema de wake word.

**Respuesta Exitosa (200 OK)**:
```json
{
  "enabled": true,
  "is_listening": false
}
```

**Campos de Respuesta**:
- `enabled` (boolean): Si la detección está habilitada
- `is_listening` (boolean): Si el sistema está escuchando actualmente

**Ejemplo de Uso**:
```bash
curl http://127.0.0.1:8000/api/wake_word/status
```

---

## Implementación Técnica

### Clase Principal: `VoskWakeWordEngine`

```python
class VoskWakeWordEngine:
    def __init__(
        self,
        wake_word_callback: Optional[Callable] = None,
        wake_word: str = "LILY",
        model_path: str = "models/vosk-model-small-es-0.42",
        sample_rate: int = 16000
    ):
        """
        Inicializa el motor de detección de wake word con Vosk
        
        Args:
            wake_word_callback: Función a llamar cuando se detecta la palabra clave
            wake_word: Palabra clave a detectar
            model_path: Ruta al modelo Vosk
            sample_rate: Frecuencia de muestreo
        """
```

### Métodos Principales

#### `start_listening()`

Inicia la escucha de la palabra clave.

```python
engine.start_listening()
```

#### `stop_listening()`

Detiene la escucha de la palabra clave.

```python
engine.stop_listening()
```

#### `set_callback(callback)`

Establece la función de callback para cuando se detecta la palabra clave.

```python
def my_callback():
    print("¡Palabra clave detectada!")

engine.set_callback(my_callback)
```

#### `is_available()`

Verifica si el motor está disponible (modelo cargado correctamente).

```python
if engine.is_available():
    print("Wake word engine está listo")
```

### Loop de Escucha

El método `_listen_loop()` implementa el bucle principal de escucha:

```python
def _listen_loop(self):
    """Bucle principal de escucha usando Vosk"""
    audio = pyaudio.PyAudio()
    stream = None
    
    try:
        # Abrir stream de audio
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Crear reconocedor
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(False)  # No necesitamos timestamps para wake word
        
        while self.is_listening:
            # Leer datos del micrófono
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Procesar con Vosk
            if rec.AcceptWaveform(data):
                # Resultado completo
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
                
                if self.wake_word in text:
                    if self.wake_word_callback:
                        self.wake_word_callback()
            else:
                # Resultado parcial
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "").lower()
                
                if self.wake_word in partial_text:
                    if self.wake_word_callback:
                        self.wake_word_callback()
    
    finally:
        # Limpiar recursos
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()
```

---

## Troubleshooting

### Problemas Comunes

#### ❌ "Modelo Vosk no encontrado"

**Síntomas**: Mensaje de error indicando que no se encuentra el modelo Vosk.

**Causa**: El modelo Vosk no está instalado o no está en la ruta correcta.

**Solución**:
1. Verificar que el modelo esté en `models/vosk-model-small-es-0.42/`
2. Asegurarse de que el nombre de la carpeta coincida exactamente
3. Descargar el modelo desde: https://alphacephei.com/vosk/models

#### ❌ "Micrófono no disponible"

**Síntomas**: El sistema no puede acceder al micrófono.

**Causa**: Problemas de permisos o el micrófono está siendo usado por otra aplicación.

**Solución**:
1. Verificar permisos de micrófono en Configuración de Windows
2. Cerrar otras aplicaciones que puedan estar usando el micrófono
3. Probar el micrófono en otra aplicación

#### ❌ "Fallo en la detección de wake word"

**Síntomas**: La palabra clave no se detecta aunque se pronuncie claramente.

**Causa**: Puede ser debido a ruido ambiental, volumen de voz bajo o modelo inadecuado.

**Solución**:
1. Probar en un entorno más silencioso
2. Hablar más claramente y a volumen adecuado
3. Considerar usar un modelo más grande para mayor precisión

#### ❌ "Consumo elevado de CPU"

**Síntomas**: El sistema consume muchos recursos mientras escucha.

**Causa**: El loop de escucha está corriendo continuamente.

**Solución**:
1. Asegurarse de detener la escucha cuando no sea necesaria
2. Considerar implementar períodos de escucha limitada
3. Verificar que no haya múltiples instancias corriendo

### Diagnóstico

Para diagnosticar problemas con el wake word, puedes usar el script de prueba incluido:

```bash
python -m models.vosk_wake_word_engine
```

Este script iniciará un proceso de prueba donde puedes decir "LILY" para verificar la detección.

---

## Consideraciones de Rendimiento

### Uso de Recursos

- **CPU**: 5-15% durante la escucha activa
- **RAM**: ~50-100 MB para el modelo Vosk
- **Latencia**: <100ms para detección típica

### Optimización

#### 1. Uso de Modelo Pequeño

Para wake word, el modelo pequeño (`vosk-model-small-es-0.42`) es suficiente y más eficiente.

#### 2. Control de Escucha

Implementar lógica para activar/desactivar la escucha según sea necesario:

```python
# Ejemplo de control de escucha
import time

def smart_listening():
    """Activar escucha solo en horarios o situaciones específicas"""
    while True:
        hour = time.localtime().tm_hour
        if 9 <= hour <= 22:  # Solo entre 9 AM y 10 PM
            if not engine.is_listening:
                engine.start_listening()
        else:
            if engine.is_listening:
                engine.stop_listening()
        
        time.sleep(60)  # Verificar cada minuto
```

#### 3. Gestión de Threads

El sistema utiliza threads separados para no bloquear la aplicación principal:

```python
# El sistema crea un thread separado para la escucha
self.listening_thread = threading.Thread(target=self._listen_loop)
self.listening_thread.daemon = True
self.listening_thread.start()
```

### Limitaciones

- **Requiere micrófono funcional**
- **Sensible a ruido ambiental**
- **Depende de la claridad de la pronunciación**
- **Un solo wake word activo a la vez**

---

## Integración con Lily AI

### En `ai_engine.py`

El sistema de wake word se integra con el motor principal de Lily:

```python
# En AIEngine.__init__()
if self.wake_word_enabled:
    try:
        self.wake_word_engine = VoskWakeWordEngine(
            wake_word_callback=self.on_wake_word_detected,
            wake_word="LILY"
        )
    except Exception as e:
        print(f"Error iniciando sistema de palabra clave: {e}")
```

### Callback de Activación

Cuando se detecta la wake word, se ejecuta el callback:

```python
def on_wake_word_detected(self):
    """Callback cuando se detecta la palabra clave"""
    print("¡Palabra clave 'LILY' detectada!")
    
    # Aquí puedes implementar la lógica deseada
    # Por ejemplo, iniciar grabación de audio o mostrar notificación
    try:
        response_text = "¡Hola! ¿En qué puedo ayudarte?"
        print(f"Lily responde: {response_text}")
    except Exception as e:
        print(f"Error en la respuesta de activación: {e}")
```

---

## Buenas Prácticas

### 1. Manejo de Errores

Siempre verificar disponibilidad del modelo antes de iniciar:

```python
if engine.is_available():
    engine.start_listening()
else:
    print("Modelo no disponible, no se puede iniciar wake word")
```

### 2. Limpieza de Recursos

Asegurarse de detener la escucha y liberar recursos:

```python
# En el destructor o al salir
def cleanup():
    if engine.is_listening:
        engine.stop_listening()
```

### 3. Configuración de Callbacks

Definir callbacks claros y eficientes:

```python
def wake_word_handler():
    """Handler eficiente para la detección de wake word"""
    # Lógica mínima aquí
    # Evitar operaciones pesadas en el thread de detección
    trigger_activation_event()  # Despachar evento a otro thread si es necesario
```

---

## Seguridad y Privacidad

### Procesamiento Local

- Todo el procesamiento de voz ocurre localmente
- No se envían datos de audio a servidores externos
- La palabra clave se procesa internamente sin almacenamiento

### Acceso al Micrófono

- El sistema solo accede al micrófono cuando está activo
- Se puede desactivar la escucha en cualquier momento
- No se almacenan grabaciones de audio

---

## Recursos Adicionales

- **Documentación Vosk**: https://alphacephei.com/vosk/
- **Modelos Vosk**: https://alphacephei.com/vosk/models
- **Repositorio Lily AI**: https://github.com/Mijin-VT/LILY-VIRTUAL-3.0

---

*Documentación actualizada para Lily AI Virtual 3.0*
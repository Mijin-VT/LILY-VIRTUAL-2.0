# 📚 Documentación de LILY AI Virtual 3.0

**Bienvenido a la documentación oficial de Lily AI Virtual Companion**

---

## 📁 Archivos de Documentación

Esta carpeta contiene toda la documentación técnica y de usuario del proyecto:

| Archivo | Descripción |
|---------|-------------|
| [`CARACTERISTICAS_IMPLEMENTADAS.md`](CARACTERISTICAS_IMPLEMENTADAS.md) | Lista completa de características y su estado de implementación |
| [`CONTROL_MEDIA.md`](CONTROL_MEDIA.md) | Guía de uso para el control de YouTube y medios |
| [`GUIA_INSTALADOR.md`](GUIA_INSTALADOR.md) | Instrucciones para usar el instalador automático |
| [`SOLUCION_BAT_SE_CIERRA.md`](SOLUCION_BAT_SE_CIERRA.md) | Solución de problemas con archivos .bat |
| [`README.md`](README.md) | Este archivo - Índice de documentación |

---

## 🚀 Primeros Pasos

### Si eres usuario nuevo:

1. **Lee el README principal**: [`../README.md`](../README.md)
2. **Usa el instalador**: Ejecuta `Lily_Setup.bat`
3. **Inicia Lily**: Ejecuta `INICIAR_LILY.bat`
4. **Consulta problemas**: [`SOLUCION_BAT_SE_CIERRA.md`](SOLUCION_BAT_SE_CIERRA.md)

### Si quieres usar el control de medios:

1. **Lee la guía**: [`CONTROL_MEDIA.md`](CONTROL_MEDIA.md)
2. **Prueba los comandos**: "Pon música de...", "pausa", "siguiente"

### Si eres desarrollador:

1. **Revisa características**: [`CARACTERISTICAS_IMPLEMENTADAS.md`](CARACTERISTICAS_IMPLEMENTADAS.md)
2. **Estudia la arquitectura**: Ver sección "Arquitectura del Sistema"
3. **API Endpoints**: Ver [`../README.md`](../README.md) sección API

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                   LILY AI VIRTUAL 3.0                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Interfaz Web   │◄────►│   FastAPI        │◄────►│ Ollama       │
│  (Edge/Chrome)  │      │   Backend        │      │ (Qwen3 0.6B) │
└─────────────────┘      └──────────────────┘      └──────────────┘
        │                         │
        │                         ├──► Emotional Intelligence
        │                         ├──► Memory System
        │                         ├──► TTS Engine (Kokoro-ONNX)
        │                         ├──► Wake Word (Vosk)
        │                         ├──► YouTube Controller
        │                         ├──► Media Controller
        │                         └──► AI Engine
        │
        ▼
┌─────────────────┐
│  HTML/CSS/JS    │
│  - Avatar       │
│  - Chat         │
│  - Emociones    │
└─────────────────┘
```

---

## 📊 Estado del Proyecto

| Componente | Estado | Versión |
|------------|--------|---------|
| Core IA (Qwen3 0.6B) | ✅ Completo | 3.0 |
| Inteligencia Emocional | ✅ Completo | 3.0 |
| Reconocimiento de Voz (Vosk) | ✅ Completo | 3.0 |
| Control de YouTube/Medios | ✅ Completo | 3.0 |
| Sistema de Memoria | ✅ Completo | 3.0 |
| Interfaz Web | ✅ Completo | 3.0 |
| Texto a Voz | ✅ Funcional | 3.0 |

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI 0.115.6** - Framework web
- **Uvicorn 0.34.0** - Servidor ASGI
- **Pydantic 2.10.5** - Validación de datos

### Inteligencia Artificial
- **Qwen3 0.6B** - Modelo de lenguaje vía Ollama
- **TextBlob 0.19.0** - Análisis de sentimientos

### Reconocimiento de Voz
- **Vosk 0.3.45** - Reconocimiento offline
- **PyAudio 0.2.14** - Captura de audio

### Control de Medios
- **PyAutoGUI 0.9.54** - Automatización de teclado

### Texto a Voz
- **Kokoro-ONNX** - Síntesis de voz neuronal
- **torch** - Motor de inferencia
- **soundfile** - Procesamiento de audio

### Memoria
- **ChromaDB 0.4.22** - Base de datos vectorial

---

## 📝 Notas de Versión

### Versión 3.0 (Febrero 2025)
- ✅ Reconocimiento de voz 100% offline con Vosk
- ✅ Wake word "LILY" completamente local
- ✅ Control de YouTube y medios por voz
- ✅ Integración con Qwen3 0.6B vía Ollama
- ✅ Sistema de memoria persistente
- ✅ Inteligencia emocional avanzada
- ✅ Interfaz web con tema anime

---

## 🔗 Enlaces Útiles

- **Python**: https://www.python.org/downloads/
- **Ollama**: https://ollama.ai/
- **Vosk**: https://alphacephei.com/vosk/

---

## 🤝 Contribuir

¿Quieres contribuir al proyecto?

1. Fork del repositorio
2. Crea una rama para tu feature
3. Commit de tus cambios
4. Push a la rama
5. Abre un Pull Request

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT.

Ver archivo [`../LICENSE`](../LICENSE) para más detalles.

---

## 💕 Agradecimientos

- **LilyBell** - Inspiración para el proyecto
- **Qwen3** - Modelo de lenguaje
- **Ollama** - Ejecución local de modelos
- **Vosk** - Reconocimiento de voz offline

---

**Última actualización**: Julio 2026

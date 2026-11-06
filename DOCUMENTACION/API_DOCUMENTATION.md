# 🌸 Lily AI - API Documentation

**Versión**: 2.1 | **Lily AI Virtual Companion**

La API de Lily AI se ejecuta localmente y proporciona endpoints para interactuar con la compañera virtual, gestionar la base de conocimiento local (RAG), configurar preferencias de usuario y sintetizar audio.

---

## Base URL

```text
http://127.0.0.1:8000
```

---

## 1. Chat y Conversación

### POST `/api/chat`
Envía un mensaje y recibe una respuesta de Lily con análisis emocional y audio TTS.

**Request Body**:
```json
{
  "message": "Hola Lily, ¿cómo estás?",
  "user_id": "default_user"
}
```

**Response (200 OK)**:
```json
{
  "response": "¡Hola Mijin! Estoy muy contenta de hablar contigo hoy.",
  "emotion": "feliz",
  "audio_url": "/static/audio/lily_17198754_3ab98.mp3",
  "timestamp": "2026-07-02T16:00:00"
}
```

---

## 2. Configuración y Preferencias de Usuario

### GET `/api/preferences/{user_id}`
Obtiene el nombre de usuario configurado y el estado de las integraciones con Gmail y Gemini.

**Response (200 OK)**:
```json
{
  "gmail_user": "correo_usuario@gmail.com",
  "has_password": true,
  "user_name": "Mijin",
  "has_gemini_key": true,
  "gemini_key_preview": "AIzaSy...a1b2"
}
```

### POST `/api/preferences/{user_id}`
Guarda o actualiza las preferencias del usuario (nombre, credenciales de Gmail y clave API de Gemini).

**Request Body**:
```json
{
  "user_name": "Kinetic",
  "gmail_user": "correo_usuario@gmail.com",
  "gmail_password": "tu_clave_de_aplicacion",
  "gemini_api_key": "AIzaSy..."
}
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Preferencias actualizadas"
}
```

> **Proveedor de LLM**: si el usuario tiene `gemini_api_key` configurada, Lily usa los modelos de Google Gemini (por defecto `gemini-3.5-flash-lite`, con respaldo automático a `gemini-3.6-flash` y `gemini-3.8-flash`; configurable con la preferencia `gemini_model` o la variable de entorno `LILY_GEMINI_MODEL`). Si un modelo no está disponible o devuelve `504 DEADLINE_EXCEEDED`, se prueba el siguiente automáticamente. Si Gemini falla o no hay clave, se usa Ollama automáticamente como respaldo.

> **Formatos de clave**: Google AI Studio ahora emite claves de autorización (*auth keys*) que empiezan por `AQ.`. Las claves estándar anteriores empiezan por `AIza`. Lily acepta ambos formatos y envía la clave a Google mediante la cabecera `x-goog-api-key`.

### POST `/api/gemini/test/{user_id}`
Prueba la conexión con Gemini usando la clave API guardada. Se ejecuta en un hilo aparte para no bloquear el servidor.

**Response (200 OK)**:
```json
{
  "status": "success",
  "model": "gemini-3.5-flash-lite",
  "response": "OK"
}
```

**Response (400 Bad Request)**: si no hay clave configurada o la clave no es válida.
```json
{
  "detail": "Error conectando con Gemini: API key not valid..."
}
```

---

## 3. Base de Conocimiento (RAG Local)

### GET `/api/rag/stats`
Devuelve el número total de documentos, fragmentos de texto (chunks) indexados en ChromaDB y el umbral de similitud semántica.

**Response (200 OK)**:
```json
{
  "total_documents": 5,
  "total_chunks": 120,
  "similarity_threshold": 0.45
}
```

### GET `/api/rag/documents`
Lista todos los documentos indexados en la base de datos de conocimiento local.

**Response (200 OK)**:
```json
{
  "documents": [
    {
      "id": "doc_12345",
      "metadata": {
        "source": "knowledge/manual_usuario.pdf",
        "chunks": 14
      }
    }
  ]
}
```

### POST `/api/rag/upload-document`
Sube y procesa un archivo (`.pdf`, `.txt`, `.md`, `.docx`) para indexarlo de forma inmediata en ChromaDB.

**Request Form-Data**:
- `file`: (archivo binario)

**Response (200 OK)**:
```json
{
  "status": "success",
  "filename": "manual_usuario.pdf",
  "chunks_created": 14
}
```

### DELETE `/api/rag/document/{doc_id}`
Elimina un documento indexado y todos sus fragmentos de ChromaDB.

**Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Documento eliminado correctamente"
}
```

### POST `/api/rag/ingest-knowledge`
Escanea la carpeta local `knowledge/` para indexar cualquier archivo nuevo que se haya colocado manualmente allí.

**Response (200 OK)**:
```json
{
  "status": "success",
  "files_ingested": 2
}
```

---

## 4. Audio y Transcripciones

### POST `/api/tts`
Sintetiza texto a voz modulando los parámetros de velocidad y pitch según la emoción.

**Request Body**:
```json
{
  "text": "Hola Mijin",
  "emotion": "feliz"
}
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "audio_url": "/static/audio/lily_17198754_3ab98.mp3",
  "text": "Hola Mijin",
  "emotion": "feliz"
}
```

### POST `/api/transcribe`
Transcribe una grabación de voz (`.wav` / `.mp3`) enviada por el cliente utilizando Whisper Online.

**Request Form-Data**:
- `file`: (archivo de audio)

**Response (200 OK)**:
```json
{
  "text": "hola lily como estas hoy",
  "engine": "whisper"
}
```
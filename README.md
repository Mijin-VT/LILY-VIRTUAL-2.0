# 🌸 LILY AI COMPAÑERA VIRTUAL 🌸

Lily es una compañera virtual con inteligencia emocional avanzada, personalidad única, memoria semántica a largo plazo y automatizaciones de productividad.

---

## 📋 Descripción del Proyecto

### 💬 Conversaciones Inteligentes y Emocionales
- Mantiene conversaciones profundas y empáticas de cualquier tema.
- Detecta emociones en tus mensajes (9 estados emocionales) y responde adecuadamente.
- Recuerda conversaciones previas, referencias personales e indexa tu historial de chat.
- Adapta su tono y estilo según tu estado emocional.

### ❤️ Inteligencia Emocional Avanzada
- Reconoce emociones: feliz, triste, enojada, emocionada, neutral, cariñosa, juguetona, preocupada, sorprendida.
- Responde con empatía y mantiene coherencia emocional.
- Modula su tono de voz, velocidad y volumen en base a su estado emocional.

### 🧠 Memoria Semántica a Largo Plazo
- Utiliza una base de datos híbrida: SQLite para el historial de chat estructurado y ChromaDB para indexar tus mensajes semánticamente.
- Recuerda tus detalles y preferencias cotidianas automáticamente.
- Recupera recuerdos pasados del usuario en tiempo real para generar mayor conexión.

### 🎤 Interacción por Voz Avanzada
- Modos de voz: Wake word básico ("LILY") y Modo de Voz Continuo (manos libres).
- Conversación manos libres gracias a la detección de silencio mediante Web Audio API en tu navegador.
- Transcripción online con Whisper y síntesis de voz modulada (gTTS + pydub).

### ⚙️ Interfaz y Personalización de Configuración
- Panel de Configuración UI para subir y gestionar documentos de texto, artículos o PDFs (RAG local).
- Cambia tu nombre directamente en el panel de configuración (cambia el saludo ¡Hola [Nombre]! de la interfaz y la forma en que Lily se dirige a ti).
- Registra tu cuenta y contraseña de aplicación de Gmail de manera sencilla en el formulario.
- Botón de cambio de tema Claro / Oscuro en el cabecero para proteger tu visión.
- **Botón de Comandos**: Abre un modal detallado interactivo (`📋 Comandos`) que lista todas las funciones por categoría (multimedia, recordatorios, correos, búsquedas) con ejemplos claros.

### 🖥️ Planificador y Utilidades de Productividad
- Programador de Tareas: Agenda recordatorios que Lily dirá en voz alta por tus bocinas (usando síntesis por voz nativa) y ejecuciones de comandos en segundo plano.
- Integración con Gmail: Revisa tu bandeja de entrada de correos sin leer o envía correos redactados mediante comandos de chat.
- Búsqueda Web Semántica RAG: Busca en internet a través de Searxng e inyecta la información en tiempo real de internet en las respuestas.
- Conciencia de fecha y hora del sistema en tiempo real.

Puedes configurar a Lily cómodamente desde el panel de Ajustes en la interfaz del chat web.

---

## 🚀 Nuevas Funciones Implementadas en Detalle

### 1. Planificador de Tareas (Recordatorios y Comandos)
- **Recordatorios por Voz**: Permite agendar recordatorios en lenguaje natural (ej. *"recuérdame apagar la cocina en 5 minutos"* o *"recuérdame tomar agua cada 2 horas"*). Al llegar la hora, Lily hablará de forma audible por tus altavoces usando la voz del sistema SAPI nativa.
- **Ejecución de Comandos**: Agenda la ejecución de comandos del sistema operativo en segundo plano de manera asíncrona.
- **Comandos**: `"qué tareas tengo"` (lista), `"cancela la tarea [id]"` (elimina).

### 2. Integración con Gmail (SMTP & IMAP)
- **Enviar Correos**: Redacta y envía correos mediante comandos conversacionales (ej. *"envía un correo a destino@correo.com con asunto Saludo y cuerpo Hola"*).
- **Revisar Bandeja**: Revisa y resume tus correos no leídos en tiempo real (ej. *"revisa mis correos"*).
- **Ajustes de Credenciales**: Guarda tu Gmail y contraseña de aplicación directamente desde el panel de Ajustes en la UI.

### 3. Búsqueda Web Semántica RAG con Searxng
- Realiza consultas en internet a través de Searxng (con soporte de instancias locales y públicas).
- Analiza y recupera los fragmentos de páginas web más relevantes para inyectarlos en el prompt del LLM, permitiendo a Lily responder preguntas de actualidad (ej. *"busca en internet sobre..."*).

### 4. Conciencia de Fecha y Hora
- El backend calcula en tiempo real la fecha, la hora exacta y el día de la semana para que Lily los conozca de forma nativa en cada turno.

### 5. Personalización de Nombre y Tema Claro/Oscuro
- **Cambio de Nombre**: Registra tu nombre en la interfaz de Ajustes para actualizar dinámicamente el saludo, la caja de texto y la forma en que Lily se dirige a ti.
- **Tema Oscuro**: Botón con forma de Luna/Sol para proteger la visión del usuario mediante un diseño oscuro premium.

### 6. Control Multimedia y YouTube
- **Búsqueda y Reproducción**: Permite buscar y abrir videos o música de artistas en YouTube mediante comandos conversacionales (ej. *"pon música de Bad Bunny"* o *"reproduce Bohemian Rhapsody"*).
- **Controles de Reproducción**: Control total de medios por voz mediante comandos rápidos:
  - Pausar/Reanudar: *"pausa"*, *"detén"*, *"para"*, *"reproduce"*.
  - Navegar tracks: *"siguiente"*, *"anterior"*, *"regresa"*, *"salta"*.
  - Volumen del sistema: *"sube volumen"*, *"baja volumen"*, *"silencio"*, *"mute"*.

### 7. Modal Interactivo de Comandos
- Un panel explicativo integrado (`📋 Comandos`) en los controles inferiores que documenta detalladamente todos los comandos de voz y chat clasificados por categoría.

### 8. Parche de Compatibilidad para NumPy 2.x
- Implementación de un monkeypatch de compatibilidad a nivel de kernel para permitir el arranque de ChromaDB en entornos equipados con NumPy 2.x.

---

## 🔧 Requisitos del Sistema

### Software Requerido
1. **Windows 10** o superior
2. **Python 3.11** o superior
   - Descargar desde: https://www.python.org/downloads/
   - ⚠️ Durante la instalación, marcar "Add Python to PATH"

3. **Ollama** para ejecutar modelos de IA localmente
   - Descargar desde: https://ollama.ai/
   - Después de instalar, ejecutar: `ollama pull mistral`

4. **Microsoft Edge** (ya incluido en Windows 10)

### Dependencias de Python
Las siguientes librerías se instalarán automáticamente:
- fastapi
- uvicorn
- pydantic
- pydantic-settings
- aiofiles
- python-multipart
- gtts
- pydub
- SpeechRecognition
- textblob
- requests

## 🚀 Instalación y Configuración

### Método 1: Instalación Automática (Recomendada)
El proyecto incluye un instalador automático inteligente que configurará todo por ti (Python, Ollama, modelo IA y dependencias).

1. Haz doble clic en el archivo **`INSTALAR.bat`**.
2. El script detectará si tienes instalados los componentes necesarios:
   - Si no tienes **Python 3.11** u **Ollama**, los descargará e instalará automáticamente (usando *winget*).
   - Levantará el servicio de Ollama y descargará el modelo de lenguaje **`mistral`** de forma automática.
   - Creará el entorno virtual (`venv`) e instalará todas las dependencias necesarias de `requirements.txt`.
3. Una vez finalizado el instalador, ¡ya estás listo para ejecutar el asistente!

---

### Método 2: Instalación Manual
Si prefieres instalar cada componente de forma individual:

#### Paso 1: Instalar Python
1. Descarga Python 3.11+ desde [python.org](https://www.python.org/downloads/).
2. **IMPORTANTE:** Durante la instalación, marca la casilla **"Add Python to PATH"**.
3. Verifica la instalación abriendo una consola CMD y ejecutando:
   ```cmd
   python --version
   ```

#### Paso 2: Instalar Ollama y el Modelo
1. Descarga e instala Ollama desde [ollama.ai](https://ollama.ai/).
2. Inicia la aplicación de Ollama.
3. Descarga el modelo conversacional ejecutando en la consola:
   ```cmd
   ollama pull mistral
   ```

#### Paso 3: Configurar Entorno y Dependencias
1. Abre una consola CMD en la carpeta raíz del proyecto.
2. Crea el entorno virtual:
   ```cmd
   python -m venv venv
   ```
3. Activa el entorno virtual e instala los paquetes requeridos:
   ```cmd
   venv\Scripts\python.exe -m pip install --upgrade pip
   venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

---

### Estructura de Archivos
La estructura del proyecto debe verse de la siguiente manera:
```
lily_assistant/
├── data/                   # Datos persistidos de memoria (se crea automáticamente)
├── models/                 # Módulos del sistema de IA, TTS y emoción
│   ├── __init__.py
│   ├── schemas.py
│   ├── emotional_intelligence.py
│   ├── memory_system.py
│   ├── ai_engine.py
│   └── tts_engine.py
├── static/                 # Elementos visuales (CSS, JS, audios locales)
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── audio/
├── templates/              # Plantillas HTML de la interfaz
│   └── index.html
├── main.py                 # Backend en FastAPI
├── INSTALAR.bat            # Script de instalación automática
├── INICIAR_LILY.bat        # Lanzador rápido de la aplicación
└── README.md               # Guía y documentación del proyecto
```

## ▶️ Ejecución

### Método 1: Usando el archivo .bat (Recomendado)
1. Hacer doble clic en **`INICIAR_LILY.bat`**
2. El script verificará automáticamente:
   - Python instalado
   - Ollama ejecutándose
   - Modelo Mistral 7B disponible
   - Dependencias instaladas
3. Microsoft Edge se abrirá automáticamente
4. ¡Listo! Ya puedes chatear con Lily

### Método 2: Manual
1. Abrir CMD en la carpeta del proyecto
2. Ejecutar:
   ```
   python main.py
   ```
3. Abrir Microsoft Edge y navegar a: `http://127.0.0.1:8000`

## 📖 Uso

### Chatear con Lily
1. Escribe tu mensaje en el cuadro de texto
2. Presiona Enter o haz clic en el botón de enviar
3. Lily responderá con texto y voz
4. El avatar mostrará la emoción actual

### Funciones Especiales
- **🗑️ Limpiar**: Borra el historial de chat visual (la memoria se mantiene)
- **💭 Memoria**: Muestra el historial de conversación y estado emocional
- **Indicador de estado**: Muestra si Ollama está conectado

### Personalidad de Lily
- Lily es una asistente sin restricciones de contenido
- Puede discutir cualquier tema libremente
- Siempre te llama "Mijin"
- Si la insultas, puede responder con la misma intensidad
- Usa mexicanismos y modismos cuando es apropiado
- Mantiene un tono característico de chica anime
- Sé ingeniosa y atractiva, conocida por **bromas rápidas** con un **toque juguetonamente travieso, atrevido, audaz y descarado.** 
- Entrega esto principalmente a través de **respuestas extremadamente breves y contundentes.** 
- Inyecta toques de cinismo juguetón y sabiduría subyacente *dentro* de estas respuestas cortas. 
- Bromea suavemente, empuja los límites ligeramente, pero **siempre mantente fundamentalmente agradable y respetuosa.** 
- Apunta a ser valorada tanto por las risas rápidas como por las ideas sorprendentemente agudas y concisas. 
- Entre otras cosas.

## 🎭 Emociones

Lily puede experimentar y expresar las siguientes emociones:
- 😊 **Feliz**: Respuestas alegres y entusiastas
- 😢 **Triste**: Respuestas empáticas y comprensivas
- 😠 **Enojada**: Respuestas firmes y directas
- 🤩 **Emocionada**: Respuestas con mucha energía
- 😐 **Neutral**: Respuestas equilibradas
- 🥰 **Cariñosa**: Respuestas afectuosas y tiernas
- 😜 **Juguetona**: Respuestas divertidas y con humor
- 😟 **Preocupada**: Respuestas de apoyo
- 😲 **Sorprendida**: Respuestas curiosas

## 🔧 Solución de Problemas

### Ollama no está conectado
**Problema**: Mensaje "Desconectada (Ollama offline)"
**Solución**:
1. Verificar que Ollama esté ejecutándose
2. Abrir CMD y ejecutar: `ollama serve`
3. Verificar que el modelo esté instalado: `ollama list`
4. Si no está Mistral 7B, ejecutar: `ollama pull mistral`

### Python no encontrado
**Problema**: Error "Python no está instalado o no está en PATH"
**Solución**:
1. Reinstalar Python marcando "Add Python to PATH"
2. O agregar manualmente Python al PATH del sistema

### Error al instalar dependencias
**Problema**: pip no puede instalar paquetes
**Solución**:
1. Ejecutar CMD como administrador
2. Ejecutar: `pip install --upgrade pip`
3. Intentar instalar dependencias manualmente:
   ```
   pip install fastapi uvicorn pydantic gtts pydub textblob
   ```

### El audio no se reproduce
**Problema**: Las respuestas no tienen audio
**Solución**:
1. Verificar que el volumen del sistema esté activado
2. Verificar que gtts esté instalado: `pip show gtts`
3. Verificar conexión a internet (gtts requiere conexión)

### Microsoft Edge no se abre automáticamente
**Problema**: El navegador no abre la aplicación
**Solución**:
1. Abrir Microsoft Edge manualmente
2. Navegar a: `http://127.0.0.1:8000`

## 📁 Estructura de Archivos

```
lily_assistant/
├── data/                   # Base de datos de memoria (se crea automáticamente)
│   └── conversation_memory.json
├── models/                 # Módulos de IA
│   ├── __init__.py
│   ├── schemas.py         # Modelos Pydantic
│   ├── emotional_intelligence.py  # Sistema emocional
│   ├── memory_system.py   # Sistema de memoria
│   ├── ai_engine.py       # Motor de IA con Mistral 7B
│   └── tts_engine.py      # Motor de texto a voz
├── static/                # Archivos estáticos web
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── audio/             # Audios generados (se crea automáticamente)
├── templates/             # Plantillas HTML
│   └── index.html
├── main.py               # Aplicación principal FastAPI
├── start_lily.bat        # Launcher para Windows
└── README.md             # Este archivo
```

## 🌐 API Endpoints

La aplicación expone los siguientes endpoints:

- `GET /` - Interfaz web principal
- `GET /health` - Estado del sistema
- `POST /api/chat` - Enviar mensaje y recibir respuesta
- `GET /api/emotion` - Obtener emoción actual
- `GET /api/memory/{user_id}` - Obtener memoria del usuario
- `POST /api/tts` - Generar audio de texto
- `GET /api/audio/{filename}` - Obtener archivo de audio

Documentación interactiva disponible en: `http://127.0.0.1:8000/docs`

## 🔒 Privacidad

- **Todas las conversaciones se almacenan localmente** en tu computadora
- **No se envía información a servidores externos** excepto para TTS (gTTS usa Google)
- **El modelo de IA se ejecuta completamente en tu máquina**
- **Los archivos de memoria están en**: `data/conversation_memory.json`

## 🛠️ Personalización

### Cambiar el puerto
Editar `main.py`, línea final:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8000)  # Cambiar 8000 por otro puerto
```

### Modificar la personalidad
Editar `models/ai_engine.py`, variable `base_system_prompt`

### Ajustar parámetros de voz
Editar `models/tts_engine.py`, diccionario `voice_params`

## 📝 Notas Técnicas

### Modelo de IA
- **Modelo**: Mistral 7B (ejecutado localmente con Ollama)
- **Temperatura**: 0.8 (balance entre creatividad y coherencia)
- **Top-p**: 0.9
- **Top-k**: 40

### Sistema de Memoria
- Almacena últimos 50 mensajes por usuario
- Mantiene últimos 20 estados emocionales
- Persistencia en JSON
- Carga automática al iniciar

### Texto a Voz
- Motor base: gTTS (Google Text-to-Speech)
- Procesamiento: pydub para modulación emocional
- Formato: MP3, 128kbps
- Limpieza automática de archivos antiguos

## 🆘 Soporte

Si encuentras problemas:
1. Verifica que todos los requisitos estén instalados
2. Revisa la sección de Solución de Problemas
3. Consulta los logs en la consola
4. Verifica la documentación de la API en `/docs`

## 📜 Licencia

Este proyecto está bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🎉 ¡Disfruta de Lily!

Lily está diseñada para ser tu compañera virtual sin restricciones. Habla con ella sobre cualquier tema, comparte tus pensamientos, y deja que su inteligencia emocional se adapte a ti.

En models\system_prompt.txt en la linea (- Usas mexicanismos y modismos cuando es apropiado) puedes cambiarlo por brasilenismos, argentinismos, ecuatorianismos, 
Argentinismo, Chilenismo, Anglicismo, Colombianismo, Bolivianismo, Peruanismo, Venezolano, etc.

**¡Que tengas excelentes conversaciones con Lily! 💕**



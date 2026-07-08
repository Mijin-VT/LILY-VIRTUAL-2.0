# 🎤 Manual de Usuario: Comandos de Voz y Controles Multimedia

**Versión**: 2.1 | **Lily AI Virtual Companion**

---

## 📖 Índice

1. [Introducción](#introducción)
2. [Modos de Activación de Voz](#modos-de-activación-de-voz)
3. [Comandos de Música y YouTube](#comandos-de-música-y-youtube)
4. [Controles de Reproducción y Volumen](#controles-de-reproducción-y-volumen)
5. [Programador de Tareas (Planificador)](#programador-de-tareas-planificador)
6. [Integración con Gmail](#integración-con-gmail)
7. [Búsqueda Web Semántica RAG](#búsqueda-web-semántica-rag)
8. [Personalización de Nombre y Tema](#personalización-de-nombre-y-tema)

---

## Introducción

Este manual te guiará sobre cómo interactuar con Lily AI Virtual Companion usando comandos de voz y chat. Lily puede entender y responder a comandos conversacionales estructurados y de lenguaje natural para reproducir música, controlar medios, agendar recordatorios de voz, ejecutar tareas del sistema, enviar y leer correos electrónicos, y buscar en la web.

---

## Modos de Activación de Voz

### 1. Modo de Conversación Continua (Manos Libres) 🎤
- **Activación**: Presiona el icono del micrófono en la interfaz web.
- **Funcionamiento**: Lily detecta cuando comienzas a hablar y cuando guardas silencio (usando un analizador de audio local en tu navegador). Al guardar silencio (1.8s de inactividad), detiene la captura de voz, transcribe tu mensaje, te responde con su voz neuronal, y **se reactiva automáticamente** para seguir escuchando tu próximo turno.
- **Desactivación**: Presiona el icono del micrófono por segunda vez.

### 2. Wake Word "LILY" 🌸
- **Activación**: Di la palabra clave **"LILY"** en voz alta frente a tu micrófono.
- El navegador web iniciará la escucha de tu consulta y Lily responderá de forma conversacional.

---

## Comandos de Música y YouTube

Lily detecta intenciones musicales y abre automáticamente una pestaña del navegador con resultados de YouTube:

| Comando | Acción | Ejemplo |
|---------|--------|---------|
| `"Pon música de [artista]"` | Busca y reproduce canciones del artista | `"Pon música de Taylor Swift"` |
| `"Reproduce [canción]"` | Busca y reproduce la canción especificada | `"Reproduce Lamento Boliviano"` |
| `"Play [artista/canción]"` | Abre la canción en YouTube | `"Play Linkin Park"` |
| `"Escuchar [artista/canción]"` | Alternativa conversacional | `"Escuchar Dua Lipa"` |

---

## Controles de Reproducción y Volumen

Estos comandos controlan la reproducción multimedia en tu computadora usando atajos del sistema:

| Comando | Acción |
|---------|--------|
| `"pausa"` / `"pause"` / `"para"` / `"detén"` | Pausa o reanuda la música |
| `"siguiente"` / `"next"` / `"skip"` / `"salta"` | Avanza al siguiente video de la lista |
| `"anterior"` / `"previous"` / `"atrás"` / `"regresa"` | Vuelve al video anterior |
| `"sube volumen"` / `"más volumen"` | Incrementa el volumen del sistema |
| `"baja volumen"` / `"menos volumen"` | Reduce el volumen del sistema |
| `"silencio"` / `"mute"` / `"calla"` | Activa o desactiva el silencio |

---

## Programador de Tareas (Planificador)

Lily cuenta con un planificador de tareas persistente en SQLite:

### 1. Recordatorios por Voz ⏰
El planificador ejecutará tu recordatorio a la hora configurada hablándote en voz alta por tus altavoces usando la voz del sistema:
- *Relativos*: `"recuérdame tomar agua en 30 minutos"`
- *Específicos*: `"recuérdame a las 5:30 pm apagar la estufa"`
- *Recurrentes*: `"recuérdame estirarme cada 1 hora"`

### 2. Ejecución de Comandos 🖥️
Ejecuta comandos del sistema operativo (consola cmd/PowerShell) de manera automatizada:
- `"ejecuta el comando start notepad a las 10:00 am"`
- `"corre el comando python test.py en 5 minutos"`

### 3. Gestión de Tareas
- **Listar**: `"qué tareas tengo"` / `"lista de recordatorios"` / `"recordatorios activos"`
- **Cancelar**: `"cancela la tarea [id]"` / `"borra el recordatorio [id]"`

---

## Integración con Gmail

Configura tus credenciales de Gmail de forma segura en el modal de Ajustes para utilizar estos comandos:

- **Revisar Correos**: `"revisa mi bandeja de entrada"` / `"tengo correos nuevos"` / `"revisa mis correos"`. Lily buscará tus correos sin leer, te los resumirá en el chat y te los leerá en voz alta.
- **Enviar Correos**: `"envía un correo a destino@correo.com con asunto Saludo y cuerpo hola como estas"`. Lily enviará el correo por ti.

---

## Búsqueda Web Semántica RAG

Cuando necesitas información del presente o de internet, pídeselo a Lily con estas frases:

- `"busca en internet sobre el partido de fútbol de hoy"`
- `"busca en la web sobre las últimas noticias de inteligencia artificial"`
- `"investiga en internet sobre la receta del ceviche peruano"`

Lily realizará una consulta en tiempo real mediante **Searxng** y utilizará los pasajes web devueltos para responder a tu pregunta con hechos recientes y fuentes de consulta.

---

## Personalización de Nombre y Tema

Puedes cambiar cómo te trata Lily de forma interactiva en la interfaz:
1. Haz clic en el botón **Settings (⚙️)** en el panel inferior.
2. En la sección **Personalización de Nombre**, escribe tu nombre y haz clic en **Guardar Nombre**.
   - Esto actualizará el saludo de la interfaz (`¡Hola [Nombre]! 💞`), el cajón de texto, y Lily te llamará por este nombre en sus respuestas.
3. Haz clic en el **icono de la Luna / Sol** en el cabecero de la página para cambiar entre el tema claro (lindo y vibrante) y el modo oscuro (relajante para la vista).
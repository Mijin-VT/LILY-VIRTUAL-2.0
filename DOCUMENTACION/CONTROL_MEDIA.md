# 🎵 Control de YouTube y Medios - Guía Completa

**Versión**: 3.0 | **Archivos**: `youtube_controller.py`, `media_controller.py`

---

## 🎯 Funcionalidades Implementadas

Lily puede controlar YouTube y los medios de tu sistema usando comandos de voz naturales en español.

---

## 🎤 Comandos de Voz Disponibles

### 🎵 Reproducir Música en YouTube

**Comandos válidos**:
- `"Pon música de [artista/canción]"`
- `"Reproduce [artista/canción]"`
- `"Play [artista/canción]"`
- `"Ponme música de [artista/canción]"`
- `"Escuchar [artista/canción]"`
- `"Quiero escuchar [artista/canción]"`

**Ejemplos de uso**:
```
Usuario: "LILY, pon música de Juan Gabriel"
Lily: "¡Claro Carlos! Buscando Juan Gabriel en YouTube 🎵"
→ Se abre Microsoft Edge con la búsqueda en YouTube Music

Usuario: "Reproduce Bohemian Rhapsody"
Lily: "¡Claro Carlos! Buscando Bohemian Rhapsody en YouTube 🎵"
→ Abre YouTube con la canción

Usuario: "Play Shakira"
Lily: "¡Claro Carlos! Buscando Shakira en YouTube 🎵"
```

---

### ⏯️ Control de Reproducción

#### Pausar / Reanudar
**Comandos**: `pausa`, `pause`, `detén`, `para`, `reproduce`, `play`

```
Usuario: "pausa"
Lily: "¡Listo Carlos! Pausado/Reproduciendo 🎵"
→ Presiona la tecla ESPACIO
```

#### Siguiente Video
**Comandos**: `siguiente`, `next`, `skip`, `salta`, `próximo`

```
Usuario: "siguiente"
Lily: "¡Siguiente video Carlos! ⏭️"
→ Presiona Shift+N (atajo de YouTube)
```

#### Video Anterior
**Comandos**: `anterior`, `previous`, `atrás`, `regresa`, `previo`

```
Usuario: "anterior"
Lily: "¡Video anterior Carlos! ⏮️"
→ Presiona Shift+P (atajo de YouTube)
```

---

### 🔊 Control de Volumen del Sistema

#### Subir Volumen
**Comandos**:
- `"Sube volumen"`
- `"Más volumen"`
- `"Volumen arriba"`
- `"Sube el volumen"`
- `"Aumenta volumen"`

```
Usuario: "sube volumen"
Lily: "¡Volumen subido Carlos! 🔊"
→ Aumenta 3 pasos el volumen del sistema
```

#### Bajar Volumen
**Comandos**:
- `"Baja volumen"`
- `"Menos volumen"`
- `"Volumen abajo"`
- `"Baja el volumen"`
- `"Disminuye volumen"`

```
Usuario: "baja volumen"
Lily: "¡Volumen bajado Carlos! 🔉"
→ Reduce 3 pasos el volumen del sistema
```

#### Silenciar / Desilenciar
**Comandos**: `silencio`, `mute`, `calla`, `desilenciar`, `unmute`

```
Usuario: "silencio"
Lily: "¡Silencio activado/desactivado Carlos! 🔇"
→ Alterna entre silencio y volumen normal
```

---

## 🏗️ Cómo Funciona

### Flujo Completo - Reproducción de Música

```
1. Usuario: "LILY" (wake word)
   ↓
2. Lily activa reconocimiento de voz Vosk
   ↓
3. Usuario: "pon música de Juan Gabriel"
   ↓
4. ai_engine.py detecta comando de música
   ↓
5. youtube_controller.py abre YouTube con búsqueda
   ↓
6. Lily responde: "¡Claro Carlos! Buscando Juan Gabriel en YouTube 🎵"
```

### Flujo - Control de Reproducción

```
Video reproduciéndose en YouTube
   ↓
Usuario: "pausa"
   ↓
media_controller.py presiona tecla ESPACIO
   ↓
Video se pausa/reanuda
```

---

## 📁 Archivos del Sistema

### 1. `youtube_controller.py`
**Ubicación**: `models/youtube_controller.py`

**Funciones**:
- `search_and_play(query)` - Busca y abre YouTube con la consulta
- `play_direct_video(video_id)` - Reproduce video específico
- `search_music(artist, song)` - Búsqueda con artista/canción
- `open_youtube_music(query)` - Abre YouTube Music

### 2. `media_controller.py`
**Ubicación**: `models/media_controller.py`

**Funciones**:
- `pause_play()` - Pausar/Reanudar (ESPACIO)
- `next_video()` - Siguiente video (Shift+N)
- `previous_video()` - Video anterior (Shift+P)
- `volume_up(steps=3)` - Subir volumen
- `volume_down(steps=3)` - Bajar volumen
- `mute_unmute()` - Silenciar/Desilenciar
- `fullscreen()` - Pantalla completa (F)

### 3. Modificaciones en `ai_engine.py`
**Ubicación**: `models/ai_engine.py`

**Funciones añadidas**:
- `process_media_command(text)` - Detecta comandos de medios
- `_extract_music_query(text)` - Extrae artista/canción del texto
- Integración en `generate_response()` para detección automática

---

## 🎮 Atajos de Teclado Implementados

### YouTube
| Tecla | Acción |
|-------|--------|
| **ESPACIO** | Pausar / Reproducir |
| **Shift + N** | Siguiente video |
| **Shift + P** | Video anterior |
| **F** | Pantalla completa |
| **M** | Silenciar (en YouTube) |
| **↑ / ↓** | Subir/Bajar volumen (YouTube) |

### Sistema (Windows)
| Tecla | Acción |
|-------|--------|
| **Volume Up** | Subir volumen del sistema |
| **Volume Down** | Bajar volumen del sistema |
| **Volume Mute** | Silenciar/Desilenciar sistema |

---

## 🔧 Requisitos Técnicos

### Dependencias
```txt
pyautogui==0.9.54  # Control de teclado y mouse
```

### Instalación
```bash
pip install pyautogui
```

Ya incluido automáticamente en:
- ✅ `requirements.txt`
- ✅ `Lily_Setup.bat`

### Compatibilidad
- **Sistema operativo**: Windows 10/11
- **Navegador**: Microsoft Edge (recomendado), Chrome, Firefox
- **YouTube**: Versión web (youtube.com)

---

## ⚡ Ejemplos de Uso Completos

### Ejemplo 1: Sesión de Música Completa

```
Usuario: "LILY"
Lily: "¡Hola Carlos! ¿En qué puedo ayudarte?"

Usuario: "pon música de Shakira"
Lily: "¡Claro Carlos! Buscando Shakira en YouTube 🎵"
[Se abre YouTube con resultados de Shakira]

Usuario: "sube volumen"
Lily: "¡Volumen subido Carlos! 🔊"
[Volumen del sistema aumenta]

Usuario: "siguiente"
Lily: "¡Siguiente video Carlos! ⏭️"
[Cambia al siguiente video en la lista]

Usuario: "pausa"
Lily: "¡Listo Carlos! Pausado 🎵"
[Video se pausa]
```

### Ejemplo 2: Control Rápido

```
Usuario: "LILY, pon música de Mozart"
Lily: "¡Claro Carlos! Buscando Mozart en YouTube 🎵"
[Se abre YouTube con música clásica]

Usuario: "silencio"
Lily: "¡Silencio activado Carlos! 🔇"
[Sistema se silencia]

Usuario: "pausa"
Lily: "¡Listo Carlos! 🎵"
[Video se pausa]
```

---

## 🐛 Solución de Problemas

### "No se abre YouTube"

**Posibles causas**:
- No hay conexión a internet
- Microsoft Edge no está instalado
- Problemas con el navegador predeterminado

**Solución**:
1. Verifica conexión a internet
2. Asegúrate de que Microsoft Edge esté instalado
3. Establece Edge como navegador predeterminado
4. Alternativa: Abre manualmente el navegador

---

### "Los atajos de teclado no funcionan"

**Posibles causas**:
- La ventana de YouTube no está en foco
- pyautogui no está instalado correctamente
- Permisos de accesibilidad en Windows

**Solución**:
1. Asegúrate de que la ventana de YouTube esté activa (haz clic en ella)
2. Verifica que pyautogui esté instalado:
   ```bash
   pip show pyautogui
   ```
3. En Windows, ejecuta como administrador si es necesario
4. Verifica que el teclado funcione correctamente

---

### "El volumen no cambia"

**Posibles causas**:
- pyautogui usa las teclas multimedia del sistema
- El teclado no tiene teclas de volumen
- Drivers de audio desactualizados

**Solución**:
1. pyautogui usa las teclas de volumen del sistema (funciona en Windows)
2. Verifica que tu teclado tenga teclas de volumen funcionales
3. Actualiza los drivers de audio de tu computadora
4. Controla el volumen manualmente mientras se soluciona

---

### "Lily no detecta los comandos de música"

**Posibles causas**:
- El comando no coincide con los patrones reconocidos
- Problema con el reconocimiento de voz Vosk

**Solución**:
1. Habla claramente y cerca del micrófono
2. Usa los comandos exactos listados arriba
3. Verifica que el modelo Vosk esté instalado
4. Prueba escribiendo el comando en lugar de usar voz

---

## 🚀 Mejoras Futuras Posibles

### 1. YouTube API Integration
- Buscar videos específicos más precisos
- Control directo de reproducción (play, pause)
- Gestión de playlists
- Reproducción de videos específicos

### 2. Playlists Personalizadas
- "Crea una playlist de..."
- "Reproduce mi playlist de..."
- Guardar playlists favoritas

### 3. Integración con Spotify/Apple Music
- Control de Spotify local
- Integración con Apple Music
- Soporte para Amazon Music
- Gestión de bibliotecas de música

### 4. Comandos Adicionales
- "Adelanta 10 segundos"
- "Retrocede 30 segundos"
- "Activa subtítulos"
- "Cambia calidad a 1080p"
- "Repite este video"

---

## ✅ Resumen

- ✅ **Reproducción de música** en YouTube por comandos de voz
- ✅ **Control de pausa/reproducción** con voz o teclado
- ✅ **Navegación entre videos** (siguiente/anterior)
- ✅ **Control de volumen** del sistema (subir/bajar/silenciar)
- ✅ **Respuestas naturales** de Lily
- ✅ **100% integrado** con wake word "LILY"
- ✅ **Sin anuncios** (usa YouTube estándar)

**¡Lily ahora es tu DJ personal!** 🎵🎉

---

## 📞 Soporte

Si encuentras problemas con el control de medios:

1. Verifica que `pyautogui` esté instalado
2. Asegúrate de que YouTube esté en foco
3. Consulta la sección de Solución de Problemas arriba
4. Reporta issues en: https://github.com/Carlos-VT/LILY-VIRTUAL-3.0/issues

---

*Documentación actualizada para LILY-VIRTUAL-3.0*

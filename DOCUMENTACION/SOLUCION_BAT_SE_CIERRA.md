# 🔧 Solución de Problemas: Archivos .bat se cierran inmediatamente

**Versión**: 3.0 | **Archivos afectados**: `INICIAR_LILY.bat`, `Lily_Setup.bat`

---

## 🎯 Problema

Los archivos `.bat` se abren y cierran inmediatamente sin mostrar mensajes de error, o muestran un error brevemente antes de cerrarse.

---

## 🔍 Causas Comunes

### 1. **Python no está en el PATH** (Más común)
Windows no puede encontrar el comando `python` porque no está agregado a las variables de entorno del sistema.

### 2. **Ubicación incorrecta del archivo .bat**
El archivo `.bat` debe estar en la misma carpeta que `main.py` y los demás archivos del proyecto.

### 3. **Permisos insuficientes**
Windows bloquea la ejecución de scripts por políticas de seguridad.

### 4. **Error en el código Python**
Hay un error al importar módulos o ejecutar el código que causa cierre inesperado.

### 5. **Versión incorrecta de Python**
El script requiere Python 3.11 o superior, pero está instalada una versión anterior.

---

## 🔧 Soluciones Paso a Paso

### **SOLUCIÓN 1: Ejecutar en Modo Debug**

Si tienes problemas, primero diagnosticaremos el error:

1. **Abre CMD manualmente**:
   - Presiona `Windows + R`
   - Escribe `cmd` y presiona Enter

2. **Navega a la carpeta del proyecto**:
   ```cmd
   cd "C:\Ruta\A\LILY-VIRTUAL-3.0"
   ```

3. **Ejecuta el .bat desde CMD**:
   ```cmd
   INICIAR_LILY.bat
   ```

4. **Lee los mensajes de error** que aparecen en la pantalla (no se cerrará automáticamente)

---

### **SOLUCIÓN 2: Verificar Python en PATH**

#### Paso 1: Verificar si Python está instalado

1. Presiona `Windows + R`
2. Escribe `cmd` y presiona Enter
3. En la ventana de CMD, escribe:
   ```cmd
   python --version
   ```

**Resultados posibles**:
- ✅ **Si ves** `Python 3.13.x` (o 3.11, 3.12) → Python está instalado correctamente
- ❌ **Si ves** `'python' no se reconoce...` → Python NO está en el PATH

#### Paso 2: Agregar Python al PATH

**Opción A - Reinstalar Python (Recomendado)**:

1. Descargar Python 3.13 desde https://www.python.org/downloads/
2. Ejecutar el instalador
3. **MUY IMPORTANTE**: Marcar la casilla **"Add Python to PATH"**

   ![Add Python to PATH](https://docs.python.org/3/_images/win_installer.png)

4. Hacer clic en "Install Now"
5. Reiniciar la computadora
6. Verificar nuevamente con `python --version`

**Opción B - Agregar manualmente al PATH**:

1. Busca "Variables de entorno" en el menú de Windows
2. Clic en "Variables de entorno"
3. En "Variables del sistema", busca "Path"
4. Clic en "Editar"
5. Clic en "Nuevo"
6. Agrega la ruta de Python (ejemplo):
   ```
   C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python313\
   ```
7. Agrega también:
   ```
   C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python313\Scripts\
   ```
8. Guardar y reiniciar la computadora

---

### **SOLUCIÓN 3: Ejecutar Manualmente desde CMD**

Si los .bat no funcionan, ejecuta Lily manualmente:

1. **Abrir CMD**:
   - Presiona `Windows + R`
   - Escribe `cmd` y presiona Enter

2. **Navegar a la carpeta del proyecto**:
   ```cmd
   cd "C:\Ruta\A\LILY-VIRTUAL-3.0"
   ```

3. **Instalar dependencias** (si no están instaladas):
   ```cmd
   pip install -r requirements.txt
   ```

4. **Iniciar el servidor**:
   ```cmd
   python main.py
   ```

5. **Abrir el navegador**:
   - Abre Microsoft Edge
   - Ve a: `http://127.0.0.1:8000`

---

### **SOLUCIÓN 4: Ejecutar como Administrador**

1. **Clic derecho** en `INICIAR_LILY.bat`
2. Seleccionar **"Ejecutar como administrador"**
3. Aceptar el mensaje de Control de Cuentas de Usuario (UAC)
4. Verificar si ahora funciona

---

### **SOLUCIÓN 5: Verificar ubicación de archivos**

El archivo `INICIAR_LILY.bat` debe estar en la misma carpeta que estos archivos:

```
LILY-VIRTUAL-3.0/
├── main.py                    ← Debe estar aquí
├── INICIAR_LILY.bat           ← Y aquí
├── requirements.txt           ← Y aquí
├── models/                    ← Y esta carpeta
├── static/                    ← Y esta carpeta
├── templates/                 ← Y esta carpeta
└── ...
```

Si `main.py` está en otra carpeta, **mueve el .bat** a esa misma ubicación.

---

## 🐛 Errores Comunes y Soluciones

### Error: "Python no se reconoce como comando"

**Solución**: Python no está en el PATH. Sigue la **Solución 2** arriba.

---

### Error: "No module named 'fastapi'"

**Causa**: Las dependencias no están instaladas.

**Solución**:
```cmd
pip install -r requirements.txt
```

Si esto falla, prueba con:
```cmd
pip install fastapi uvicorn pydantic vosk pyaudio pyautogui qwen-tts torch soundfile
```

---

### Error: "Address already in use" o "Puerto 8000 ocupado"

**Causa**: Otro programa está usando el puerto 8000.

**Solución**:
```cmd
# Ver qué proceso usa el puerto
netstat -ano | findstr :8000

# Matar el proceso (reemplaza [PID] con el número mostrado)
taskkill /PID [PID] /F
```

Alternativa: Cambia el puerto en `main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8001)  # Cambia a 8001
```

---

### Error: "Permission denied" o "Acceso denegado"

**Causa**: Permisos insuficientes.

**Solución**: Ejecuta como administrador (Solución 4).

---

### Error: "ModuleNotFoundError: No module named 'vosk'"

**Causa**: Vosk no está instalado.

**Solución**:
```cmd
pip install vosk==0.3.45
```

Nota: Vosk a veces requiere dependencias adicionales en Windows.

---

### Error: "No se pudo importar pyaudio"

**Causa**: PyAudio requiere dependencias de sistema en Windows.

**Solución**:
```cmd
# Descargar el wheel apropiado para tu versión de Python desde:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

# Ejemplo para Python 3.13 64-bit:
pip install PyAudio-0.2.14-cp313-cp313-win_amd64.whl
```

---

## 📝 Script Python Alternativo

Si los .bat siguen sin funcionar, crea un archivo `start.py`:

```python
import os
import subprocess
import webbrowser
import time
import sys

print("=" * 60)
print("🌸 Lily AI Assistant - Iniciando 🌸")
print("=" * 60)

# Verificar Python
print("\n[1/4] Verificando Python...")
result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
print(f"Python detectado: {result.stdout.strip()}")

# Instalar dependencias
print("\n[2/4] Verificando dependencias...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
if result.returncode == 0:
    print("✅ Dependencias instaladas")
else:
    print("⚠️ Algunas dependencias pueden faltar")

# Iniciar servidor
print("\n[3/4] Iniciando servidor...")
server = subprocess.Popen([sys.executable, "main.py"],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

# Esperar a que el servidor inicie
time.sleep(5)

# Abrir navegador
print("\n[4/4] Abriendo navegador...")
webbrowser.open("http://127.0.0.1:8000")

print("\n" + "=" * 60)
print("✅ Lily está ejecutándose en http://127.0.0.1:8000")
print("=" * 60)
print("\nPresiona Ctrl+C para detener el servidor")

try:
    server.wait()
except KeyboardInterrupt:
    print("\n\nDeteniendo servidor...")
    server.terminate()
    print("¡Hasta luego!")
```

**Uso**:
```cmd
python start.py
```

---

## 🔍 Verificación Final

Cuando todo funcione correctamente, deberías ver:

```
========================================
   🌸 Lily AI COMPAÑERA VIRTUAL 🌸
========================================

[OK] Python detectado
[OK] Dependencias instaladas
[OK] Ollama detectado

Servidor iniciando en: http://127.0.0.1:8000
========================================
```

Y Microsoft Edge se abrirá automáticamente con la interfaz de Lily.

---

## 📞 Información para Soporte

Si ninguna solución funciona, por favor proporciona esta información al reportar el problema:

1. **Resultado de** `python --version` en CMD
2. **Resultado de** `pip --version` en CMD
3. **Captura de pantalla** de la carpeta del proyecto mostrando los archivos
4. **Mensaje de error** exacto que aparece
5. **Versión de Windows** (Windows 10/11, Home/Pro)
6. **Versión de Python** instalada

**Reporta issues en**: https://github.com/Mijin-VT/LILY-VIRTUAL-3.0/issues

---

## ✅ Checklist de Verificación

- [ ] Python 3.11+ instalado
- [ ] "Add Python to PATH" marcado durante instalación
- [ ] Ollama instalado y ejecutándose
- [ ] Archivos del proyecto en la misma carpeta
- [ ] Ejecutado como administrador (si es necesario)
- [ ] Puerto 8000 disponible (o cambiado en main.py)
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)

---

## 🎯 Resumen Rápido

1. **Ejecuta desde CMD** para ver errores (no se cerrará)
2. **Verifica Python** con `python --version`
3. **Reinstala Python** marcando "Add to PATH" si es necesario
4. **Ejecuta manualmente** desde CMD como alternativa
5. **Envía los mensajes de error** para soporte

**¡No te preocupes, vamos a solucionar esto!** 💪

---

*Documentación actualizada para LILY-VIRTUAL-3.0*

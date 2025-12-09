@echo off
REM ========================================
REM LILY AI COMPAÑERA VIRTUAL - Launcher (Python 3.13)
REM ========================================

title Lily AI Assistant - Python 3.13

echo.
echo ========================================
echo    LILY AI COMPAÑERA VIRTUAL (Python 3.13)
echo ========================================
echo.
echo Iniciando sistema...
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"
echo Directorio actual: %CD%
echo.

REM Usar Python 3.13 específico
set PYTHON_PATH=C:\Users\MIJIN\AppData\Local\Programs\Python\Python313\python.exe
echo [1/3] Usando Python especifico...
"%PYTHON_PATH%" --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.13 no encontrado en la ubicacion esperada
    echo.
    echo SOLUCION:
    echo 1. Verifica que Python 3.13 este instalado correctamente
    echo 2. Asegurate que esta en C:\Users\MIJIN\AppData\Local\Programs\Python\Python313\
    echo.
    pause
    exit /b 1
)
echo [OK] Python 3.13 detectado
echo.

REM Verificar si los paquetes estan instalados
echo [2/3] Verificando paquetes necesarios...
"%PYTHON_PATH%" -c "import fastapi, pydub, faster_whisper, chromadb, av" 2>nul
if errorlevel 1 (
    echo [ADVERTENCIA] Algunos paquetes pueden faltar, instalando...
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] No se pudieron instalar las dependencias
        echo.
        pause
        exit /b 1
    )
    echo [OK] Paquetes instalados
) else (
    echo [OK] Todos los paquetes estan instalados
)
echo.

REM Verificar Ollama (opcional)
echo [3/3] Verificando Ollama...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Ollama no esta ejecutandose
    echo.
    echo La aplicacion funcionara pero sin IA conversacional.
    echo Para habilitar IA:
    echo 1. Instala Ollama desde https://ollama.ai/
    echo 2. Ejecuta: ollama pull mistral
    echo 3. Reinicia esta aplicacion
    echo.
) else (
    echo [OK] Ollama detectado y en linea
)
echo.

REM Iniciar servidor
echo Iniciando servidor...
echo.
echo ========================================
echo Servidor iniciando en: http://127.0.0.1:8000
echo ========================================
echo.
echo Microsoft Edge se abrira en 3 segundos...
echo.
echo Para detener el servidor: Cierra esta ventana o presiona Ctrl+C
echo ========================================
echo.

REM Esperar 3 segundos
timeout /t 3 /nobreak >nul

REM Abrir Microsoft Edge
start msedge http://127.0.0.1:8000

REM Iniciar servidor (mantener ventana abierta)
"%PYTHON_PATH%" main.py

REM Si llegamos aqui, el servidor se detuvo
echo.
echo ========================================
echo Servidor detenido
echo ========================================
echo.
pause


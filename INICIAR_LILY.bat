@echo off
REM ========================================
REM LILY AI COMPANERA VIRTUAL - Launcher (con entorno virtual)
REM ========================================

title Lily AI Assistant

echo.
echo ========================================
echo    LILY AI COMPANERA VIRTUAL
echo ========================================
echo.
echo Iniciando sistema...
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"
echo Directorio actual: %CD%
echo.

REM [1/4] Detectar Python
echo [1/4] Buscando Python...
set "PYTHON_PATH="

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_PATH=python"
    goto :python_found
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_PATH=py"
    goto :python_found
)

for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_PATH=%%D\python.exe"
        goto :python_found
    )
)

echo.
echo [ERROR] Python no encontrado
echo Instala Python 3.11+ desde https://www.python.org/ y marca "Add Python to PATH"
echo.
pause
exit /b 1

:python_found
"%PYTHON_PATH%" --version
echo [OK] Python detectado
echo.

REM [2/4] Crear/usar entorno virtual (aislado del Python global)
echo [2/4] Verificando entorno virtual...
if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual por primera vez...
    "%PYTHON_PATH%" -m venv venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
)
set "PYTHON_PATH=venv\Scripts\python.exe"
echo [OK] Entorno virtual listo
echo.

REM [3/4] Verificar e instalar requirements.txt en el venv
echo [3/4] Verificando dependencias de requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] No se encuentra el archivo requirements.txt
    pause
    exit /b 1
)

fc venv\requirements.txt.installed requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Instalando o actualizando paquetes de requirements.txt en el venv...
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] No se pudieron instalar las dependencias de requirements.txt
        echo.
        pause
        exit /b 1
    )
    copy /y requirements.txt venv\requirements.txt.installed >nul
    echo [OK] Paquetes instalados correctamente
) else (
    echo [OK] Todas las dependencias de requirements.txt ya estan instaladas
)
echo.

REM [4/4] Verificar Ollama y el Modelo Mistral
echo [4/4] Verificando Ollama...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Ollama no esta ejecutandose
    echo La aplicacion funcionara pero sin IA conversacional.
    echo 1. Instala Ollama desde https://ollama.ai/
    echo 2. Ejecuta: ollama pull mistral
    echo.
) else (
    echo [OK] Ollama detectado y en linea
    echo Verificando si el modelo 'mistral' esta instalado...
    ollama show mistral >nul 2>&1
    if errorlevel 1 (
        echo [ADVERTENCIA] El modelo 'mistral' no se encuentra en Ollama.
        echo Descargando 'mistral' - esto puede tardar varios minutos...
        ollama pull mistral
        if errorlevel 1 (
            echo [ERROR] No se pudo descargar el modelo 'mistral' automaticamente.
            echo Por favor, ejecuta en otra consola: ollama pull mistral
        ) else (
            echo [OK] Modelo 'mistral' descargado e instalado con exito
        )
    ) else (
        echo [OK] Modelo 'mistral' ya esta instalado en Ollama
    )
)
echo.

echo Iniciando servidor...
echo.
echo ========================================
echo Servidor iniciando en: http://127.0.0.1:8000
echo ========================================
echo.
echo Microsoft Edge se abrira en 3 segundos...
echo Para detener el servidor: Cierra esta ventana o presiona Ctrl+C
echo ========================================
echo.

ping 127.0.0.1 -n 4 >nul 2>&1

start msedge http://127.0.0.1:8000

"%PYTHON_PATH%" main.py

echo.
echo ========================================
echo Servidor detenido
echo ========================================
echo.
pause

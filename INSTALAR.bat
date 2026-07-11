@echo off
setlocal enabledelayedexpansion
title Instalador Lily AI

echo.
echo ============================================================
echo    🌸 INSTALADOR DE DEPENDENCIAS Y ENTORNOS - LILY AI 🌸
echo ============================================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM ------------------------------------------------------------
REM 1. BUSCAR O INSTALAR PYTHON
REM ------------------------------------------------------------
echo [1/4] Verificando instalacion de Python...

set "PYTHON_CMD="

REM Intentar detectar en el PATH
where python >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py"
    )
)

REM Si no se encuentra en el PATH, buscar en directorios comunes
if "!PYTHON_CMD!"=="" (
    for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" (
            set "PYTHON_CMD=%%D\python.exe"
        )
    )
)
if "!PYTHON_CMD!"=="" (
    for /d %%D in ("%ProgramFiles%\Python\Python3*") do (
        if exist "%%D\python.exe" (
            set "PYTHON_CMD=%%D\python.exe"
        )
    )
)

REM Si despues de buscar sigue vacio, instalar con winget
if "!PYTHON_CMD!"=="" (
    echo [INFO] Python no detectado. Intentando instalar via winget...
    winget --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo Instalando Python 3.11...
        winget install --id Python.Python.3.11 --exact --silent --accept-source-agreements --accept-package-agreements
        if !errorlevel! equ 0 (
            echo [OK] Python 3.11 instalado correctamente.
            REM Volver a buscar la ruta instalada
            for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
                if exist "%%D\python.exe" (
                    set "PYTHON_CMD=%%D\python.exe"
                )
            )
        ) else (
            echo [ERROR] Fallo la instalacion automatica de Python.
            echo Instala Python 3.11+ manualmente desde: https://www.python.org/
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] winget no esta disponible y Python no esta instalado.
        echo Por favor, instala Python 3.11+ manualmente desde https://www.python.org/
        echo Asegurate de marcar la opcion "Add Python to PATH".
        pause
        exit /b 1
    )
)

if not "!PYTHON_CMD!"=="" (
    echo [OK] Python detectado: 
    "!PYTHON_CMD!" --version
)
echo.

REM ------------------------------------------------------------
REM 2. BUSCAR O INSTALAR OLLAMA
REM ------------------------------------------------------------
echo [2/4] Verificando instalacion de Ollama...

set "OLLAMA_CMD="

where ollama >nul 2>&1
if !errorlevel! equ 0 (
    set "OLLAMA_CMD=ollama"
) else (
    if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
        set "OLLAMA_CMD=%LocalAppData%\Programs\Ollama\ollama.exe"
    ) else if exist "%ProgramFiles%\Ollama\ollama.exe" (
        set "OLLAMA_CMD=%ProgramFiles%\Ollama\ollama.exe"
    )
)

if "!OLLAMA_CMD!"=="" (
    echo [INFO] Ollama no detectado. Intentando instalar via winget...
    winget --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo Instalando Ollama...
        winget install --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
        if !errorlevel! equ 0 (
            echo [OK] Ollama instalado correctamente.
            if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
                set "OLLAMA_CMD=%LocalAppData%\Programs\Ollama\ollama.exe"
            ) else (
                set "OLLAMA_CMD=ollama"
            )
        ) else (
            echo [ERROR] Fallo la instalacion automatica de Ollama.
            echo Instala Ollama manualmente desde: https://ollama.ai/
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] winget no esta disponible y Ollama no esta instalado.
        echo Por favor, instala Ollama manualmente desde: https://ollama.ai/
        pause
        exit /b 1
    )
)

if not "!OLLAMA_CMD!"=="" (
    echo [OK] Ollama detectado.
)
echo.

REM ------------------------------------------------------------
REM 3. CONFIGURAR OLLAMA Y MODELO huihui_ai/qwen3-abliterated:0.6b
REM ------------------------------------------------------------
echo [3/4] Verificando servicio de Ollama y modelo huihui_ai/qwen3-abliterated:0.6b...

REM Comprobar si Ollama esta corriendo en el puerto 11434
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] El servicio de Ollama no esta ejecutandose. Iniciandolo...
    if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
        start "" "%LocalAppData%\Programs\Ollama\ollama.exe" serve
    ) else (
        start "" ollama serve
    )
    echo Esperando a que el servicio se inicie (10 segundos)...
    timeout /t 10 >nul
)

REM Volver a comprobar
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if !errorlevel! neq 0 (
    echo [ADVERTENCIA] No se pudo verificar si Ollama se inicio.
    echo Asegurate de que Ollama este corriendo antes de usar a Lily.
) else (
    echo [OK] Servicio de Ollama en linea.
    echo Verificando si el modelo 'huihui_ai/qwen3-abliterated:0.6b' esta instalado...
    
    REM Ejecutar comando show o api
    if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
        "%LocalAppData%\Programs\Ollama\ollama.exe" show huihui_ai/qwen3-abliterated:0.6b >nul 2>&1
    ) else (
        ollama show huihui_ai/qwen3-abliterated:0.6b >nul 2>&1
    )
    
    if !errorlevel! neq 0 (
        echo Descargando modelo 'huihui_ai/qwen3-abliterated:0.6b' - esto puede tardar varios minutos...
        if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
            "%LocalAppData%\Programs\Ollama\ollama.exe" pull huihui_ai/qwen3-abliterated:0.6b
        ) else (
            ollama pull huihui_ai/qwen3-abliterated:0.6b
        )
        if !errorlevel! equ 0 (
            echo [OK] Modelo 'huihui_ai/qwen3-abliterated:0.6b' descargado correctamente.
        ) else (
            echo [ERROR] No se pudo descargar el modelo automaticamente.
            echo Corre 'ollama pull huihui_ai/qwen3-abliterated:0.6b' manualmente mas tarde.
        )
    ) else (
        echo [OK] Modelo 'huihui_ai/qwen3-abliterated:0.6b' ya se encuentra instalado.
    )
)
echo.

REM ------------------------------------------------------------
REM 4. CREAR ENTORNO VIRTUAL E INSTALAR LIBRERIAS
REM ------------------------------------------------------------
echo [4/4] Configurando entorno virtual e instalando requisitos de Python...

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual 'venv'...
    "!PYTHON_CMD!" -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual de Python.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
) else (
    echo [OK] El entorno virtual 'venv' ya existe.
)

echo Actualizando pip e instalando dependencias de requirements.txt...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

if !errorlevel! equ 0 (
    REM Guardar una copia para indicar que ya se instalo
    copy /y requirements.txt venv\requirements.txt.installed >nul
    echo.
    echo ============================================================
    echo   🌸 ¡INSTALACION COMPLETADA CON EXITO! 🌸
    echo ============================================================
    echo.
    echo Todo esta listo. Ahora puedes iniciar la aplicacion con:
    echo --^> INICIAR_LILY.bat
    echo.
) else (
    echo.
    echo [ERROR] Hubo un error al instalar las dependencias de requirements.txt.
    echo Revisa el registro arriba para ver el error.
    echo.
)

pause
exit /b 0

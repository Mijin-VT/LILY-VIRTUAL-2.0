@echo off
REM ========================================================
REM Lanzador de FalkorDB para Lily AI (GraphRAG / Knowledge Graph)
REM ========================================================

title FalkorDB - Lily AI Graph Database

echo.
echo ========================================================
echo       FALKORDB - MOTOR DE GRAFOS PARA LILY AI
echo ========================================================
echo.
echo Verificando estado de FalkorDB en el puerto 6379...
echo.

REM 1. Verificar si Docker esta instalado y en ejecucion
where docker >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Docker no esta instalado en el sistema.
    echo Para usar FalkorDB necesitas Docker Desktop: https://www.docker.com/products/docker-desktop/
    echo.
    echo Lily continuara funcionando con SQLite y ChromaDB como de costumbre.
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Docker Desktop esta instalado pero no esta en ejecucion.
    echo Por favor inicia la aplicacion 'Docker Desktop' en Windows y vuelve a ejecutar este script.
    echo.
    pause
    exit /b 1
)

REM 2. Verificar si el contenedor 'falkordb' ya existe
docker ps -a --format "{{.Names}}" | findstr /i "^falkordb$" >nul 2>&1
if not errorlevel 1 (
    REM El contenedor existe, verificar si esta corriendo
    docker ps --format "{{.Names}}" | findstr /i "^falkordb$" >nul 2>&1
    if not errorlevel 1 (
        echo [OK] El contenedor FalkorDB ya se encuentra en ejecucion en el puerto 6379.
    ) else (
        echo Iniciando contenedor existente de FalkorDB...
        docker start falkordb
        echo [OK] FalkorDB iniciado exitosamente.
    )
) else (
    echo Creando y ejecutando nuevo contenedor de FalkorDB (falkordb/falkordb:latest)...
    echo Esto descargara la imagen oficial de FalkorDB si es la primera vez.
    echo.
    docker run -d -p 6379:6379 --name falkordb -v falkordb_data:/data falkordb/falkordb:latest
    if errorlevel 1 (
        echo [ERROR] No se pudo iniciar el contenedor de FalkorDB.
        pause
        exit /b 1
    )
    echo [OK] Contenedor FalkorDB creado y en ejecucion.
)

echo.
echo ========================================================
echo  FalkorDB activo en: 127.0.0.1:6379
echo  Lily AI se conectara automaticamente en su proximo inicio.
echo ========================================================
echo.
pause

@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM 一键部署：创建 venv → 安装依赖 → 启动 Web 服务
REM 用法:
REM   scripts\start.bat              正常启动
REM   scripts\start.bat --debug      调试模式

set "ROOT=%~dp0.."
pushd "%ROOT%"
set "ROOT=%CD%"
popd
set "VENV=%ROOT%\.venv"
set "DEBUG="

if "%~1"=="--debug" set "DEBUG=--debug"

echo ==> 项目目录: %ROOT%

REM --- venv ---
if not exist "%VENV%\Scripts\activate.bat" (
    echo ==> 创建 virtualenv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo 错误: 创建 virtualenv 失败，请确认 python 已安装且在 PATH 中。
        exit /b 1
    )
)
call "%VENV%\Scripts\activate.bat"

REM --- pip 更新 ---
echo ==> 更新 pip ...
pip install --upgrade pip -q

REM --- 依赖 ---
echo ==> 安装依赖 ...
pip install -q -e "%ROOT%[dev]"

REM --- 数据目录 ---
if not exist "%ROOT%\datastore\registry" mkdir "%ROOT%\datastore\registry"
if not exist "%ROOT%\datastore\store" mkdir "%ROOT%\datastore\store"
if not exist "%ROOT%\data" mkdir "%ROOT%\data"

REM --- 端口检测 ---
if defined AITF_PORT (
    set "PORT=%AITF_PORT%"
) else (
    set "PORT=5000"
)

netstat -ano | findstr "LISTENING" | findstr ":%PORT% " >nul 2>&1
if not errorlevel 1 (
    echo 错误: 端口 %PORT% 已被占用。
    echo 请通过环境变量 AITF_PORT 指定其他端口，例如:
    echo   set AITF_PORT=8080 ^&^& scripts\start.bat
    exit /b 1
)

REM --- 启动 ---
set "EXTRA_ARGS=--port %PORT%"
if defined AITF_HOST set "EXTRA_ARGS=!EXTRA_ARGS! --host %AITF_HOST%"
if defined DEBUG set "EXTRA_ARGS=!EXTRA_ARGS! %DEBUG%"

echo ==> 启动 Web 服务 (端口: %PORT%) ...
aitf web %EXTRA_ARGS%

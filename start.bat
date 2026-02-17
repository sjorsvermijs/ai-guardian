@echo off
REM AI Guardian - Start script (Windows)

echo AI Guardian - Starting...

REM Kill any existing processes on our ports
echo Cleaning up old processes...
for %%p in (8000 5173 5174 5175) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING 2^>nul') do (
        echo   Killing process on port %%p (PID %%a)
        taskkill /PID %%a /F >nul 2>&1
    )
)

set SCRIPT_DIR=%~dp0

REM Start backend
echo Starting backend on :8000...
start "AI Guardian Backend" cmd /c "cd /d %SCRIPT_DIR% && python -m src.backend.api"

REM Start frontend
echo Starting frontend...
start "AI Guardian Frontend" cmd /c "cd /d %SCRIPT_DIR%\src\frontend && npm run dev"

echo.
echo AI Guardian running!
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:5173
echo.
echo Close the terminal windows to stop services.
pause

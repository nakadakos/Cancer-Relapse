@echo off
echo ===================================================
echo   Cancer Relapse Prediction - Backend Setup Runner
echo ===================================================

echo.
echo [1/3] Checking for Python virtual environment...
IF NOT EXIST venv (
    echo Virtual environment not found. Creating one now...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment. Make sure python is installed and in your PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) ELSE (
    echo Virtual environment already exists.
)

echo.
echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting the FastAPI server...
echo The API will be available at http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
uvicorn main:app --reload

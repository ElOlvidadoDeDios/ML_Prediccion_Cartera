@echo off
title Pipeline de ML - Prediccion de Cartera
color 0A

echo =======================================================
echo     INICIANDO PIPELINE DE PREDICCION DE CARTERA
echo =======================================================
echo.

:: 1. Posicionarse en el directorio del proyecto
cd /d D:\Proyectos\ML_Prediccion_Cartera

:: 2. Definir la ruta del Python del entorno virtual
set PYTHON_EXE=env\Scripts\python.exe

:: 3. Ejecutar ETL (Sincronizacion nube a local)
echo [1/4] Extrayendo datos nuevos desde TRANSACMIF (Nube)...
%PYTHON_EXE% src\00_etl_sync.py
if %errorlevel% neq 0 goto error

:: 4. Ejecutar Construccion de Features
echo.
echo [2/4] Calculando quincenas, medias moviles y features...
%PYTHON_EXE% src\01_build_features.py
if %errorlevel% neq 0 goto error

:: 5. Entrenar Modelo
echo.
echo [3/4] Entrenando modelo XGBoost con nueva data...
if exist src\modelo_ops.pkl del src\modelo_ops.pkl
if exist src\modelo_xgboost.pkl del src\modelo_xgboost.pkl
if exist src\columnas_entrenamiento.pkl del src\columnas_entrenamiento.pkl
%PYTHON_EXE% src\02_train_model.py
if %errorlevel% neq 0 goto error

:: 6. Generar Predicciones
echo.
echo [4/4] Generando metas predictivas para el dia siguiente...
%PYTHON_EXE% src\03_daily_predict.py
if %errorlevel% neq 0 goto error

echo.
echo =======================================================
echo      PIPELINE COMPLETADO CON EXITO
echo =======================================================
pause
exit

:error
echo.
color 0C
echo =======================================================
echo      ERROR CRITICO: El pipeline se ha detenido.
echo      Revisa los mensajes de arriba para ver el fallo.
echo =======================================================
pause
exit
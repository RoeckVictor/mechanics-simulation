@echo off
echo Installing PyInstaller if needed...
python -m pip install pyinstaller --quiet

echo.
echo Building MechanicsSimulator.exe ...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "MechanicsSimulator" ^
    --collect-all dearpygui ^
    main.py

echo.
if exist dist\MechanicsSimulator.exe (
    echo  Build successful!
    echo  Executable: dist\MechanicsSimulator.exe
) else (
    echo  Build failed. Re-run without --windowed to see the error:
    echo    pyinstaller --onefile --name MechanicsSimulator --collect-all dearpygui main.py
)
pause

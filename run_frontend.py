#!/usr/bin/env python3
"""
Точка запуска фронтенда приложения

Использование:
    python run_frontend.py
"""
import sys
import subprocess
import os
from pathlib import Path

if __name__ == "__main__":
    # Получаем абсолютные пути
    project_root = Path(__file__).parent.absolute()
    frontend_dir = project_root / "frontend"
    main_file = frontend_dir / "main.py"
    
    if not main_file.exists():
        print(f"❌ Ошибка: файл {main_file} не найден")
        sys.exit(1)
    
    # Меняем рабочую директорию на frontend для корректной работы импортов
    os.chdir(frontend_dir)
    
    # Запускаем streamlit
    print(f"🚀 Запуск приложения на http://127.0.0.1:8501")
    print(f"📁 Рабочая директория: {frontend_dir}")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ])


#!/usr/bin/env python3
"""
Скрипт для извлечения текста из PDF учебника
"""

from pathlib import Path
import json
from config import Config, TextbookMetadata
from utils.pdf_processor import PDFProcessor
from utils.ocr_handler import OCRHandler

def extract_textbook_ocr(
    pdf_path: Path,
    metadata: TextbookMetadata,
    start_page: int = 0,
    end_page: int | None = None
):
    """
    Извлекает текст из учебника
    
    Args:
        pdf_path: Путь к PDF
        metadata: Метаданные учебника
        start_page: Начальная страница (0-indexed)
        end_page: Конечная страница (None = до конца)
    """
    print(f"Обработка учебника: {metadata.title}")
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}"
    output_dir.mkdir(exist_ok=True)
    
    # Инициализация процессоров
    pdf_processor = PDFProcessor(pdf_path)
    ocr_handler = OCRHandler(lang=Config.OCR_LANG, use_gpu=Config.OCR_USE_GPU)
    
    # Определяем диапазон страниц
    total_pages = pdf_processor.get_page_count()
    if end_page is None:
        end_page = total_pages
    
    print(f"Всего страниц: {total_pages}")
    print(f"Обработка страниц: {start_page} - {end_page}")
    
    # Обработка
    results = ocr_handler.process_pdf_pages(
        pdf_processor,
        (start_page, end_page),
        output_dir
    )
    
    # Сохраняем сводный результат
    summary = {
        'metadata': metadata.model_dump(),
        'total_pages_processed': len(results),
        'output_dir': str(output_dir)
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ OCR завершён. Результаты в: {output_dir}")
    
    pdf_processor.close()
    
    return results

if __name__ == "__main__":
    # Пример: Учебник Ткачёвой
    pdf_path = Config.RAW_DIR / "tkacheva_math_5_part1.pdf"
    
    metadata = TextbookMetadata(
        title="Математика. 5 класс. Рабочая тетрадь. Часть 1",
        author="Ткачёва М.В.",
        year=2023,
        grade=5,
        subject="математика",
        part=1
    )
    
    # Для теста обрабатываем только первую главу (стр. 3-17)
    extract_textbook_ocr(
        pdf_path=pdf_path,
        metadata=metadata,
        start_page=2,  # 0-indexed, стр. 3 в PDF
        end_page=17
    )
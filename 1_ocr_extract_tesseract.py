#!/usr/bin/env python3
"""
Скрипт для извлечения текста из PDF учебника с использованием Tesseract OCR
"""

from pathlib import Path
import json
import sys
from config import Config, TextbookMetadata
from utils.pdf_processor import PDFProcessor
from utils.tesseract_handler import TesseractHandler

def extract_textbook_ocr(
    pdf_path: Path,
    metadata: TextbookMetadata,
    start_page: int = 0,
    end_page: int | None = None,
    preprocessing: str = 'advanced',
    psm: int = 3,
    multimode: bool = False,
    dpi: int = 600,
    tesseract_cmd: str = None
):
    """
    Извлекает текст из учебника с помощью Tesseract
    
    Args:
        pdf_path: Путь к PDF
        metadata: Метаданные учебника
        start_page: Начальная страница (0-indexed)
        end_page: Конечная страница (None = до конца)
        preprocessing: Тип предобработки ('none', 'basic', 'advanced', 'aggressive')
        psm: Page Segmentation Mode (0-13)
            3 - Полностью автоматическая сегментация (по умолчанию)
            6 - Единый блок текста
            11 - Разреженный текст
        multimode: Использовать несколько режимов и выбрать лучший (медленнее, но качественнее)
        dpi: Разрешение для извлечения изображений (рекомендуется 600 для OCR)
        tesseract_cmd: Путь к tesseract.exe (если не в PATH)
    """
    print(f"{'='*60}")
    print(f"Обработка учебника: {metadata.title}")
    print(f"Метод: Tesseract OCR")
    print(f"{'='*60}")
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}_tesseract"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Инициализация процессоров
    try:
        pdf_processor = PDFProcessor(pdf_path)
    except Exception as e:
        print(f"✗ Ошибка при открытии PDF: {e}")
        return None
    
    try:
        ocr_handler = TesseractHandler(lang='rus', tesseract_cmd=tesseract_cmd)
    except Exception as e:
        print(f"✗ Ошибка при инициализации Tesseract: {e}")
        print("\nУстановка Tesseract:")
        print("1. Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Linux: sudo apt install tesseract-ocr tesseract-ocr-rus")
        print("3. Mac: brew install tesseract tesseract-lang")
        return None
    
    # Определяем диапазон страниц
    total_pages = pdf_processor.get_page_count()
    if end_page is None:
        end_page = total_pages
    
    print(f"\nВсего страниц в PDF: {total_pages}")
    print(f"Обработка страниц: {start_page} - {end_page}")
    print(f"Разрешение: {dpi} DPI")
    print(f"Предобработка: {preprocessing}")
    print(f"PSM режим: {psm}")
    print(f"Мультирежим: {'Да' if multimode else 'Нет'}")
    print()
    
    # Обработка
    try:
        results = ocr_handler.process_pdf_pages(
            pdf_processor,
            (start_page, end_page),
            output_dir,
            preprocessing=preprocessing,
            psm=psm,
            multimode=multimode,
            dpi=dpi
        )
    except Exception as e:
        print(f"\n✗ Ошибка во время OCR: {e}")
        pdf_processor.close()
        return None
    
    # Анализируем результаты
    total_chars = sum(len(r['text']) for r in results.values())
    avg_confidence = sum(r['confidence'] for r in results.values()) / len(results)
    
    suspicious_pages = []
    for page_num, result in results.items():
        if result['confidence'] < 50:
            suspicious_pages.append({
                'page': page_num,
                'confidence': result['confidence'],
                'text_length': len(result['text'])
            })
    
    # Сохраняем сводный результат
    summary = {
        'metadata': metadata.model_dump(),
        'config': {
            'preprocessing': preprocessing,
            'psm': psm,
            'multimode': multimode,
            'dpi': dpi
        },
        'total_pages_processed': len(results),
        'output_dir': str(output_dir),
        'statistics': {
            'avg_confidence': avg_confidence,
            'total_characters': total_chars,
            'avg_chars_per_page': total_chars / len(results) if results else 0,
            'suspicious_pages_count': len(suspicious_pages)
        },
        'suspicious_pages': suspicious_pages
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ OCR завершён успешно!")
    print(f"{'='*60}")
    print(f"Результаты сохранены в: {output_dir}")
    print(f"\nСтатистика:")
    print(f"  • Обработано страниц: {len(results)}")
    print(f"  • Средняя уверенность: {avg_confidence:.1f}%")
    print(f"  • Всего символов: {total_chars}")
    print(f"  • Символов на страницу: {total_chars / len(results):.0f}")
    print(f"  • Подозрительных страниц: {len(suspicious_pages)}")
    
    if suspicious_pages:
        print(f"\n⚠ Страницы с низкой уверенностью (<50%):")
        for sp in suspicious_pages[:5]:  # Показываем первые 5
            print(f"  • Страница {sp['page']}: {sp['confidence']:.1f}% confidence")
    
    pdf_processor.close()
    
    return results


if __name__ == "__main__":
    # Настройки по умолчанию
    pdf_path = Config.RAW_DIR / "tkacheva_math_5_part1.pdf"
    
    metadata = TextbookMetadata(
        title="Математика. 5 класс. Рабочая тетрадь. Часть 1",
        author="Ткачёва М.В.",
        year=2023,
        grade=5,
        subject="математика",
        part=1
    )
    
    # Путь к tesseract (если не в PATH)
    # Windows пример: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
    
    # Выбор режима обработки
    print("Выберите режим обработки:")
    print("1. Быстрый (advanced preprocessing, PSM 3)")
    print("2. Качественный (aggressive preprocessing, PSM 6)")
    print("3. Мультирежим (самый лучший, но медленный)")
    print("4. Пользовательский")
    
    try:
        choice = input("\nВведите номер (1-4) [1]: ").strip() or "1"
    except:
        choice = "1"
    
    if choice == "1":
        # Быстрый режим
        extract_textbook_ocr(
            pdf_path=pdf_path,
            metadata=metadata,
            start_page=2,
            end_page=17,
            preprocessing='advanced',
            psm=3,
            multimode=False,
            dpi=600,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "2":
        # Качественный режим
        extract_textbook_ocr(
            pdf_path=pdf_path,
            metadata=metadata,
            start_page=2,
            end_page=17,
            preprocessing='aggressive',
            psm=6,
            multimode=False,
            dpi=600,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "3":
        # Мультирежим
        extract_textbook_ocr(
            pdf_path=pdf_path,
            metadata=metadata,
            start_page=2,
            end_page=17,
            preprocessing='advanced',  # Не используется в multimode
            psm=3,  # Не используется в multimode
            multimode=True,
            dpi=600,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "4":
        # Пользовательский режим
        print("\nНастройки:")
        
        try:
            start = int(input("Начальная страница (0-indexed) [2]: ").strip() or "2")
            end = int(input("Конечная страница [17]: ").strip() or "17")
            
            print("\nПредобработка: none, basic, advanced, aggressive")
            prep = input("Выберите [advanced]: ").strip() or "advanced"
            
            psm_mode = int(input("PSM режим (0-13) [3]: ").strip() or "3")
            
            dpi_val = int(input("DPI (300-1200) [600]: ").strip() or "600")
            
            multi = input("Мультирежим? (y/n) [n]: ").strip().lower() == 'y'
            
            extract_textbook_ocr(
                pdf_path=pdf_path,
                metadata=metadata,
                start_page=start,
                end_page=end,
                preprocessing=prep,
                psm=psm_mode,
                multimode=multi,
                dpi=dpi_val,
                tesseract_cmd=tesseract_cmd
            )
        except Exception as e:
            print(f"Ошибка ввода: {e}")
            print("Использую настройки по умолчанию...")
            extract_textbook_ocr(
                pdf_path=pdf_path,
                metadata=metadata,
                start_page=2,
                end_page=17,
                tesseract_cmd=tesseract_cmd
            )
    
    else:
        print("Неверный выбор, использую быстрый режим...")
        extract_textbook_ocr(
            pdf_path=pdf_path,
            metadata=metadata,
            start_page=2,
            end_page=17,
            tesseract_cmd=tesseract_cmd
        )
#!/usr/bin/env python3
"""
Улучшенный скрипт извлечения текста с максимальным качеством OCR
"""

from pathlib import Path
import json
from config import Config, TextbookMetadata
from utils.pdf_processor import PDFProcessor
from utils.ocr_handler import OCRHandler
 

def extract_textbook_ocr_v2(
    pdf_path: Path,
    metadata: TextbookMetadata,
    start_page: int = 0,
    end_page: int | None = None,
    dpi: int = 600,
    use_preprocessing: bool = True,
    min_confidence: float = 0.5,
    save_processed_images: bool = False
):
    """
    Извлекает текст из учебника с улучшенным качеством
    
    Args:
        pdf_path: Путь к PDF
        metadata: Метаданные учебника
        start_page: Начальная страница (0-indexed)
        end_page: Конечная страница (None = до конца)
        dpi: Разрешение для экспорта (рекомендуется 600)
        use_preprocessing: Применять предобработку изображений
        min_confidence: Минимальный порог уверенности
        save_processed_images: Сохранять предобработанные изображения
    """
    print("="*60)
    print(f"УЛУЧШЕННЫЙ OCR ЭКСТРАКТОР V4 (с китайской моделью)")
    print("="*60)
    print(f"Учебник: {metadata.title}")
    print(f"DPI: {dpi}")
    print(f"Предобработка: {'ДА' if use_preprocessing else 'НЕТ'}")
    print(f"Мин. уверенность: {min_confidence}")
    print("="*60)
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}_v4"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Инициализация процессоров
    pdf_processor = PDFProcessor(pdf_path)
    ocr_handler = OCRHandler(
        lang=Config.OCR_LANG,
        use_gpu=Config.OCR_USE_GPU
    )
    
    # Определяем диапазон страниц
    total_pages = pdf_processor.get_page_count()
    if end_page is None:
        end_page = total_pages
    
    print(f"\nВсего страниц: {total_pages}")
    print(f"Обработка страниц: {start_page} - {end_page}\n")
    
    # Обработка
    results = ocr_handler.process_pdf_pages(
        pdf_processor,
        (start_page, end_page),
        output_dir
    )
    
    # Анализируем качество распознавания
    all_stats = []
    suspicious_pages = []
    
    for page_num, page_data in results.items():
        ocr_results = page_data['ocr_results']
        
        # Вычисляем статистику для страницы
        if ocr_results:
            confidences = [item['confidence'] for item in ocr_results]
            avg_confidence = sum(confidences) / len(confidences)
            total_lines = len(ocr_results)
            low_confidence_count = sum(1 for c in confidences if c < 0.7)
        else:
            avg_confidence = 0.0
            total_lines = 0
            low_confidence_count = 0
        
        stats = {
            'avg_confidence': avg_confidence,
            'total_lines': total_lines,
            'low_confidence_count': low_confidence_count
        }
        all_stats.append(stats)
        
        # Помечаем подозрительные страницы
        if (avg_confidence < 0.7 or 
            total_lines < 5 or
            low_confidence_count > total_lines * 0.3):
            suspicious_pages.append({
                'page': page_num,
                'reason': 'low_confidence' if avg_confidence < 0.7 else 'too_short',
                'avg_confidence': avg_confidence,
                'total_lines': total_lines
            })
    
    # Общая статистика
    if all_stats:
        avg_confidence = sum(s['avg_confidence'] for s in all_stats) / len(all_stats)
        total_lines = sum(s['total_lines'] for s in all_stats)
    else:
        avg_confidence = 0.0
        total_lines = 0
    
    # Сохраняем сводный результат
    summary = {
        'metadata': metadata.model_dump(),
        'config': {
            'dpi': dpi,
            'preprocessing': use_preprocessing,
            'min_confidence': min_confidence
        },
        'total_pages_processed': len(results),
        'output_dir': str(output_dir),
        'overall_statistics': {
            'avg_confidence': avg_confidence,
            'total_lines_recognized': total_lines,
            'suspicious_pages_count': len(suspicious_pages)
        },
        'suspicious_pages': suspicious_pages
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ OCR")
    print("="*60)
    print(f"✓ Обработано страниц: {len(results)}")
    print(f"✓ Распознано строк: {total_lines}")
    print(f"✓ Средняя уверенность: {avg_confidence:.2%}")
    
    if suspicious_pages:
        print(f"\n⚠ Подозрительные страницы ({len(suspicious_pages)}):")
        for sp in suspicious_pages:
            print(f"  - Страница {sp['page']}: {sp['reason']} "
                  f"(уверенность: {sp['avg_confidence']:.2%}, "
                  f"строк: {sp['total_lines']})")
    
    print(f"\n📁 Результаты сохранены в: {output_dir}")
    print(f"📄 Сводка: {summary_path}")
    
    return results, summary


def main():
    """Основная функция для запуска OCR извлечения"""
    # Настройки для тестирования
    pdf_path = Config.RAW_DIR / "matematika_-rt_-2-ch_-5-kl_-tkacheva.pdf"
    
    metadata = TextbookMetadata(
        title="Математика 5 класс (Ткачева)",
        subject="математика",
        grade=5,
        author="Ткачева М.В.",
        publisher="Просвещение",
        year=2023
    )
    
    try:
        results, summary = extract_textbook_ocr_v2(
            pdf_path=pdf_path,
            metadata=metadata,
            start_page=0,
            end_page=None,  # Обрабатываем все страницы учебника
            dpi=600,
            use_preprocessing=True,
            min_confidence=0.5,
            save_processed_images=False
        )
        
        print("\n✅ OCR извлечение завершено успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении OCR: {e}")
        raise


if __name__ == "__main__":
    main()
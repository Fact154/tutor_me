#!/usr/bin/env python3
"""
Скрипт для OCR обработки готовых изображений страниц с использованием PaddleOCR
(Не требует установки Tesseract!)
"""

from pathlib import Path
import json
import sys
from config import Config, TextbookMetadata
from utils.ocr_handler import OCRHandler
from PIL import Image
import numpy as np

def process_images_ocr_paddle(
    images_dir: Path,
    metadata: TextbookMetadata,
    lang: str = 'ru',
    use_gpu: bool = False
):
    """
    Обрабатывает готовые изображения страниц с помощью PaddleOCR
    
    Args:
        images_dir: Путь к папке с изображениями (например, data/raw/pdf/tkacheva_math_5_part1/)
        metadata: Метаданные учебника
        lang: Язык распознавания ('ru', 'en', 'ch', 'korean', 'japan')
        use_gpu: Использовать GPU (если доступно)
    """
    print(f"{'='*60}")
    print(f"OCR обработка изображений: {metadata.title}")
    print(f"Метод: PaddleOCR (не требует Tesseract!)")
    print(f"Источник: {images_dir}")
    print(f"{'='*60}")
    
    # Проверяем существование папки
    if not images_dir.exists():
        print(f"✗ Папка не найдена: {images_dir}")
        return None
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}_from_images_paddle"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Инициализация OCR
    print(f"\n🔄 Инициализация PaddleOCR (язык: {lang})...")
    print("   ⚠ Первый запуск может занять время - загружаются модели")
    try:
        ocr_handler = OCRHandler(lang=lang, use_gpu=use_gpu)
        print("   ✓ PaddleOCR готов к работе!")
    except Exception as e:
        print(f"✗ Ошибка при инициализации PaddleOCR: {e}")
        return None
    
    # Находим все изображения
    image_files = sorted(images_dir.glob("page_*.png")) + sorted(images_dir.glob("page_*.jpg"))
    
    if not image_files:
        print(f"✗ Изображения не найдены в {images_dir}")
        return None
    
    print(f"\n📊 Найдено изображений: {len(image_files)}")
    print(f"🌐 Язык: {lang}")
    print(f"🎮 GPU: {'Да' if use_gpu else 'Нет'}")
    print()
    
    results = {}
    
    # Обрабатываем каждое изображение
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Извлекаем номер страницы из имени файла
            page_num = int(img_path.stem.split('_')[1])
            
            print(f"[{idx}/{len(image_files)}] Обработка страницы {page_num}...", end=' ', flush=True)
            
            # Загружаем изображение
            img = Image.open(img_path)
            
            # Выполняем OCR
            ocr_results = ocr_handler.process_image(img)
            
            # Извлекаем текст
            text = ocr_handler.extract_text_only(ocr_results)
            
            # Вычисляем среднюю уверенность
            if ocr_results:
                avg_confidence = sum(r['confidence'] for r in ocr_results) / len(ocr_results) * 100
            else:
                avg_confidence = 0.0
            
            # Сохраняем результат для страницы
            page_result = {
                'page': page_num,
                'text': text,
                'confidence': avg_confidence,
                'image_file': img_path.name,
                'ocr_results': ocr_results,
                'config': {
                    'lang': lang,
                    'use_gpu': use_gpu
                }
            }
            
            # Сохраняем в JSON
            output_file = output_dir / f"page_{page_num:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(page_result, f, ensure_ascii=False, indent=2)
            
            results[page_num] = page_result
            
            # Выводим статистику
            chars = len(text)
            conf = avg_confidence
            print(f"✓ {chars} символов, уверенность: {conf:.1f}%")
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Анализируем результаты
    if not results:
        print("\n✗ Не удалось обработать ни одного изображения")
        return None
    
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
        'source_directory': str(images_dir),
        'config': {
            'lang': lang,
            'use_gpu': use_gpu,
            'ocr_engine': 'PaddleOCR'
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
        for sp in suspicious_pages[:5]:
            print(f"  • Страница {sp['page']}: {sp['confidence']:.1f}% confidence")
    
    return results


if __name__ == "__main__":
    # Путь к папке с изображениями
    images_dir = Path("data/raw/pdf/tkacheva_math_5_part1")
    
    metadata = TextbookMetadata(
        title="Математика. 5 класс. Рабочая тетрадь. Часть 1",
        author="Ткачёва М.В.",
        year=2023,
        grade=5,
        subject="математика",
        part=1
    )
    
    # Выбор языка
    print("Выберите язык распознавания:")
    print("1. Русский (ru) - рекомендуется")
    print("2. Английский (en)")
    print("3. Русский + Английский (ru+en)")
    
    try:
        lang_choice = input("\nВведите номер (1-3) [1]: ").strip() or "1"
    except:
        lang_choice = "1"
    
    lang_map = {
        "1": "ru",
        "2": "en",
        "3": "ru+en"
    }
    lang = lang_map.get(lang_choice, "ru")
    
    # Запускаем обработку
    process_images_ocr_paddle(
        images_dir=images_dir,
        metadata=metadata,
        lang=lang,
        use_gpu=False  # Можно включить, если есть GPU
    )


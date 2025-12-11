#!/usr/bin/env python3
"""
Скрипт для OCR обработки изображений учебника истории
"""

from pathlib import Path
import json
import sys
from config import Config, TextbookMetadata
from utils.tesseract_handler import TesseractHandler
from PIL import Image
import shutil

def process_history_images():
    """
    Обрабатывает изображения учебника истории
    """
    # Путь к папке с изображениями
    images_dir = Path("data/raw/pdf/istoriya-drevnego-mira-5kl")
    
    metadata = TextbookMetadata(
        title="История Древнего мира. 5 класс",
        author="",
        year=2023,
        grade=5,
        subject="история",
        part=1
    )
    
    print(f"{'='*60}")
    print(f"OCR обработка: {metadata.title}")
    print(f"Источник: {images_dir}")
    print(f"{'='*60}")
    
    # Проверяем существование папки
    if not images_dir.exists():
        print(f"✗ Папка не найдена: {images_dir}")
        return None
    
    # Находим все изображения
    image_files = sorted(images_dir.glob("page_*.png")) + sorted(images_dir.glob("page_*.jpg"))
    
    if not image_files:
        print(f"✗ Изображения не найдены в {images_dir}")
        return None
    
    print(f"\n📊 Найдено изображений: {len(image_files)}")
    
    # Спрашиваем диапазон страниц
    print("\nВыберите режим обработки:")
    print("1. Обработать все страницы (может занять несколько часов)")
    print("2. Обработать диапазон страниц")
    print("3. Тестовая обработка (первые 10 страниц)")
    
    try:
        choice = input("\nВведите номер (1-3) [3]: ").strip() or "3"
    except:
        choice = "3"
    
    if choice == "1":
        start_idx = 0
        end_idx = len(image_files)
    elif choice == "2":
        try:
            start = int(input(f"Начальная страница (1-{len(image_files)}) [1]: ").strip() or "1")
            end = int(input(f"Конечная страница (1-{len(image_files)}) [{min(50, len(image_files))}]: ").strip() or str(min(50, len(image_files))))
            start_idx = max(0, start - 1)
            end_idx = min(len(image_files), end)
        except:
            print("Ошибка ввода, использую первые 10 страниц")
            start_idx = 0
            end_idx = 10
    else:
        start_idx = 0
        end_idx = min(10, len(image_files))
    
    # Выбор режима OCR
    print("\nВыберите режим OCR:")
    print("1. Быстрый (advanced preprocessing, PSM 3)")
    print("2. Качественный (aggressive preprocessing, PSM 6)")
    print("3. Мультирежим (самый лучший, но медленный)")
    
    try:
        ocr_choice = input("\nВведите номер (1-3) [1]: ").strip() or "1"
    except:
        ocr_choice = "1"
    
    if ocr_choice == "2":
        preprocessing = 'aggressive'
        psm = 6
        multimode = False
    elif ocr_choice == "3":
        preprocessing = 'advanced'
        psm = 3
        multimode = True
    else:
        preprocessing = 'advanced'
        psm = 3
        multimode = False
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}_from_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Автоматический поиск Tesseract
    tesseract_cmd = None
    possible_paths = [
        Path("D:/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ]
    
    tesseract_in_path = shutil.which("tesseract")
    if tesseract_in_path:
        tesseract_cmd = tesseract_in_path
    else:
        for path in possible_paths:
            if path.exists():
                tesseract_cmd = str(path)
                break
    
    # Устанавливаем переменную окружения TESSDATA_PREFIX
    import os
    possible_tessdata = [
        Path('C:/Program Files/Tesseract-OCR/tessdata'),
        Path('C:/Program Files (x86)/Tesseract-OCR/tessdata'),
        Path('D:/Tesseract-OCR/tessdata'),
    ]
    
    if tesseract_cmd:
        possible_tessdata.insert(0, Path(tesseract_cmd).parent / 'tessdata')
    
    for td in possible_tessdata:
        if td.exists() and (td / 'rus.traineddata').exists():
            os.environ['TESSDATA_PREFIX'] = str(td)
            print(f"✓ Найден tessdata с rus.traineddata: {td}")
            break
    
    # Инициализация OCR
    try:
        ocr_handler = TesseractHandler(lang='rus', tesseract_cmd=tesseract_cmd)
    except Exception as e:
        print(f"✗ Ошибка при инициализации Tesseract: {e}")
        return None
    
    print(f"\n📊 Обработка страниц: {start_idx + 1}-{end_idx} из {len(image_files)}")
    print(f"🖼️  Предобработка: {preprocessing}")
    print(f"🎯 PSM режим: {psm}")
    print(f"🔄 Мультирежим: {'Да' if multimode else 'Нет'}")
    print()
    
    results = {}
    images_to_process = image_files[start_idx:end_idx]
    
    # Обрабатываем каждое изображение
    for idx, img_path in enumerate(images_to_process, 1):
        try:
            # Извлекаем номер страницы из имени файла
            page_num = int(img_path.stem.split('_')[1])
            
            print(f"[{idx}/{len(images_to_process)}] Обработка страницы {page_num}...", end=' ', flush=True)
            
            # Загружаем изображение
            img = Image.open(img_path)
            
            # Выполняем OCR
            if multimode:
                result = ocr_handler.ocr_multimode(img)
            else:
                result = ocr_handler.ocr_image(
                    img, 
                    preprocessing=preprocessing,
                    psm=psm
                )
            
            # Сохраняем результат для страницы
            page_result = {
                'page': page_num,
                'text': result['text'],
                'confidence': result['confidence'],
                'image_file': img_path.name,
                'config': {
                    'preprocessing': preprocessing if not multimode else 'multimode',
                    'psm': psm
                }
            }
            
            # Сохраняем в JSON
            output_file = output_dir / f"page_{page_num:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(page_result, f, ensure_ascii=False, indent=2)
            
            results[page_num] = page_result
            
            # Выводим статистику
            chars = len(result['text'])
            conf = result['confidence']
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
            'preprocessing': preprocessing,
            'psm': psm,
            'multimode': multimode
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
    process_history_images()



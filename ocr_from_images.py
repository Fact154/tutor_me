#!/usr/bin/env python3
"""
Скрипт для OCR обработки готовых изображений страниц
"""

from pathlib import Path
import json
import sys
from config import Config, TextbookMetadata
from utils.tesseract_handler import TesseractHandler
from PIL import Image

def process_images_ocr(
    images_dir: Path,
    metadata: TextbookMetadata,
    preprocessing: str = 'advanced',
    psm: int = 3,
    multimode: bool = False,
    tesseract_cmd: str = None
):
    """
    Обрабатывает готовые изображения страниц с помощью Tesseract
    
    Args:
        images_dir: Путь к папке с изображениями (например, data/raw/pdf/tkacheva_math_5_part1/)
        metadata: Метаданные учебника
        preprocessing: Тип предобработки ('none', 'basic', 'advanced', 'aggressive')
        psm: Page Segmentation Mode (0-13)
        multimode: Использовать несколько режимов
        tesseract_cmd: Путь к tesseract.exe
    """
    print(f"{'='*60}")
    print(f"OCR обработка изображений: {metadata.title}")
    print(f"Источник: {images_dir}")
    print(f"{'='*60}")
    
    # Проверяем существование папки
    if not images_dir.exists():
        print(f"✗ Папка не найдена: {images_dir}")
        return None
    
    # Создаём директорию для результатов
    output_dir = Config.OCR_DIR / f"{metadata.subject}_{metadata.grade}_from_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Устанавливаем переменную окружения TESSDATA_PREFIX
    import os
    possible_tessdata = []
    
    if tesseract_cmd:
        possible_tessdata.append(Path(tesseract_cmd).parent / 'tessdata')
    
    # Всегда проверяем стандартные места
    possible_tessdata.extend([
        Path('C:/Program Files/Tesseract-OCR/tessdata'),
        Path('C:/Program Files (x86)/Tesseract-OCR/tessdata'),
        Path('D:/Tesseract-OCR/tessdata'),
    ])
    
    # Ищем папку с rus.traineddata
    for td in possible_tessdata:
        if td.exists() and (td / 'rus.traineddata').exists():
            os.environ['TESSDATA_PREFIX'] = str(td)
            print(f"✓ Найден tessdata с rus.traineddata: {td}")
            break
    else:
        print("⚠ Внимание: TESSDATA_PREFIX не установлен, Tesseract будет искать в стандартных местах")
    
    # Инициализация OCR
    try:
        ocr_handler = TesseractHandler(lang='rus', tesseract_cmd=tesseract_cmd)
    except Exception as e:
        print(f"✗ Ошибка при инициализации Tesseract: {e}")
        print("\nУстановка Tesseract:")
        print("1. Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Linux: sudo apt install tesseract-ocr tesseract-ocr-rus")
        print("3. Mac: brew install tesseract tesseract-lang")
        print("\n⚠ Убедитесь, что установлен русский язык (rus.traineddata)")
        return None
    
    # Находим все изображения
    image_files = sorted(images_dir.glob("page_*.png")) + sorted(images_dir.glob("page_*.jpg"))
    
    if not image_files:
        print(f"✗ Изображения не найдены в {images_dir}")
        return None
    
    print(f"\nНайдено изображений: {len(image_files)}")
    print(f"Предобработка: {preprocessing}")
    print(f"PSM режим: {psm}")
    print(f"Мультирежим: {'Да' if multimode else 'Нет'}")
    print()
    
    results = {}
    
    # Обрабатываем каждое изображение
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Извлекаем номер страницы из имени файла
            page_num = int(img_path.stem.split('_')[1])
            
            print(f"[{idx}/{len(image_files)}] Обработка страницы {page_num}...", end=' ')
            
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
    
    # Путь к tesseract (если не в PATH)
    # Попробуем найти Tesseract автоматически
    import shutil
    tesseract_cmd = None
    
    # Проверяем стандартные пути
    possible_paths = [
        r"D:\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    # Проверяем PATH
    tesseract_in_path = shutil.which("tesseract")
    if tesseract_in_path:
        tesseract_cmd = tesseract_in_path
    else:
        # Проверяем возможные пути
        for path in possible_paths:
            if Path(path).exists():
                tesseract_cmd = path
                break
    
    if not tesseract_cmd:
        print("\n⚠ Tesseract не найден!")
        print("\n📥 Установка Tesseract на диск D:")
        print("1. Скачайте установщик: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. При установке выберите диск D: (например, D:\\Tesseract-OCR)")
        print("3. Убедитесь, что установлен русский язык (rus)")
        print("\nИли укажите путь к tesseract.exe вручную в скрипте")
        sys.exit(1)
    
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
        process_images_ocr(
            images_dir=images_dir,
            metadata=metadata,
            preprocessing='advanced',
            psm=3,
            multimode=False,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "2":
        # Качественный режим
        process_images_ocr(
            images_dir=images_dir,
            metadata=metadata,
            preprocessing='aggressive',
            psm=6,
            multimode=False,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "3":
        # Мультирежим
        process_images_ocr(
            images_dir=images_dir,
            metadata=metadata,
            preprocessing='advanced',
            psm=3,
            multimode=True,
            tesseract_cmd=tesseract_cmd
        )
    
    elif choice == "4":
        # Пользовательский режим
        print("\nНастройки:")
        
        try:
            print("\nПредобработка: none, basic, advanced, aggressive")
            prep = input("Выберите [advanced]: ").strip() or "advanced"
            
            psm_mode = int(input("PSM режим (0-13) [3]: ").strip() or "3")
            
            multi = input("Мультирежим? (y/n) [n]: ").strip().lower() == 'y'
            
            process_images_ocr(
                images_dir=images_dir,
                metadata=metadata,
                preprocessing=prep,
                psm=psm_mode,
                multimode=multi,
                tesseract_cmd=tesseract_cmd
            )
        except Exception as e:
            print(f"Ошибка ввода: {e}")
            print("Использую настройки по умолчанию...")
            process_images_ocr(
                images_dir=images_dir,
                metadata=metadata,
                tesseract_cmd=tesseract_cmd
            )
    
    else:
        print("Неверный выбор, использую быстрый режим...")
        process_images_ocr(
            images_dir=images_dir,
            metadata=metadata,
            tesseract_cmd=tesseract_cmd
        )


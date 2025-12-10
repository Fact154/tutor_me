"""
Скрипт для конвертации PDF файлов в изображения.
Каждая страница PDF сохраняется как отдельное изображение.
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    import PyMuPDF as fitz

from PIL import Image
from pathlib import Path
import argparse
from typing import Optional
import sys


class PDFToImageConverter:
    """Конвертер PDF в изображения"""
    
    def __init__(self, pdf_path: str, output_dir: Optional[str] = None, dpi: int = 300):
        """
        Инициализация конвертера
        
        Args:
            pdf_path: Путь к PDF файлу
            output_dir: Директория для сохранения изображений (по умолчанию создается в data/raw/pdf/)
            dpi: Разрешение изображений (по умолчанию 300)
        """
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
        
        # Определяем директорию для сохранения
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Создаем структуру: data/raw/pdf/название_файла/
            pdf_parent = self.pdf_path.parent  # data/raw
            self.output_dir = pdf_parent / "pdf" / self.pdf_path.stem
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Открываем PDF
        try:
            self.doc = fitz.open(str(self.pdf_path))
        except Exception as e:
            raise Exception(f"Ошибка при открытии PDF: {e}")
    
    def convert_page(self, page_num: int, image_format: str = "png") -> Path:
        """
        Конвертирует одну страницу PDF в изображение
        
        Args:
            page_num: Номер страницы (начиная с 0)
            image_format: Формат изображения (png, jpg, jpeg)
        
        Returns:
            Path: Путь к сохраненному изображению
        """
        if page_num < 0 or page_num >= len(self.doc):
            raise ValueError(f"Неверный номер страницы: {page_num}. Всего страниц: {len(self.doc)}")
        
        # Получаем страницу
        page = self.doc[page_num]
        
        # Увеличиваем разрешение
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертируем в PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Формируем имя файла
        output_filename = f"page_{page_num + 1:04d}.{image_format}"
        output_path = self.output_dir / output_filename
        
        # Сохраняем изображение
        if image_format.lower() in ['jpg', 'jpeg']:
            img.save(output_path, 'JPEG', quality=95)
        else:
            img.save(output_path, 'PNG')
        
        return output_path
    
    def convert_all(self, image_format: str = "png", start_page: Optional[int] = None, 
                    end_page: Optional[int] = None) -> list[Path]:
        """
        Конвертирует все страницы PDF в изображения
        
        Args:
            image_format: Формат изображения (png, jpg, jpeg)
            start_page: Начальная страница (по умолчанию с первой)
            end_page: Конечная страница (по умолчанию до последней)
        
        Returns:
            list[Path]: Список путей к сохраненным изображениям
        """
        total_pages = len(self.doc)
        
        # Определяем диапазон страниц
        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages
        
        # Проверяем корректность диапазона
        start = max(0, start)
        end = min(total_pages, end)
        
        if start >= end:
            raise ValueError(f"Некорректный диапазон страниц: {start}-{end}")
        
        print(f"📄 Конвертация PDF: {self.pdf_path.name}")
        print(f"📊 Всего страниц в PDF: {total_pages}")
        print(f"🎯 Конвертируем страницы: {start + 1}-{end}")
        print(f"💾 Сохранение в: {self.output_dir}")
        print(f"🖼️  Формат: {image_format.upper()}, DPI: {self.dpi}")
        print()
        
        saved_images = []
        
        for page_num in range(start, end):
            try:
                output_path = self.convert_page(page_num, image_format)
                saved_images.append(output_path)
                print(f"✅ Страница {page_num + 1}/{total_pages} → {output_path.name}")
            except Exception as e:
                print(f"❌ Ошибка на странице {page_num + 1}: {e}")
        
        print()
        print(f"✨ Готово! Сохранено {len(saved_images)} изображений")
        
        return saved_images
    
    def close(self):
        """Закрывает PDF документ"""
        if hasattr(self, 'doc'):
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Конвертация PDF файлов в изображения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Конвертировать весь PDF в PNG (300 DPI)
  # Сохранится в data/raw/pdf/название_файла/
  python pdf_to_images.py data/raw/document.pdf
  
  # Конвертировать в JPG с разрешением 150 DPI
  python pdf_to_images.py data/raw/document.pdf --format jpg --dpi 150
  
  # Конвертировать только страницы 1-10
  python pdf_to_images.py data/raw/document.pdf --start 1 --end 10
  
  # Указать свою папку для сохранения
  python pdf_to_images.py data/raw/document.pdf --output ./my_images
        """
    )
    
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Путь к PDF файлу'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Директория для сохранения изображений (по умолчанию: data/raw/pdf/название_файла/)'
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['png', 'jpg', 'jpeg'],
        default='png',
        help='Формат изображений (по умолчанию: png)'
    )
    
    parser.add_argument(
        '-d', '--dpi',
        type=int,
        default=300,
        help='Разрешение изображений в DPI (по умолчанию: 300)'
    )
    
    parser.add_argument(
        '-s', '--start',
        type=int,
        default=None,
        help='Начальная страница (нумерация с 1)'
    )
    
    parser.add_argument(
        '-e', '--end',
        type=int,
        default=None,
        help='Конечная страница (включительно)'
    )
    
    args = parser.parse_args()
    
    try:
        # Корректируем номера страниц (пользователь считает с 1, программа с 0)
        start_page = args.start - 1 if args.start is not None else None
        end_page = args.end if args.end is not None else None
        
        # Создаем конвертер и выполняем конвертацию
        with PDFToImageConverter(args.pdf_path, args.output, args.dpi) as converter:
            converter.convert_all(
                image_format=args.format,
                start_page=start_page,
                end_page=end_page
            )
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

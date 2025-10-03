import fitz  # PyMuPDF
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict
import json

class PDFProcessor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def get_page_count(self) -> int:
        return len(self.doc)
    
    def extract_page_as_image(self, page_num: int, dpi: int = 300) -> Image.Image:
        """Извлекает страницу как изображение"""
        page = self.doc[page_num]
        
        # Увеличиваем разрешение для лучшего OCR
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертируем в PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    
    def extract_text_native(self, page_num: int) -> str:
        """Пытается извлечь текст напрямую из PDF"""
        page = self.doc[page_num]
        return page.get_text()
    
    def get_page_dimensions(self, page_num: int) -> Dict:
        """Получает размеры страницы"""
        page = self.doc[page_num]
        rect = page.rect
        return {
            "width": rect.width,
            "height": rect.height
        }
    
    def close(self):
        self.doc.close()
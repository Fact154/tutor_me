from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
from tqdm import tqdm

class OCRHandler:
    def __init__(self, lang='ru', use_gpu=False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
    
    def process_image(self, image: Image.Image) -> List[Dict]:
        """
        Обрабатывает изображение и возвращает распознанный текст
        
        Returns:
            List[Dict]: [{
                'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                'text': str,
                'confidence': float
            }]
        """
        # Конвертируем PIL Image в numpy array
        img_array = np.array(image)
        
        # OCR
        result = self.ocr.ocr(img_array, cls=True)
        
        if result is None or len(result) == 0:
            return []
        
        # Парсим результаты
        parsed_results = []
        for line in result[0]:
            bbox = line[0]
            text_info = line[1]
            
            parsed_results.append({
                'bbox': bbox,
                'text': text_info[0],
                'confidence': float(text_info[1])
            })
        
        return parsed_results
    
    def extract_text_only(self, results: List[Dict]) -> str:
        """Извлекает только текст из результатов OCR"""
        return '\n'.join([item['text'] for item in results])
    
    def process_pdf_pages(
        self, 
        pdf_processor, 
        page_range: Tuple[int, int],
        output_dir: Path
    ) -> Dict[int, Dict]:
        """
        Обрабатывает страницы PDF
        
        Returns:
            Dict[page_num, {
                'text': str,
                'ocr_results': List[Dict],
                'dimensions': Dict
            }]
        """
        results = {}
        
        start_page, end_page = page_range
        
        for page_num in tqdm(range(start_page, end_page), desc="OCR Processing"):
            # Извлекаем изображение
            image = pdf_processor.extract_page_as_image(page_num)
            
            # OCR
            ocr_results = self.process_image(image)
            
            # Извлекаем текст
            text = self.extract_text_only(ocr_results)
            
            # Размеры страницы
            dimensions = pdf_processor.get_page_dimensions(page_num)
            
            results[page_num] = {
                'text': text,
                'ocr_results': ocr_results,
                'dimensions': dimensions
            }
            
            # Сохраняем промежуточный результат
            page_output = output_dir / f"page_{page_num:03d}.json"
            with open(page_output, 'w', encoding='utf-8') as f:
                json.dump(results[page_num], f, ensure_ascii=False, indent=2)
        
        return results
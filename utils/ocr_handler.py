from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from .image_preprocessor import ImagePreprocessor
from config import Config

class OCRHandler:
    def __init__(self, lang='ru', use_gpu=False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,  # Используем переданный язык
            use_gpu=use_gpu,
            show_log=False,  # Отключаем логи для чистоты вывода
            det_model_dir=None,  # Используем встроенную модель
            rec_model_dir=None,  # Используем встроенную модель
            cls_model_dir=None   # Используем встроенную модель
        )
        
        # Инициализируем ImagePreprocessor если включён
        if Config.PREPROCESSING_ENABLED:
            self.preprocessor = ImagePreprocessor(
                target_dpi=Config.PREPROCESSING_TARGET_DPI,
                enable_contrast=Config.PREPROCESSING_ENABLE_CONTRAST,
                enable_sharpening=Config.PREPROCESSING_ENABLE_SHARPENING,
                enable_binarization=Config.PREPROCESSING_ENABLE_BINARIZATION,
                enable_denoising=Config.PREPROCESSING_ENABLE_DENOISING
            )
        else:
            self.preprocessor = None
    
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
        # Применяем предобработку если включена
        if self.preprocessor is not None:
            image = self.preprocessor.process(image)
        
        # Конвертируем PIL Image в numpy array
        img_array = np.array(image)
        
        # OCR
        result = self.ocr.ocr(img_array, cls=True)
        
        return self._parse_ocr_result(result)
    
    def extract_text_only(self, results: List[Dict]) -> str:
        """Извлекает только текст из результатов OCR"""
        return '\n'.join([item['text'] for item in results])
    
    def process_image_with_variants(self, image: Image.Image) -> Dict[str, List[Dict]]:
        """
        Обрабатывает изображение с разными вариантами предобработки для сравнения
        
        Returns:
            Dict[str, List[Dict]]: {
                'original': результаты без предобработки,
                'light': результаты с лёгкой предобработкой,
                'medium': результаты со средней предобработкой,
                'heavy': результаты с агрессивной предобработкой
            }
        """
        if self.preprocessor is None:
            # Если предобработка отключена, возвращаем только оригинал
            return {'original': self.process_image(image)}
        
        variants = {}
        
        # Оригинал без предобработки
        img_array = np.array(image)
        result = self.ocr.ocr(img_array, cls=True)
        variants['original'] = self._parse_ocr_result(result)
        
        # Получаем варианты предобработки
        processed_variants = self.preprocessor.process_with_multiple_variants(image)
        
        # Обрабатываем каждый вариант
        for variant_name, variant_image in processed_variants.items():
            if variant_name == 'original':
                continue  # Уже обработали выше
                
            img_array = np.array(variant_image)
            result = self.ocr.ocr(img_array, cls=True)
            variants[variant_name] = self._parse_ocr_result(result)
        
        return variants
    
    def _parse_ocr_result(self, result) -> List[Dict]:
        """Парсит результат OCR в стандартный формат"""
        if result is None or len(result) == 0:
            return []
        
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
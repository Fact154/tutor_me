import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
from tqdm import tqdm

class TesseractHandler:
    """
    Обработчик Tesseract OCR с различными режимами предобработки изображений
    """
    
    def __init__(self, lang='rus', tesseract_cmd=None):
        """
        Args:
            lang: Язык распознавания ('rus', 'eng', 'rus+eng')
            tesseract_cmd: Путь к tesseract.exe (если не в PATH)
        """
        self.lang = lang
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # Устанавливаем путь к tessdata
            import os
            tessdata_dir = Path(tesseract_cmd).parent / 'tessdata'
            if tessdata_dir.exists() and (tessdata_dir / 'rus.traineddata').exists():
                os.environ['TESSDATA_PREFIX'] = str(tessdata_dir)
            else:
                # Пробуем найти в стандартных местах
                possible_tessdata = [
                    Path('C:/Program Files/Tesseract-OCR/tessdata'),
                    Path('C:/Program Files (x86)/Tesseract-OCR/tessdata'),
                    Path('D:/Tesseract-OCR/tessdata'),
                ]
                for td in possible_tessdata:
                    if td.exists() and (td / 'rus.traineddata').exists():
                        os.environ['TESSDATA_PREFIX'] = str(td)
                        break
        
        # Проверяем доступность Tesseract
        try:
            pytesseract.get_tesseract_version()
            print(f"✓ Tesseract версия: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"✗ Ошибка: Tesseract не найден. Установите tesseract-ocr")
            raise e
    
    def preprocess_image_basic(self, image: Image.Image) -> Image.Image:
        """
        Базовая предобработка: конвертация в ч/б, увеличение контраста
        """
        # Конвертируем в grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Увеличиваем контраст
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Увеличиваем резкость
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    def preprocess_image_advanced(self, image: Image.Image) -> Image.Image:
        """
        Продвинутая предобработка с использованием OpenCV
        """
        # Конвертируем PIL в numpy array
        img_array = np.array(image)
        
        # Если цветное изображение, конвертируем в grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Удаление шумов
        img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        
        # Adaptive thresholding (лучше для неравномерного освещения)
        img_array = cv2.adaptiveThreshold(
            img_array, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Морфологические операции для улучшения текста
        kernel = np.ones((1, 1), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        
        # Конвертируем обратно в PIL
        return Image.fromarray(img_array)
    
    def preprocess_image_aggressive(self, image: Image.Image) -> Image.Image:
        """
        Агрессивная предобработка для очень плохого качества
        """
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Увеличиваем контраст с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        
        # Сильное удаление шумов
        img_array = cv2.fastNlMeansDenoising(img_array, None, 20, 7, 21)
        
        # Otsu thresholding
        _, img_array = cv2.threshold(
            img_array, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Дилатация для утолщения текста
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        return Image.fromarray(img_array)
    
    def deskew_image(self, image: Image.Image) -> Image.Image:
        """
        Исправление наклона изображения
        """
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Определяем угол наклона
        coords = np.column_stack(np.where(img_array > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Поворачиваем изображение
        (h, w) = img_array.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_array = cv2.warpAffine(
            img_array, 
            M, 
            (w, h),
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return Image.fromarray(img_array)
    
    def process_image(
        self, 
        image: Image.Image, 
        preprocessing: str = 'advanced',
        psm: int = 3,
        deskew: bool = False
    ) -> Dict:
        """
        Обрабатывает изображение и возвращает распознанный текст
        
        Args:
            image: PIL Image
            preprocessing: 'none', 'basic', 'advanced', 'aggressive'
            psm: Page Segmentation Mode (0-13)
                3 - Fully automatic page segmentation (по умолчанию)
                6 - Assume a single uniform block of text
                11 - Sparse text
            deskew: Исправлять наклон изображения
        
        Returns:
            {
                'text': str,
                'confidence': float,
                'details': List[Dict]  # Детальная информация по словам
            }
        """
        # Предобработка
        if preprocessing == 'basic':
            image = self.preprocess_image_basic(image)
        elif preprocessing == 'advanced':
            image = self.preprocess_image_advanced(image)
        elif preprocessing == 'aggressive':
            image = self.preprocess_image_aggressive(image)
        
        # Исправление наклона
        if deskew:
            image = self.deskew_image(image)
        
        # Настройки Tesseract
        custom_config = f'--psm {psm} --oem 3'
        
        # Извлекаем текст
        text = pytesseract.image_to_string(
            image, 
            lang=self.lang, 
            config=custom_config
        )
        
        # Получаем детальную информацию
        try:
            data = pytesseract.image_to_data(
                image, 
                lang=self.lang, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Вычисляем среднюю уверенность
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Собираем детальную информацию
            details = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:
                    details.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        }
                    })
        except:
            avg_confidence = 0
            details = []
        
        return {
            'text': text.strip(),
            'confidence': avg_confidence,
            'details': details
        }
    
    def ocr_image(
        self, 
        image: Image.Image, 
        preprocessing: str = 'advanced',
        psm: int = 3
    ) -> Dict:
        """
        Простой метод для OCR одного изображения (алиас для process_image)
        
        Args:
            image: PIL Image
            preprocessing: 'none', 'basic', 'advanced', 'aggressive'
            psm: Page Segmentation Mode (0-13)
        
        Returns:
            {'text': str, 'confidence': float, 'details': List[Dict]}
        """
        return self.process_image(image, preprocessing=preprocessing, psm=psm, deskew=True)
    
    def ocr_multimode(self, image: Image.Image) -> Dict:
        """
        OCR с несколькими режимами (алиас для process_image_multimode)
        """
        return self.process_image_multimode(image)
    
    def process_image_multimode(self, image: Image.Image) -> Dict:
        """
        Обрабатывает изображение несколькими режимами и выбирает лучший результат
        """
        results = []
        
        # Режим 1: Advanced preprocessing + PSM 3
        result1 = self.process_image(image, 'advanced', psm=3, deskew=True)
        results.append(('advanced_psm3', result1))
        
        # Режим 2: Aggressive preprocessing + PSM 6
        result2 = self.process_image(image, 'aggressive', psm=6, deskew=True)
        results.append(('aggressive_psm6', result2))
        
        # Режим 3: Basic preprocessing + PSM 3
        result3 = self.process_image(image, 'basic', psm=3, deskew=False)
        results.append(('basic_psm3', result3))
        
        # Выбираем результат с наибольшей уверенностью и длиной текста
        best_result = max(
            results, 
            key=lambda x: (x[1]['confidence'], len(x[1]['text']))
        )
        
        return {
            'text': best_result[1]['text'],
            'confidence': best_result[1]['confidence'],
            'details': best_result[1]['details'],
            'best_mode': best_result[0]
        }
    
    def process_pdf_pages(
        self, 
        pdf_processor, 
        page_range: Tuple[int, int],
        output_dir: Path,
        preprocessing: str = 'advanced',
        psm: int = 3,
        multimode: bool = False,
        dpi: int = 300
    ) -> Dict[int, Dict]:
        """
        Обрабатывает страницы PDF
        
        Args:
            pdf_processor: Объект PDFProcessor
            page_range: (start_page, end_page)
            output_dir: Директория для сохранения результатов
            preprocessing: Тип предобработки
            psm: Page Segmentation Mode
            multimode: Использовать множественные режимы и выбирать лучший
            dpi: Разрешение для извлечения изображений
        
        Returns:
            Dict[page_num, result_dict]
        """
        results = {}
        start_page, end_page = page_range
        
        for page_num in tqdm(range(start_page, end_page), desc="Tesseract OCR"):
            # Извлекаем изображение с высоким DPI
            image = pdf_processor.extract_page_as_image(page_num, dpi=dpi)
            
            # OCR
            if multimode:
                ocr_result = self.process_image_multimode(image)
            else:
                ocr_result = self.process_image(
                    image, 
                    preprocessing=preprocessing, 
                    psm=psm,
                    deskew=True
                )
            
            # Размеры страницы
            dimensions = pdf_processor.get_page_dimensions(page_num)
            
            results[page_num] = {
                'text': ocr_result['text'],
                'confidence': ocr_result['confidence'],
                'ocr_details': ocr_result.get('details', []),
                'dimensions': dimensions,
                'mode': ocr_result.get('best_mode', f'{preprocessing}_psm{psm}')
            }
            
            # Сохраняем промежуточный результат
            page_output = output_dir / f"page_{page_num:03d}.json"
            with open(page_output, 'w', encoding='utf-8') as f:
                json.dump(results[page_num], f, ensure_ascii=False, indent=2)
        
        return results
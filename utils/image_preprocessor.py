#!/usr/bin/env python3
"""
Модуль для предобработки изображений перед OCR
"""

from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Класс для улучшения качества изображений перед OCR
    """
    
    def __init__(
        self,
        target_dpi: int = 600,
        enable_contrast: bool = True,
        enable_sharpening: bool = True,
        enable_binarization: bool = True,
        enable_denoising: bool = True
    ):
        """
        Args:
            target_dpi: Целевое разрешение
            enable_contrast: Увеличивать контрастность
            enable_sharpening: Увеличивать резкость
            enable_binarization: Применять бинаризацию
            enable_denoising: Применять шумоподавление
        """
        self.target_dpi = target_dpi
        self.enable_contrast = enable_contrast
        self.enable_sharpening = enable_sharpening
        self.enable_binarization = enable_binarization
        self.enable_denoising = enable_denoising
    
    def process(self, image: Image.Image) -> Image.Image:
        """
        Полный пайплайн обработки изображения
        
        Args:
            image: Исходное изображение PIL
            
        Returns:
            Улучшенное изображение PIL
        """
        # 1. Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 2. Увеличиваем контрастность
        if self.enable_contrast:
            image = self._enhance_contrast(image)
        
        # 3. Увеличиваем резкость
        if self.enable_sharpening:
            image = self._sharpen_image(image)
        
        # 4. Конвертируем в numpy для OpenCV операций
        img_array = np.array(image)
        
        # 5. Шумоподавление
        if self.enable_denoising:
            img_array = self._denoise(img_array)
        
        # 6. Бинаризация (чёрно-белое)
        if self.enable_binarization:
            img_array = self._binarize(img_array)
        
        # 7. Конвертируем обратно в PIL
        processed_image = Image.fromarray(img_array)
        
        return processed_image
    
    def _enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Увеличивает контрастность изображения
        
        Args:
            image: Исходное изображение
            factor: Коэффициент контрастности (1.0 = без изменений, >1.0 = больше контраста)
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _sharpen_image(self, image: Image.Image, radius: int = 2, percent: int = 150) -> Image.Image:
        """
        Увеличивает резкость изображения
        
        Args:
            image: Исходное изображение
            radius: Радиус sharpening
            percent: Процент усиления (100 = без изменений)
        """
        # Используем UnsharpMask filter для более качественного результата
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
    
    def _denoise(self, img_array: np.ndarray) -> np.ndarray:
        """
        Удаляет шум с изображения
        
        Args:
            img_array: Изображение в формате numpy array (RGB)
        """
        # Используем fastNlMeansDenoisingColored для цветных изображений
        # h - сила фильтрации (10 - умеренная)
        # templateWindowSize - размер окна для сравнения
        # searchWindowSize - размер области поиска
        denoised = cv2.fastNlMeansDenoisingColored(
            img_array,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def _binarize(self, img_array: np.ndarray) -> np.ndarray:
        """
        Преобразует изображение в чёрно-белое (бинаризация)
        Использует адаптивную бинаризацию для лучших результатов
        
        Args:
            img_array: Изображение в формате numpy array
        """
        # Конвертируем в grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Применяем адаптивную бинаризацию
        # ADAPTIVE_THRESH_GAUSSIAN_C - использует взвешенную сумму окрестности
        # THRESH_BINARY - бинарное пороговое значение
        # 11 - размер окрестности для расчёта порога
        # 2 - константа, вычитаемая из среднего
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # Применяем морфологические операции для очистки
        binary = self._morphological_cleanup(binary)
        
        # Конвертируем обратно в RGB для совместимости
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        return binary_rgb
    
    def _morphological_cleanup(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Удаляет мелкие артефакты с бинарного изображения
        
        Args:
            binary_img: Бинарное изображение (grayscale)
        """
        # Создаём kernel для морфологических операций
        kernel = np.ones((2, 2), np.uint8)
        
        # Closing - заполняет мелкие дырки в буквах
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Opening - удаляет мелкий шум
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned
    
    def process_with_multiple_variants(self, image: Image.Image) -> dict:
        """
        Создаёт несколько вариантов обработки для сравнения
        
        Returns:
            dict с разными вариантами: {
                'original': исходное,
                'light': лёгкая обработка,
                'medium': средняя обработка,
                'heavy': агрессивная обработка
            }
        """
        variants = {}
        
        # Оригинал
        variants['original'] = image
        
        # Лёгкая обработка (только контраст + резкость)
        light_processor = ImagePreprocessor(
            enable_contrast=True,
            enable_sharpening=True,
            enable_binarization=False,
            enable_denoising=False
        )
        variants['light'] = light_processor.process(image)
        
        # Средняя обработка (контраст + резкость + шумоподавление)
        medium_processor = ImagePreprocessor(
            enable_contrast=True,
            enable_sharpening=True,
            enable_binarization=False,
            enable_denoising=True
        )
        variants['medium'] = medium_processor.process(image)
        
        # Агрессивная обработка (всё включено)
        heavy_processor = ImagePreprocessor(
            enable_contrast=True,
            enable_sharpening=True,
            enable_binarization=True,
            enable_denoising=True
        )
        variants['heavy'] = heavy_processor.process(image)
        
        return variants


# Утилита для быстрого тестирования
def test_preprocessing(input_image_path: str, output_dir: str = "./test_preprocessing"):
    """
    Тестирует разные варианты preprocessing на одном изображении
    
    Args:
        input_image_path: Путь к тестовому изображению
        output_dir: Папка для сохранения результатов
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Загружаем изображение
    image = Image.open(input_image_path)
    
    # Создаём preprocessor
    processor = ImagePreprocessor()
    
    # Получаем все варианты
    variants = processor.process_with_multiple_variants(image)
    
    # Сохраняем результаты
    for variant_name, variant_image in variants.items():
        output_file = output_path / f"{variant_name}.png"
        variant_image.save(output_file)
        print(f"Сохранён вариант: {output_file}")
    
    print(f"\n✓ Все варианты сохранены в {output_path}")


if __name__ == "__main__":
    # Пример использования
    print("ImagePreprocessor готов к использованию!")
    print("\nДля тестирования используйте:")
    print("  from utils.image_preprocessor import test_preprocessing")
    print("  test_preprocessing('path/to/image.png')")
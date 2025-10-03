#!/usr/bin/env python3
"""
Структурирование извлечённых данных в chunks
"""

import json
from pathlib import Path
from typing import List, Dict
from config import Config, TextbookMetadata
import re

class ChunkCreator:
    def __init__(self, subject: str):
        self.subject = subject
        self.chunk_id_counter = 0
    
    def create_chunk_id(self, metadata: Dict) -> str:
        """Генерирует уникальный ID для chunk'а"""
        self.chunk_id_counter += 1
        return (
            f"{self.subject}_{metadata['grade']}_"
            f"ch{metadata.get('chapter', 0)}_"
            f"p{metadata['page']}_"
            f"{self.chunk_id_counter}"
        )
    
    def extract_math_tasks(self, page_text: str, page_num: int) -> List[Dict]:
        """
        Извлекает математические задания со страницы
        
        Ищет паттерны типа:
        - "1. Заполните..."
        - "а) 5 см и 14 см;"
        """
        chunks = []
        
        # Паттерн для заданий
        task_pattern = r'(\d+)\.\s+(.*?)(?=\n\d+\.|$)'
        tasks = re.findall(task_pattern, page_text, re.DOTALL)
        
        for task_num, task_text in tasks:
            chunk = {
                'chunk_id': f"math_temp_{page_num}_{task_num}",
                'metadata': {
                    'page': page_num,
                    'task_number': int(task_num),
                    'content_type': 'task'
                },
                'content': {
                    'text': task_text.strip(),
                    'formulas': self._extract_formulas(task_text)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def extract_history_content(self, page_text: str, page_num: int) -> List[Dict]:
        """
        Извлекает исторический контент
        
        Ищет:
        - Даты
        - Имена исторических личностей
        - События
        """
        chunks = []
        
        # Разбиваем на параграфы
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        
        for idx, para in enumerate(paragraphs):
            # Извлекаем даты
            dates = re.findall(r'\b\d{1,4}\s*г\.?|\b\d{1,4}[-–]\d{1,4}\s*гг\.?', para)
            
            # Извлекаем имена (упрощённо - слова с заглавной буквы)
            names = re.findall(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+', para)
            
            chunk = {
                'chunk_id': f"history_temp_{page_num}_{idx}",
                'metadata': {
                    'page': page_num,
                    'content_type': 'text',
                    'dates': dates,
                    'historical_figures': list(set(names))
                },
                'content': {
                    'text': para
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _extract_formulas(self, text: str) -> List[str]:
        """Простое извлечение формул (можно улучшить)"""
        # Ищем математические выражения с =, +, -, *, /
        formula_pattern = r'[a-zа-я0-9\s\+\-\*\/\(\)]+\s*=\s*[a-zа-я0-9\s\+\-\*\/\(\)]+'
        formulas = re.findall(formula_pattern, text, re.IGNORECASE)
        return [f.strip() for f in formulas]

def structure_textbook(
    ocr_dir: Path,
    textbook_metadata: TextbookMetadata,
    output_dir: Path
):
    """
    Структурирует OCR результаты в chunks
    """
    print(f"Структурирование: {textbook_metadata.title}")
    
    chunk_creator = ChunkCreator(textbook_metadata.subject)
    all_chunks = []
    
    # Читаем все страницы
    page_files = sorted(ocr_dir.glob("page_*.json"))
    
    for page_file in page_files:
        with open(page_file, 'r', encoding='utf-8') as f:
            page_data = json.load(f)
        
        page_num = int(page_file.stem.split('_')[1])
        page_text = page_data['text']
        
        # Извлекаем chunks в зависимости от предмета
        if textbook_metadata.subject == "математика":
            chunks = chunk_creator.extract_math_tasks(page_text, page_num)
        elif textbook_metadata.subject == "история":
            chunks = chunk_creator.extract_history_content(page_text, page_num)
        else:
            continue
        
        # Добавляем общие метаданные
        for chunk in chunks:
            chunk['metadata'].update({
                'textbook_title': textbook_metadata.title,
                'grade': textbook_metadata.grade,
                'subject': textbook_metadata.subject,
                'author': textbook_metadata.author
            })
            
            # Генерируем финальный ID
            chunk['chunk_id'] = chunk_creator.create_chunk_id(chunk['metadata'])
        
        all_chunks.extend(chunks)
    
    # Сохраняем структурированные данные
    output_file = output_dir / f"{textbook_metadata.subject}_{textbook_metadata.grade}_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': textbook_metadata.model_dump(),
            'chunks': all_chunks,
            'total_chunks': len(all_chunks)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Создано {len(all_chunks)} chunks. Сохранено в: {output_file}")
    
    return all_chunks

if __name__ == "__main__":
    # Пример обработки математики
    ocr_dir = Config.OCR_DIR / "математика_5_v2"
    
    metadata = TextbookMetadata(
        title="Математика. 5 класс. Рабочая тетрадь. Часть 1",
        author="Ткачёва М.В.",
        year=2023,
        grade=5,
        subject="математика",
        part=1
    )
    
    structure_textbook(
        ocr_dir=ocr_dir,
        textbook_metadata=metadata,
        output_dir=Config.STRUCTURED_DIR
    )   
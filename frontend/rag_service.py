"""
Сервис для работы с RAG системой
"""
import json
import os
import sys
from typing import Dict, Any, List
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Импортируем функции из core.rag (общий модуль RAG)
try:
    from core import rag
    # Используем функции из модуля
    find_relevant_chunks = rag.find_relevant_chunks
    format_chunks_for_prompt = rag.format_chunks_for_prompt
    load_chunks = rag.load_chunks
    load_preprompt = rag.load_preprompt
except ImportError as e:
    # Если не можем импортировать, создаем заглушки
    print(f"[!] Предупреждение: не удалось импортировать core.rag: {e}")
    def find_relevant_chunks(*args, **kwargs):
        return []
    def format_chunks_for_prompt(*args, **kwargs):
        return ""
    def load_chunks(*args, **kwargs):
        return [], "Ошибка загрузки", {}
    def load_preprompt(*args, **kwargs):
        return ""

import ollama

TEXTBOOKS_CONFIG = Path(__file__).parent.parent / "textbooks_config.json"

def load_textbooks_config() -> Dict[str, Any]:
    """Загружает конфигурацию учебников"""
    try:
        from core import rag
        return rag.load_textbooks_config()
    except ImportError:
        # Fallback на прямую загрузку
        if TEXTBOOKS_CONFIG.exists():
            with open(TEXTBOOKS_CONFIG, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

def get_textbooks_list() -> List[Dict[str, Any]]:
    """Возвращает список доступных учебников"""
    config = load_textbooks_config()
    return config.get('textbooks', [])

def get_textbook_by_id(textbook_id: str) -> Dict[str, Any]:
    """Получает учебник по ID"""
    textbooks = get_textbooks_list()
    for tb in textbooks:
        if tb['id'] == textbook_id:
            return tb
    return None

def answer_question_simple(
    query: str,
    textbook_id: str,
    grade: int,
    model_name: str = "qwen3:8b",
    preprompt_mode: str = None
) -> Dict[str, Any]:
    """
    Упрощенная версия ответа на вопрос
    Использует существующую логику из core.rag
    """
    # Загружаем чанки для учебника
    chunks_data, error, doc_info = load_chunks(textbook_id)
    
    if not chunks_data:
        return {
            'answer': f"[!] Ошибка загрузки данных учебника: {error}",
            'sources': [],
            'error': error
        }
    
    # Загружаем предпромт (с поддержкой режимов)
    preprompt = load_preprompt(textbook_id, preprompt_mode)
    
    # Находим релевантные чанки
    relevant_chunks = find_relevant_chunks(query, chunks_data, top_k=3)
    
    if not relevant_chunks:
        return {
            'answer': "[!] Не найдено релевантных чанков для вашего вопроса. Попробуйте переформулировать вопрос или указать номер задания/страницу.",
            'sources': [],
            'error': 'No relevant chunks found'
        }
    
    # Форматируем контекст
    context = format_chunks_for_prompt(relevant_chunks)
    
    # Формируем полный промпт
    full_prompt = f"""{preprompt}

=== КОНТЕКСТ ИЗ УЧЕБНИКА ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ИНСТРУКЦИЯ ===
Используй информацию из контекста выше, чтобы ответить на вопрос пользователя. 
Следуй формату ответа, указанному в предпромпте.
"""
    
    try:
        # Отправляем запрос в Ollama с оптимизированными параметрами
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"Ты помощник-репетитор для школьника {grade} класса. Ты помогаешь решать задачи из учебника, используя предоставленный контекст."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            options={
                'temperature': 0.3,  # Низкая температура для точности и следования инструкциям
                'top_p': 0.9,        # Ограничиваем вариативность
                'top_k': 40,         # Уменьшаем случайность
                'repeat_penalty': 1.1  # Избегаем повторений
            }
        )
        
        answer = response['message']['content']
        
        # Формируем источники
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'page': chunk.get('pages', 'Не указана'),
                'section': chunk.get('section', 'Не указан'),
                'topic': chunk.get('topic', 'Не указана')
            })
        
        return {
            'answer': answer,
            'sources': sources,
            'error': None
        }
    
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            return {
                'answer': f"[X] Модель '{model_name}' не найдена. Установите модель командой: `ollama pull {model_name}`",
                'sources': [],
                'error': error_msg
            }
        return {
            'answer': f"[X] Ошибка при обращении к LLM: {error_msg}",
            'sources': [],
            'error': error_msg
        }


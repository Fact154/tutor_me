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

def generate_practice_task(
    original_query: str,
    original_solution: str,
    textbook_id: str,
    grade: int,
    complexity_level: int = 1,
    model_name: str = "qwen3:8b",
    preprompt_mode: str = None
) -> Dict[str, Any]:
    """
    Генерирует похожее практическое задание на основе предыдущего решения
    """
    preprompt = load_preprompt(textbook_id, preprompt_mode)
    
    generation_prompt = f"""{preprompt}

=== ПРЕДЫДУЩЕЕ ЗАДАНИЕ ===
{original_query}

=== РЕШЕНИЕ ПРЕДЫДУЩЕГО ЗАДАНИЯ ===
{original_solution}

=== ИНСТРУКЦИЯ ===
Сгенерируй похожее задание с другими числами, но той же структурой и логикой.
Уровень сложности: {complexity_level} (1-простое, 5-очень сложное)

Формат ответа (строго соблюдай):
**ЗАДАНИЕ:**
[Текст задания]

**ПРАВИЛЬНЫЙ ОТВЕТ:**
[Полное решение с ответом]

**ПОДСКАЗКА:**
[Краткая подсказка, какой метод использовать]
"""
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"Ты репетитор по математике для {grade} класса. Генерируешь похожие задания для практики."
                },
                {
                    "role": "user",
                    "content": generation_prompt
                }
            ],
            options={
                'temperature': 0.5,
                'top_p': 0.9,
                'top_k': 40
            }
        )
        
        generated_text = response['message']['content']
        
        # Парсим ответ
        import re
        task_match = re.search(r'\*\*ЗАДАНИЕ:\*\*\s*(.*?)(?=\*\*ПРАВИЛЬНЫЙ ОТВЕТ:\*\*|\*\*ПОДСКАЗКА:\*\*|$)', generated_text, re.DOTALL)
        answer_match = re.search(r'\*\*ПРАВИЛЬНЫЙ ОТВЕТ:\*\*\s*(.*?)(?=\*\*ПОДСКАЗКА:\*\*|$)', generated_text, re.DOTALL)
        hint_match = re.search(r'\*\*ПОДСКАЗКА:\*\*\s*(.*?)$', generated_text, re.DOTALL)
        
        task_text = task_match.group(1).strip() if task_match else generated_text.split('\n\n')[0].strip()
        correct_answer = answer_match.group(1).strip() if answer_match else ""
        hint = hint_match.group(1).strip() if hint_match else ""
        
        return {
            'task_text': task_text,
            'correct_answer': correct_answer,
            'hint': hint,
            'full_text': generated_text,
            'error': None
        }
    
    except Exception as e:
        return {
            'task_text': "",
            'correct_answer': "",
            'hint': "",
            'full_text': f"Ошибка при генерации задания: {str(e)}",
            'error': str(e)
        }

def evaluate_practice_answer(
    task_text: str,
    correct_answer: str,
    student_answer: str,
    textbook_id: str,
    grade: int,
    model_name: str = "qwen3:8b",
    preprompt_mode: str = None
) -> Dict[str, Any]:
    """
    Оценивает ответ ученика на практическое задание
    """
    preprompt = load_preprompt(textbook_id, preprompt_mode)
    
    evaluation_prompt = f"""{preprompt}

=== ЗАДАНИЕ ===
{task_text}

=== ПРАВИЛЬНЫЙ ОТВЕТ ===
{correct_answer}

=== ОТВЕТ УЧЕНИКА ===
{student_answer}

=== ИНСТРУКЦИЯ ===
Оцени ответ ученика строго, но честно по шкале 1-5:
- 5: Полностью правильное решение с объяснением
- 4: Правильный ответ, но есть мелкие недочеты
- 3: Частично правильное решение
- 2: Неправильный ответ, но есть попытка решения
- 1: Неправильный ответ без попытки решения

В ответе укажи:
**ОЦЕНКА: [1-5]**

**Что правильно:** [Что ученик сделал верно]

**Ошибки:** [Что неправильно, конкретно и без прикрас]

**Как нужно было решить:** [Краткое правильное решение]

**Что повторить:** [Какие темы нужно повторить]
"""
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"Ты строгий, но справедливый учитель математики для {grade} класса. Оценивай ответы честно, без завышения оценок."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ],
            options={
                'temperature': 0.2,  # Очень низкая температура для строгой оценки
                'top_p': 0.8,
                'top_k': 30
            }
        )
        
        evaluation = response['message']['content']
        
        # Извлекаем оценку из текста
        grade = None
        if "ОЦЕНКА:" in evaluation:
            import re
            match = re.search(r'ОЦЕНКА:\s*(\d+)', evaluation)
            if match:
                grade = int(match.group(1))
                # Ограничиваем диапазон
                if grade < 1:
                    grade = 1
                elif grade > 5:
                    grade = 5
        
        return {
            'evaluation': evaluation,
            'grade': grade,
            'feedback': evaluation,
            'error': None
        }
    
    except Exception as e:
        return {
            'evaluation': f"Ошибка при оценке: {str(e)}",
            'grade': None,
            'feedback': f"Ошибка при оценке: {str(e)}",
            'error': str(e)
        }


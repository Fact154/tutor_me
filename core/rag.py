"""
Модуль для работы с RAG системой
Содержит функции для загрузки учебников, поиска релевантных чанков и форматирования
"""
import json
import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Конфигурационный файл с учебниками
TEXTBOOKS_CONFIG = Path(__file__).parent.parent / "textbooks_config.json"

# Кэш конфигурации (опционально, для производительности)
_textbooks_config_cache = None


def load_textbooks_config() -> Dict[str, Any]:
    """Загружает конфигурацию учебников"""
    global _textbooks_config_cache
    
    if _textbooks_config_cache is not None:
        return _textbooks_config_cache
    
    if TEXTBOOKS_CONFIG.exists():
        try:
            with open(TEXTBOOKS_CONFIG, 'r', encoding='utf-8') as f:
                _textbooks_config_cache = json.load(f)
                return _textbooks_config_cache
        except Exception as e:
            print(f"Ошибка загрузки конфигурации учебников: {e}")
            return {}
    else:
        print(f"Файл конфигурации {TEXTBOOKS_CONFIG} не найден. Создайте его для работы с несколькими учебниками.")
        return {}


def get_textbook_list() -> List[Tuple[str, str]]:
    """Возвращает список учебников в формате (id, название) для dropdown"""
    textbooks_config = load_textbooks_config()
    if not textbooks_config or 'textbooks' not in textbooks_config:
        return []
    
    return [(tb['id'], tb['name']) for tb in textbooks_config['textbooks']]


def load_chunks(textbook_id: str = None) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Загружает чанки для указанного учебника. Возвращает (чанки, сообщение_об_ошибке, информация_о_документе)"""
    textbooks_config = load_textbooks_config()
    
    all_chunks = []
    error_msg = ""
    doc_info = {}
    
    # Если учебник не указан, используем первый из конфигурации
    if not textbook_id:
        if textbooks_config and 'textbooks' in textbooks_config and textbooks_config['textbooks']:
            textbook_id = textbooks_config['textbooks'][0]['id']
        else:
            # Fallback на старый способ загрузки
            return load_chunks_legacy()
    
    # Находим конфигурацию учебника
    textbook = None
    if textbooks_config and 'textbooks' in textbooks_config:
        for tb in textbooks_config['textbooks']:
            if tb['id'] == textbook_id:
                textbook = tb
                break
    
    if not textbook:
        error_msg = f"Учебник с ID '{textbook_id}' не найден в конфигурации."
        return [], error_msg, {}
    
    try:
        # Загружаем чанки из всех указанных файлов
        chunks_files = textbook.get('chunks_files', [])
        if not chunks_files:
            error_msg = f"Для учебника '{textbook['name']}' не указаны файлы с чанками."
            return [], error_msg, {}
        
        for chunks_file in chunks_files:
            chunks_path = Path(chunks_file)
            if not chunks_path.is_absolute():
                # Относительный путь от корня проекта
                chunks_path = Path(__file__).parent.parent / chunks_file
            
            if chunks_path.exists():
                try:
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'chunks' in data:
                            all_chunks.extend(data['chunks'])
                        # Сохраняем информацию о документе из первого файла
                        if 'document_info' in data and not doc_info:
                            doc_info = data['document_info']
                except Exception as e:
                    error_msg += f"Ошибка загрузки {chunks_file}: {str(e)}. "
            else:
                error_msg += f"Файл {chunks_file} не найден. "
        
        # Если doc_info пустой, используем информацию из конфигурации
        if not doc_info:
            doc_info = {
                'title': textbook.get('name', 'Неизвестно'),
                'author': textbook.get('author', ''),
                'publisher': textbook.get('publisher', ''),
                'year': textbook.get('year', '')
            }
        
        print(f"Загружено чанков для '{textbook['name']}': {len(all_chunks)}")
        if error_msg:
            print(f"Предупреждения: {error_msg}")
    except Exception as e:
        error_msg = f"Критическая ошибка при загрузке чанков: {str(e)}"
        print(error_msg)
    
    return all_chunks, error_msg.strip(), doc_info


def load_chunks_legacy() -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Старый способ загрузки чанков (для обратной совместимости)"""
    CHUNKS1_PATH = Path(__file__).parent.parent / "promt" / "chancs1.txt"
    CHUNKS2_PATH = Path(__file__).parent.parent / "promt" / "chancs2.txt"
    
    all_chunks = []
    error_msg = ""
    doc_info = {}
    
    for chunks_file in [CHUNKS1_PATH, CHUNKS2_PATH]:
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chunks' in data:
                        all_chunks.extend(data['chunks'])
                    if 'document_info' in data and not doc_info:
                        doc_info = data['document_info']
            except Exception as e:
                error_msg += f"Ошибка загрузки {chunks_file}: {str(e)}. "
    
    return all_chunks, error_msg.strip(), doc_info


def get_preprompt_modes(textbook_id: str = None) -> Dict[str, str]:
    """Возвращает доступные режимы предпромптов для учебника"""
    textbooks_config = load_textbooks_config()
    
    if not textbook_id:
        if textbooks_config and 'textbooks' in textbooks_config and textbooks_config['textbooks']:
            textbook_id = textbooks_config['textbooks'][0]['id']
    
    if textbook_id and textbooks_config and 'textbooks' in textbooks_config:
        for tb in textbooks_config['textbooks']:
            if tb['id'] == textbook_id:
                # Новый формат с несколькими режимами
                if 'preprompt_modes' in tb:
                    return tb['preprompt_modes']
                # Старый формат с одним файлом
                elif 'preprompt_file' in tb:
                    return {"Стандартный": tb['preprompt_file']}
                break
    
    # Fallback
    return {"Стандартный": "promt/предпромт.txt"}


def get_default_mode(textbook_id: str = None) -> str:
    """Возвращает режим по умолчанию для учебника"""
    textbooks_config = load_textbooks_config()
    
    if not textbook_id:
        if textbooks_config and 'textbooks' in textbooks_config and textbooks_config['textbooks']:
            textbook_id = textbooks_config['textbooks'][0]['id']
    
    if textbook_id and textbooks_config and 'textbooks' in textbooks_config:
        for tb in textbooks_config['textbooks']:
            if tb['id'] == textbook_id:
                return tb.get('default_mode', 'Стандартный')
    
    return 'Стандартный'


def load_preprompt(textbook_id: str = None, mode: str = None) -> str:
    """Загружает предпромт для указанного учебника и режима"""
    textbooks_config = load_textbooks_config()
    
    # Если учебник не указан, используем первый из конфигурации
    if not textbook_id:
        if textbooks_config and 'textbooks' in textbooks_config and textbooks_config['textbooks']:
            textbook_id = textbooks_config['textbooks'][0]['id']
        else:
            # Fallback на старый путь
            preprompt_path = Path(__file__).parent.parent / "promt" / "предпромт.txt"
            if preprompt_path.exists():
                with open(preprompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
    
    # Если режим не указан, используем режим по умолчанию
    if not mode:
        mode = get_default_mode(textbook_id)
    
    # Получаем доступные режимы
    modes = get_preprompt_modes(textbook_id)
    
    # Находим файл для выбранного режима
    preprompt_file = modes.get(mode, modes.get('Стандартный', 'promt/предпромт.txt'))
    
    preprompt_path = Path(preprompt_file)
    if not preprompt_path.is_absolute():
        preprompt_path = Path(__file__).parent.parent / preprompt_file
    
    if preprompt_path.exists():
        try:
            with open(preprompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Ошибка загрузки предпромта: {e}")
    
    # Fallback на старый путь
    preprompt_path = Path(__file__).parent.parent / "promt" / "предпромт.txt"
    if preprompt_path.exists():
        with open(preprompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Находит наиболее релевантные чанки для запроса с улучшенным поиском"""
    # Нормализуем запрос: приводим к нижнему регистру и удаляем знаки препинания
    query_normalized = re.sub(r'[^\w\s]', ' ', query.lower())
    query_words = set([w for w in query_normalized.split() if len(w) > 2])  # Игнорируем короткие слова
    
    # Если запрос слишком короткий, используем все слова
    if not query_words:
        query_words = set(query_normalized.split())
    
    scored_chunks = []
    
    for chunk in chunks:
        score = 0
        
        # Создаём объединённый текст из всех полей чанка для более полного поиска
        chunk_text_parts = []
        
        # Добавляем раздел
        if 'section' in chunk:
            chunk_text_parts.append(chunk['section'].lower())
        
        # Добавляем тему (высокий приоритет)
        if 'topic' in chunk:
            topic_text = chunk['topic'].lower()
            chunk_text_parts.append(topic_text)
            # Проверяем совпадения в теме отдельно (более высокий вес)
            for word in query_words:
                if word in topic_text:
                    score += 4  # Увеличиваем вес темы
                # Проверяем частичные совпадения
                if any(word in part or part in word for part in topic_text.split()):
                    score += 2
        
        # Добавляем ключевые слова
        if 'keywords' in chunk:
            keywords_text = ' '.join([kw.lower() for kw in chunk['keywords']])
            chunk_text_parts.append(keywords_text)
            for word in query_words:
                if any(word in kw.lower() or kw.lower() in word for kw in chunk['keywords']):
                    score += 3  # Увеличиваем вес ключевых слов
        
        # Добавляем содержание
        if 'content' in chunk:
            content_lower = chunk['content'].lower()
            chunk_text_parts.append(content_lower)
            # Подсчитываем количество совпадений в содержании
            matches = sum(1 for word in query_words if word in content_lower)
            score += matches  # Бонус за каждое совпадение
        
        # Проверяем номер задания (самый высокий приоритет)
        if 'exercises' in chunk:
            numbers = re.findall(r'\d+', query)
            for num in numbers:
                if int(num) in chunk['exercises']:
                    score += 10  # Очень высокий приоритет для номеров заданий
        
        # Проверяем страницы
        if 'pages' in chunk:
            page_numbers = re.findall(r'\d+', chunk['pages'])
            query_numbers = re.findall(r'\d+', query)
            for q_num in query_numbers:
                if any(q_num == p_num for p_num in page_numbers):
                    score += 6
        
        # Дополнительная проверка: ищем совпадения в объединённом тексте
        # Это помогает находить чанки даже если слова разбросаны по разным полям
        if chunk_text_parts:
            combined_text = ' '.join(chunk_text_parts)
            # Проверяем, сколько слов из запроса встречается в чанке
            found_words = sum(1 for word in query_words if word in combined_text)
            if found_words > 0:
                # Бонус за процент совпадения слов
                match_ratio = found_words / len(query_words) if query_words else 0
                score += int(match_ratio * 5)  # Бонус до 5 баллов за полное совпадение
        
        # Добавляем чанк, если есть хотя бы минимальный релевантность
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Сортируем по релевантности
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Если нашли чанки, возвращаем top_k
    if scored_chunks:
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    # Если ничего не нашли, пробуем более мягкий поиск (частичные совпадения)
    # Это помогает, когда запрос использует другие формулировки
    if not scored_chunks:
        for chunk in chunks:
            score = 0
            combined_text = ''
            
            # Собираем весь текст чанка
            if 'section' in chunk:
                combined_text += chunk['section'].lower() + ' '
            if 'topic' in chunk:
                combined_text += chunk['topic'].lower() + ' '
            if 'keywords' in chunk:
                combined_text += ' '.join([kw.lower() for kw in chunk['keywords']]) + ' '
            if 'content' in chunk:
                combined_text += chunk['content'].lower()[:500]  # Первые 500 символов
            
            # Проверяем частичные совпадения
            query_clean = query_normalized.replace(' ', '')
            for word in query_words:
                if len(word) > 3 and word in combined_text:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    return []


def format_chunks_for_prompt(chunks: List[Dict]) -> str:
    """Форматирует чанки для включения в промпт"""
    formatted = []
    for chunk in chunks:
        chunk_text = f"=== Чанк {chunk.get('chunk_id', '?')} ===\n"
        chunk_text += f"Раздел: {chunk.get('section', 'Не указан')}\n"
        chunk_text += f"Тема: {chunk.get('topic', 'Не указана')}\n"
        chunk_text += f"Страницы: {chunk.get('pages', 'Не указаны')}\n"
        if 'content' in chunk:
            chunk_text += f"Содержание:\n{chunk['content']}\n"
        formatted.append(chunk_text)
    return "\n\n".join(formatted)


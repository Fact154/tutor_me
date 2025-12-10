import json
import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Tuple
import ollama
import gradio as gr

# Настройка логирования для подавления asyncio ошибок
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Подавляем предупреждения о разрыве соединений
def handle_exception(exc_type, exc_value, exc_traceback):
    """Обработчик необработанных исключений"""
    if issubclass(exc_type, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        # Игнорируем ошибки разрыва соединения
        return
    # Для остальных ошибок используем стандартную обработку
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

# Импортируем функции RAG из общего модуля
from core import rag
from core.validator import QueryValidator

# Конфигурационный файл с учебниками (для совместимости)
TEXTBOOKS_CONFIG = "textbooks_config.json"

# Глобальные переменные (для состояния Gradio интерфейса)
chunks_data = []
preprompt = ""
chunks_error = ""
current_textbook_id = None
textbooks_config = {}
doc_info = {}

def load_textbooks_config() -> Dict[str, Any]:
    """Загружает конфигурацию учебников (обертка над core.rag)"""
    global textbooks_config
    textbooks_config = rag.load_textbooks_config()
    return textbooks_config

def get_textbook_list() -> List[Tuple[str, str]]:
    """Возвращает список учебников в формате (id, название) для dropdown"""
    return rag.get_textbook_list()

def load_chunks(textbook_id: str = None) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Загружает чанки для указанного учебника. Возвращает (чанки, сообщение_об_ошибке, информация_о_документе)"""
    global current_textbook_id
    
    # Используем функцию из core.rag
    result = rag.load_chunks(textbook_id)
    all_chunks, error_msg, doc_info = result
    
    # Обновляем глобальную переменную для состояния интерфейса
    if textbook_id:
        current_textbook_id = textbook_id
    elif all_chunks:
        # Если учебник не указан, но чанки загружены, устанавливаем первый из конфигурации
        config = rag.load_textbooks_config()
        if config and 'textbooks' in config and config['textbooks']:
            current_textbook_id = config['textbooks'][0]['id']
    
    return result

def get_preprompt_modes(textbook_id: str = None) -> Dict[str, str]:
    """Возвращает доступные режимы предпромптов для учебника"""
    if not textbook_id:
        textbook_id = current_textbook_id
        if not textbook_id:
            config = rag.load_textbooks_config()
            if config and 'textbooks' in config and config['textbooks']:
                textbook_id = config['textbooks'][0]['id']
    return rag.get_preprompt_modes(textbook_id)


def get_default_mode(textbook_id: str = None) -> str:
    """Возвращает режим по умолчанию для учебника"""
    if not textbook_id:
        textbook_id = current_textbook_id
        if not textbook_id:
            config = rag.load_textbooks_config()
            if config and 'textbooks' in config and config['textbooks']:
                textbook_id = config['textbooks'][0]['id']
    return rag.get_default_mode(textbook_id)


def load_preprompt(textbook_id: str = None, mode: str = None) -> str:
    """Загружает предпромт для указанного учебника и режима"""
    # Если учебник не указан, используем текущий или первый из конфигурации
    if not textbook_id:
        textbook_id = current_textbook_id
        if not textbook_id:
            config = rag.load_textbooks_config()
            if config and 'textbooks' in config and config['textbooks']:
                textbook_id = config['textbooks'][0]['id']
    
    # Используем функцию из core.rag
    return rag.load_preprompt(textbook_id, mode)

# Используем функции из core.rag
find_relevant_chunks = rag.find_relevant_chunks
format_chunks_for_prompt = rag.format_chunks_for_prompt

def check_ollama_connection() -> Tuple[bool, str]:
    """Проверяет подключение к Ollama"""
    try:
        models = ollama.list()
        return True, "Подключено"
    except Exception as e:
        return False, f"Ошибка подключения: {str(e)}"

def get_available_models() -> List[str]:
    """Получает список доступных моделей из Ollama"""
    try:
        models_response = ollama.list()
        # Проверяем разные форматы ответа
        if isinstance(models_response, dict):
            models_list = models_response.get('models', [])
        elif isinstance(models_response, list):
            models_list = models_response
        else:
            models_list = []
        
        # Извлекаем имена моделей
        model_names = []
        for model in models_list:
            if isinstance(model, dict):
                # Пробуем разные ключи для имени модели
                name = model.get('name') or model.get('model') or str(model)
                if name:
                    model_names.append(name)
            elif isinstance(model, str):
                model_names.append(model)
        
        if not model_names:
            print("⚠️ Модели не найдены в Ollama. Используется дефолтная модель.")
            return ["qwen3:8b"]  # Дефолтная модель
        return model_names
    except Exception as e:
        print(f"❌ Ошибка при получении списка моделей: {e}")
        return ["qwen3:8b"]  # Возвращаем дефолтную модель при ошибке

def query_llm(user_query: str, model_name: str = "qwen3:8b") -> str:
    """Отправляет запрос в LLM через Ollama"""
    global chunks_data, preprompt
    
    # Проверяем подключение к Ollama
    is_connected, connection_msg = check_ollama_connection()
    if not is_connected:
        return f"❌ {connection_msg}\n\nУбедитесь, что Ollama запущен. Запустите: `ollama serve`"
    
    # Находим релевантные чанки
    relevant_chunks = find_relevant_chunks(user_query, chunks_data, top_k=3)
    
    if not relevant_chunks:
        return "⚠️ Не найдено релевантных чанков для вашего вопроса. Попробуйте переформулировать вопрос или указать номер задания/страницу."
    
    # Форматируем контекст
    context = format_chunks_for_prompt(relevant_chunks)
    
    # Формируем полный промпт
    full_prompt = f"""{preprompt}

=== КОНТЕКСТ ИЗ УЧЕБНИКА ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{user_query}

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
                    "content": "Ты помощник-репетитор по математике для 5 класса. Ты помогаешь решать задачи из учебника, используя предоставленный контекст."
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
        
        return response['message']['content']
    
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            return f"❌ Модель '{model_name}' не найдена.\n\nУстановите модель командой:\n```bash\nollama pull {model_name}\n```\n\nИли выберите другую модель из списка."
        return f"❌ Ошибка при обращении к LLM: {error_msg}\n\nУбедитесь, что:\n1. Ollama запущен (`ollama serve`)\n2. Модель {model_name} установлена (`ollama pull {model_name}`)"

def chat_interface(message: str, history: List, model_name: str, preprompt_mode: str = None) -> tuple:
    """Интерфейс для чата"""
    global preprompt
    
    # Инициализируем history, если она None или не является списком
    if history is None:
        history = []
    if not isinstance(history, list):
        history = []
    
    # Проверяем, что сообщение не пустое
    if not message or not message.strip():
        return history, ""
    
    try:
        # Загружаем предпромпт для выбранного режима
        if preprompt_mode:
            preprompt = load_preprompt(current_textbook_id, preprompt_mode)
        
        # Получаем ответ от LLM
        response = query_llm(message, model_name)
        
        # Добавляем в историю
        history.append([message, response])
        
        return history, ""
    except Exception as e:
        # В случае ошибки добавляем сообщение об ошибке в историю
        error_response = f"❌ Произошла ошибка: {str(e)}"
        history.append([message, error_response])
        return history, ""

def main():
    """Основная функция для запуска приложения"""
    global chunks_data, preprompt, chunks_error, textbooks_config, doc_info
    
    print("Загрузка конфигурации учебников...")
    textbooks_config = load_textbooks_config()
    
    print("Загрузка данных...")
    # Загружаем данные для первого учебника по умолчанию
    chunks_data, chunks_error, doc_info = load_chunks()
    preprompt = load_preprompt()
    
    if not chunks_data:
        print("ВНИМАНИЕ: Чанки не загружены! Проверьте пути к файлам.")
        if chunks_error:
            print(f"Ошибки: {chunks_error}")
    
    if not preprompt:
        print("ВНИМАНИЕ: Предпромт не загружен! Проверьте путь к файлу.")
    
    # Создаем Gradio интерфейс
    with gr.Blocks(title="Репетитор по математике 5 класс", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📚 Репетитор по математике 5 класс
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Выбор учебника
                textbook_list = get_textbook_list()
                # Преобразуем список кортежей в формат для Gradio: список строк (названий)
                textbook_choices = [name for _, name in textbook_list] if textbook_list else []
                textbook_values = [id_val for id_val, _ in textbook_list] if textbook_list else []
                default_textbook_idx = 0 if textbook_list else None
                default_textbook_value = textbook_values[0] if textbook_values else None
                
                textbook_dropdown = gr.Dropdown(
                    choices=textbook_choices,
                    value=textbook_choices[0] if textbook_choices else None,
                    label="📖 Учебник",
                    info="Выберите учебник для работы",
                    interactive=True
                )
                
                # Получаем список доступных моделей из Ollama
                try:
                    available_models = get_available_models()
                    # Дефолтная модель - 8b, если доступна, иначе первая из списка
                    default_model = "qwen3:8b"
                    if "qwen3:8b" not in available_models and available_models:
                        default_model = available_models[0]
                except Exception as e:
                    print(f"Ошибка при загрузке моделей: {e}")
                    available_models = ["qwen3:8b"]
                    default_model = "qwen3:8b"
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="🤖 Модель Ollama",
                    info="Выберите модель для использования"
                )
                
                refresh_models_btn = gr.Button("🔄 Обновить список моделей", variant="secondary", size="sm")
                
                def refresh_models():
                    try:
                        new_models = get_available_models()
                        # Дефолтная модель - 8b, если доступна, иначе первая из списка
                        default = "qwen3:8b"
                        if "qwen3:8b" not in new_models:
                            if new_models:
                                default = new_models[0]
                            else:
                                default = "qwen3:8b"
                        return gr.Dropdown(choices=new_models, value=default)
                    except Exception as e:
                        print(f"Ошибка при обновлении моделей: {e}")
                        return gr.Dropdown(choices=["qwen3:8b"], value="qwen3:8b")
                
                refresh_models_btn.click(refresh_models, None, model_dropdown)
                
                # Выбор режима репетитора
                preprompt_modes = get_preprompt_modes(default_textbook_value)
                mode_choices = list(preprompt_modes.keys())
                default_mode = get_default_mode(default_textbook_value)
                
                mode_dropdown = gr.Dropdown(
                    choices=mode_choices,
                    value=default_mode,
                    label="🎭 Режим репетитора",
                    info="Выберите стиль ответов"
                )
                
                gr.Markdown("""
                **Режимы:**
                - **Стандартный** — сбалансированные ответы
                - **Краткий** — быстрые ответы без деталей
                - **Подробный** — детальные объяснения + задание для закрепления
                """)
                
                gr.Markdown("### 📊 Статистика")
                
                # Формируем статус чанков с безопасной обработкой
                try:
                    chunks_count = len(chunks_data) if chunks_data else 0
                    chunks_status = f"**{chunks_count}**"
                    if chunks_error:
                        error_display = chunks_error[:50] + "..." if len(chunks_error) > 50 else chunks_error
                        chunks_status += f" ⚠️ ({error_display})"
                except Exception as e:
                    chunks_status = f"**0** ⚠️ (Ошибка: {str(e)})"
                
                # Безопасная проверка подключения Ollama
                try:
                    is_connected, conn_msg = check_ollama_connection()
                    ollama_status = "✓ Подключен" if is_connected else f"✗ {conn_msg}"
                except Exception as e:
                    ollama_status = f"✗ Ошибка: {str(e)}"
                
                # Получаем название текущего учебника
                current_textbook_name = "Неизвестно"
                if doc_info and 'title' in doc_info:
                    current_textbook_name = doc_info['title']
                elif default_textbook_value:
                    for tb in textbooks_config.get('textbooks', []):
                        if tb['id'] == default_textbook_value:
                            current_textbook_name = tb['name']
                            break
                
                stats_text = gr.Markdown(f"""
                - Учебник: **{current_textbook_name}**
                - Загружено чанков: {chunks_status}
                - Предпромт: **{"✓ Загружен" if preprompt else "✗ Не загружен"}**
                - Ollama: **{ollama_status}**
                """)
                
                # Функция для обновления данных при смене учебника
                def change_textbook(textbook_name):
                    """Обновляет данные при смене учебника"""
                    global chunks_data, preprompt, chunks_error, doc_info
                    
                    if not textbook_name:
                        return gr.Markdown("Выберите учебник"), gr.Chatbot(value=[])
                    
                    # Находим ID учебника по названию
                    textbook_id = None
                    for id_val, name in textbook_list:
                        if name == textbook_name:
                            textbook_id = id_val
                            break
                    
                    if not textbook_id:
                        return gr.Markdown("❌ Учебник не найден"), gr.Chatbot(value=[])
                    
                    try:
                        # Загружаем данные для выбранного учебника
                        new_chunks, new_error, new_doc_info = load_chunks(textbook_id)
                        new_preprompt = load_preprompt(textbook_id)
                        
                        chunks_data = new_chunks
                        preprompt = new_preprompt
                        chunks_error = new_error
                        doc_info = new_doc_info  # Обновляем информацию о документе
                        
                        # Формируем статус
                        chunks_count = len(chunks_data) if chunks_data else 0
                        chunks_status = f"**{chunks_count}**"
                        if chunks_error:
                            error_display = chunks_error[:50] + "..." if len(chunks_error) > 50 else chunks_error
                            chunks_status += f" ⚠️ ({error_display})"
                        
                        # Получаем название учебника
                        textbook_display_name = doc_info.get('title', textbook_name) if doc_info else textbook_name
                        
                        stats = f"""
                        - Учебник: **{textbook_display_name}**
                        - Загружено чанков: {chunks_status}
                        - Предпромт: **{"✓ Загружен" if preprompt else "✗ Не загружен"}**
                        - Ollama: **{"✓ Подключен" if check_ollama_connection()[0] else "✗ Не подключен"}**
                        """
                        
                        return gr.Markdown(stats), gr.Chatbot(value=[])
                    except Exception as e:
                        error_msg = f"Ошибка при загрузке учебника: {str(e)}"
                        return gr.Markdown(f"❌ {error_msg}"), gr.Chatbot(value=[])
                
                refresh_stats_btn = gr.Button("🔄 Обновить статистику", variant="secondary", size="sm")
                
                def refresh_stats():
                    global chunks_data, chunks_error, doc_info, current_textbook_id
                    try:
                        # Перезагружаем чанки для текущего учебника
                        textbook_id = current_textbook_id
                        if not textbook_id and textbooks_config and 'textbooks' in textbooks_config and textbooks_config['textbooks']:
                            textbook_id = textbooks_config['textbooks'][0]['id']
                        chunks_data, chunks_error, doc_info = load_chunks(textbook_id)
                        
                        # Безопасная обработка статуса чанков
                        try:
                            chunks_count = len(chunks_data) if chunks_data else 0
                            chunks_status = f"**{chunks_count}**"
                            if chunks_error:
                                error_display = chunks_error[:50] + "..." if len(chunks_error) > 50 else chunks_error
                                chunks_status += f" ⚠️ ({error_display})"
                        except Exception as e:
                            chunks_status = f"**0** ⚠️ (Ошибка: {str(e)})"
                        
                        # Безопасная проверка подключения Ollama
                        try:
                            is_connected, msg = check_ollama_connection()
                            ollama_status = "✓ Подключен" if is_connected else f"✗ {msg}"
                        except Exception as e:
                            ollama_status = f"✗ Ошибка: {str(e)}"
                        
                        # Получаем название учебника
                        textbook_name = doc_info.get('title', 'Неизвестно') if doc_info else 'Неизвестно'
                        
                        return gr.Markdown(f"""
                        - Учебник: **{textbook_name}**
                        - Загружено чанков: {chunks_status}
                        - Предпромт: **{"✓ Загружен" if preprompt else "✗ Не загружен"}**
                        - Ollama: **{ollama_status}**
                        """)
                    except Exception as e:
                        return gr.Markdown(f"""
                        - Загружено чанков: **0** ⚠️ (Ошибка обновления: {str(e)})
                        - Предпромт: **{"✓ Загружен" if preprompt else "✗ Не загружен"}**
                        - Ollama: **✗ Ошибка проверки**
                        """)
                
                refresh_stats_btn.click(refresh_stats, None, stats_text)
            
            with gr.Column(scale=3):
                # Определяем chatbot в начале столбца, чтобы он был доступен в обработчиках
                chatbot = gr.Chatbot(
                    label="Чат с репетитором",
                    height=500,
                    show_copy_button=True,
                    value=[]  # Инициализируем пустым списком
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Например: Помоги решить задание 5 на странице 5",
                        scale=4,
                        value=""  # Инициализируем пустой строкой
                    )
                    submit_btn = gr.Button("Отправить", variant="primary", scale=1)
                
                clear_btn = gr.Button("Очистить чат", variant="secondary")
        
        # Обработчики событий
        def clear_chat():
            """Очищает чат"""
            return [], ""
        
        # Функция для обновления режимов при смене учебника
        def update_modes_on_textbook_change(textbook_name):
            """Обновляет список режимов при смене учебника"""
            # Находим ID учебника
            textbook_id = None
            for id_val, name in textbook_list:
                if name == textbook_name:
                    textbook_id = id_val
                    break
            
            if textbook_id:
                modes = get_preprompt_modes(textbook_id)
                default = get_default_mode(textbook_id)
                return gr.Dropdown(choices=list(modes.keys()), value=default)
            return gr.Dropdown(choices=["Стандартный"], value="Стандартный")
        
        # Привязываем смену учебника к обновлению данных (после определения всех компонентов)
        textbook_dropdown.change(
            change_textbook,
            inputs=[textbook_dropdown],
            outputs=[stats_text, chatbot]
        )
        
        # Обновляем режимы при смене учебника
        textbook_dropdown.change(
            update_modes_on_textbook_change,
            inputs=[textbook_dropdown],
            outputs=[mode_dropdown]
        )
        
        msg.submit(chat_interface, [msg, chatbot, model_dropdown, mode_dropdown], [chatbot, msg])
        submit_btn.click(chat_interface, [msg, chatbot, model_dropdown, mode_dropdown], [chatbot, msg])
        clear_btn.click(clear_chat, None, [chatbot, msg])
        
        gr.Markdown("""
        ### 💡 Примеры вопросов:
        - "Помоги решить задание 1 на странице 3"
        - "Как решить задачу про таблицу с расстояниями?"
        - "Объясни тему про отрезки и их длину"
        - "Что такое координатная прямая?"
        """)
    
    # Запускаем интерфейс
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=False  # Не показывать ошибки в UI
        )
    except KeyboardInterrupt:
        print("\nПриложение остановлено пользователем.")
    except Exception as e:
        if "10054" not in str(e) and "ConnectionResetError" not in str(type(e).__name__):
            # Выводим только не связанные с разрывом соединения ошибки
            print(f"Ошибка при запуске: {e}")

if __name__ == "__main__":
    main()


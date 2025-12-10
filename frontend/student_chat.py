"""
Модуль чата для школьника
"""
import streamlit as st
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    authenticate_student, get_student, get_or_create_chat, create_chat,
    get_chat_messages, add_message, rate_message, get_student_chats, get_chat, delete_chat,
    get_pending_practice_task, add_practice_task, submit_practice_answer, get_next_complexity_level
)
from rag_service import get_textbooks_list, answer_question_simple, evaluate_practice_answer, generate_practice_task
from core import rag
import ollama

def check_ollama_connection():
    """Проверяет подключение к Ollama"""
    try:
        ollama.list()
        return True, "Подключено"
    except Exception as e:
        return False, f"Ошибка: {str(e)}"

def get_available_models():
    """Получает список доступных моделей"""
    try:
        models_response = ollama.list()
        if isinstance(models_response, dict):
            models_list = models_response.get('models', [])
        elif isinstance(models_response, list):
            models_list = models_response
        else:
            models_list = []
        
        model_names = []
        for model in models_list:
            if isinstance(model, dict):
                name = model.get('name') or model.get('model') or str(model)
                if name:
                    model_names.append(name)
            elif isinstance(model, str):
                model_names.append(model)
        
        return model_names if model_names else ["qwen3:8b"]
    except:
        return ["qwen3:8b"]

def get_subject_from_textbook(textbook_name: str) -> str:
    """Определяет предмет по названию учебника"""
    textbook_name_lower = textbook_name.lower()
    if "математика" in textbook_name_lower:
        return "математика"
    elif "русский" in textbook_name_lower or "язык" in textbook_name_lower:
        return "русский язык"
    elif "история" in textbook_name_lower:
        return "история"
    elif "физика" in textbook_name_lower:
        return "физика"
    elif "химия" in textbook_name_lower:
        return "химия"
    elif "биология" in textbook_name_lower:
        return "биология"
    elif "география" in textbook_name_lower:
        return "география"
    elif "литература" in textbook_name_lower:
        return "литература"
    else:
        return "математика"  # По умолчанию

def show_student_chat():
    """Отображает интерфейс чата для школьника"""
    
    # Проверка авторизации
    if 'student_id' not in st.session_state:
        st.error("Вы не авторизованы. Пожалуйста, войдите в систему.")
        return
    
    student_id = st.session_state['student_id']
    student = get_student(student_id)
    
    if not student:
        st.error("Ошибка: данные школьника не найдены")
        return
    
    # Получаем все существующие чаты
    existing_chats = get_student_chats(student_id)
    textbooks = get_textbooks_list()
    
    if not textbooks:
        st.error("Учебники не найдены. Проверьте конфигурацию.")
        return
    
    textbook_options = {tb['id']: tb['name'] for tb in textbooks}
    
    # Инициализация выбранного чата из session state
    if 'selected_chat_id' not in st.session_state:
        if existing_chats:
            st.session_state['selected_chat_id'] = existing_chats[0]['id']
        else:
            st.session_state['selected_chat_id'] = None
    
    # Sidebar с чатами
    with st.sidebar:
        st.header("Чаты")
        
        # CSS и JavaScript для показа кнопки удаления при наведении и стилизации палитры оценок
        st.markdown("""
            <style>
            /* Делаем кнопку удаления полупрозрачной по умолчанию */
            .delete-chat-btn {
                opacity: 0.3 !important;
                transition: opacity 0.2s ease !important;
            }
            /* При наведении на строку с чатом показываем кнопку полностью */
            .chat-row:hover .delete-chat-btn {
                opacity: 1 !important;
            }
            /* Стилизация кнопок оценок - делаем их более заметными */
            button:has-text("★") {
                font-size: 1.2rem !important;
                padding: 0.5rem !important;
                min-height: 2.5rem !important;
            }
            </style>
            <script>
            // Добавляем классы для hover эффекта
            setTimeout(function() {
                const buttons = document.querySelectorAll('button');
                buttons.forEach(btn => {
                    if (btn.textContent.includes('[X]') || btn.textContent.includes('X')) {
                        btn.classList.add('delete-chat-btn');
                        // Находим родительский контейнер строки
                        let row = btn.closest('[data-testid="stHorizontalBlock"]');
                        if (row) {
                            row.classList.add('chat-row');
                        }
                    }
                });
            }, 100);
            </script>
            """, unsafe_allow_html=True)
        
        # Информация о школьнике
        st.caption(f"{student.get('full_name', 'Не указано')} • {student['grade']} класс")
        st.divider()
        
        # Список существующих чатов
        if existing_chats:
            st.subheader("Мои чаты", divider=False)
            
            # Группируем чаты по предмету и добавляем номера
            subject_counts = {}
            for chat in existing_chats:
                subject = chat.get('subject', 'математика')
                if subject not in subject_counts:
                    subject_counts[subject] = []
                subject_counts[subject].append(chat)
            
            # Отображаем чаты с номерами
            for chat in existing_chats:
                subject = chat.get('subject', 'математика')
                subject_capitalized = subject.capitalize()
                
                # Определяем номер чата для этого предмета
                same_subject_chats = subject_counts[subject]
                # Сортируем по дате создания, чтобы номера были стабильными
                same_subject_chats_sorted = sorted(same_subject_chats, key=lambda x: x.get('created_at', ''))
                chat_number = same_subject_chats_sorted.index(chat) + 1
                
                # Если чатов с этим предметом больше одного, добавляем номер в скобках
                if len(same_subject_chats) > 1:
                    chat_label = f"{subject_capitalized} ({chat_number})"
                else:
                    chat_label = f"{subject_capitalized}"
                
                # Выделяем активный чат
                is_active = st.session_state.get('selected_chat_id') == chat['id']
                button_type = "primary" if is_active else "secondary"
                
                # Создаем контейнер с кнопкой чата и кнопкой удаления
                col_chat, col_delete = st.columns([4, 1])
                
                with col_chat:
                    if st.button(
                        chat_label,
                        key=f"chat_btn_{chat['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state['selected_chat_id'] = chat['id']
                        st.rerun()
                
                with col_delete:
                    # Кнопка удаления (показывается всегда, но маленькая)
                    if st.button(
                        "[X]",
                        key=f"delete_chat_{chat['id']}",
                        help="Удалить чат",
                        use_container_width=True
                    ):
                        if delete_chat(chat['id'], student_id):
                            # Если удаляемый чат был активным, выбираем другой
                            if st.session_state.get('selected_chat_id') == chat['id']:
                                remaining_chats = [c for c in existing_chats if c['id'] != chat['id']]
                                if remaining_chats:
                                    st.session_state['selected_chat_id'] = remaining_chats[0]['id']
                                else:
                                    st.session_state['selected_chat_id'] = None
                            st.success("Чат удален")
                            st.rerun()
                        else:
                            st.error("Ошибка при удалении чата")
        
        st.divider()
        
        # Создание нового чата
        st.subheader("Создать чат", divider=False)
        
        textbook_options_list = {tb['name']: tb['id'] for tb in textbooks}
        selected_textbook_name = st.selectbox(
            "Учебник",
            options=list(textbook_options_list.keys()),
            key="new_textbook_select",
            label_visibility="collapsed"
        )
        selected_textbook_id = textbook_options_list[selected_textbook_name]
        
        if st.button("Создать новый чат", key="create_chat_btn", use_container_width=True):
            # Определяем предмет из названия учебника
            selected_textbook = next((tb for tb in textbooks if tb['id'] == selected_textbook_id), None)
            if selected_textbook:
                subject = get_subject_from_textbook(selected_textbook['name'])
            else:
                subject = "математика"  # По умолчанию
            
            # Всегда создаем новый чат (можно создавать несколько чатов по одному предмету)
            new_chat_id = create_chat(
                student_id=student_id,
                textbook_id=selected_textbook_id,
                subject=subject,
                grade=student['grade']
            )
            st.session_state['selected_chat_id'] = new_chat_id
            st.success("Чат создан!")
            st.rerun()
    
    # Основная область - чат
    chat_id = st.session_state.get('selected_chat_id')
    
    if not chat_id:
        st.info("Создайте новый чат в боковой панели, чтобы начать общение")
        return
    
    selected_chat = get_chat(chat_id)
    if not selected_chat:
        st.error("Чат не найден")
        return
    
    current_textbook_id = selected_chat['textbook_id']
    current_subject = selected_chat['subject']
    current_textbook_name = textbook_options.get(current_textbook_id, current_textbook_id)
    
    # Заголовок с кнопкой выхода
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("Чат с репетитором")
        st.caption(f"{current_textbook_name} • {current_subject}")
    with col2:
        st.write("")  # Отступ
        st.write("")  # Отступ
        if st.button("Выйти", key="logout_btn", use_container_width=True):
            st.session_state['student_id'] = None
            st.session_state['user_type'] = None
            st.session_state['selected_chat_id'] = None
            st.rerun()
    
    st.divider()
    
    # Загружаем историю сообщений
    messages = get_chat_messages(chat_id)
    
    # Отображаем историю чата
    for msg in messages:
        with st.chat_message("user"):
            st.write(msg['student_query'])
        
        with st.chat_message("assistant"):
            st.write(msg['llm_response'])
            
            # Проверяем, есть ли ошибка в ответе (по формату ответа)
            has_error = msg['llm_response'].startswith('[X]') or msg['llm_response'].startswith('[!]') or 'Ошибка' in msg['llm_response']
            
            # Показываем оценку, если есть
            if msg['rating']:
                with st.container():
                    st.markdown("---")
                    st.markdown(f"**Оценка:** {'★' * msg['rating']}{'☆' * (5 - msg['rating'])} ({msg['rating']}/5)")
                    if msg['is_best_answer']:
                        st.success("Лучший ответ")
            
            # Кнопка оценки (только если нет ошибки и еще не оценено)
            if not msg['rating'] and not has_error:
                st.markdown("---")
                # Создаем палитру для оценки
                rating_key = f"rating_palette_{msg['id']}"
                st.markdown(f"""
                    <div id="{rating_key}" style="
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 1rem;
                        background-color: #f9f9f9;
                        margin: 1rem 0;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: #333; font-size: 1.1rem;">ОЦЕНИТЕ ОТВЕТ</h4>
                    </div>
                """, unsafe_allow_html=True)
                cols = st.columns(5)
                for i, col in enumerate(cols, 1):
                    with col:
                        if st.button(
                            "★" * i + "☆" * (5 - i),
                            key=f"rate_{msg['id']}_{i}",
                            use_container_width=True,
                            help=f"Оценить на {i} из 5"
                        ):
                            rate_message(msg['id'], i)
                            st.rerun()
    
    # Настройки внизу (модель и режим репетитора)
    st.divider()
    
    # Получаем доступные режимы для текущего учебника
    preprompt_modes = rag.get_preprompt_modes(current_textbook_id)
    mode_choices = list(preprompt_modes.keys())
    default_mode = rag.get_default_mode(current_textbook_id)
    
    # Определяем индекс режима по умолчанию
    default_mode_index = mode_choices.index(default_mode) if default_mode in mode_choices else 0
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Выбор режима репетитора
        selected_mode = st.selectbox(
            "Режим репетитора",
            options=mode_choices,
            index=default_mode_index,
            key="preprompt_mode_select",
            help="Выберите стиль ответов: Стандартный, Краткий или Подробный"
        )
    
    with col2:
        # Выбор модели
        available_models = get_available_models()
        # Дефолтная модель - 8b, если доступна, иначе первая из списка
        default_model = "qwen3:8b"
        if "qwen3:8b" not in available_models and available_models:
            default_model = available_models[0]
        default_index = available_models.index(default_model) if default_model in available_models else 0
        selected_model = st.selectbox(
            "Модель",
            options=available_models,
            index=default_index,
            key="model_select"
        )
    
    with col3:
        st.write("")  # Отступ
        st.write("")  # Отступ
        is_connected, conn_msg = check_ollama_connection()
        if not is_connected:
            st.warning(f"{conn_msg}")
    
    # Проверяем, есть ли незавершенное практическое задание
    pending_task = get_pending_practice_task(chat_id)
    
    if pending_task:
        # Показываем текущее практическое задание
        st.info(f"**Практическое задание:**\n\n{pending_task['task_text']}")
        st.caption("Введите ваш ответ ниже")
    
    # Поле для нового вопроса или ответа на практическое задание
    placeholder_text = "Введите ответ на практическое задание..." if pending_task else "Задайте вопрос по учебнику..."
    user_query = st.chat_input(placeholder_text)
    
    if user_query:
        # Если есть незавершенное задание, обрабатываем ответ
        if pending_task:
            # Добавляем ответ ученика
            with st.chat_message("user"):
                st.write(user_query)
            
            # Оцениваем ответ
            with st.chat_message("assistant"):
                with st.spinner("Оцениваю ответ..."):
                    evaluation_result = evaluate_practice_answer(
                        task_text=pending_task['task_text'],
                        correct_answer=pending_task['correct_answer'],
                        student_answer=user_query,
                        textbook_id=current_textbook_id,
                        grade=student['grade'],
                        model_name=selected_model,
                        preprompt_mode=selected_mode
                    )
                    
                    st.write(evaluation_result['evaluation'])
                    
                    # Сохраняем оценку
                    if evaluation_result['grade']:
                        submit_practice_answer(
                            task_id=pending_task['id'],
                            student_answer=user_query,
                            grade=evaluation_result['grade'],
                            feedback=evaluation_result['feedback']
                        )
                        st.success(f"Оценка сохранена: {evaluation_result['grade']}/5")
                        
                        # Если оценка хорошая (4-5), следующее задание будет сложнее
                        if evaluation_result['grade'] >= 4:
                            st.info("Отлично! Следующее задание будет сложнее.")
                        elif evaluation_result['grade'] <= 2:
                            st.info("Попробуйте еще раз. Следующее задание будет того же уровня.")
            
            st.rerun()
        
        else:
            # Обычный запрос
            # Проверяем, не согласился ли пользователь на практику
            user_query_lower = user_query.lower().strip()
            wants_practice = any(word in user_query_lower for word in ['да', 'хочу', 'готов', 'давай', 'попробую', 'решу'])
            
            # Добавляем вопрос пользователя в историю
            with st.chat_message("user"):
                st.write(user_query)
            
            # Получаем ответ от LLM
            with st.chat_message("assistant"):
                with st.spinner("Обработка запроса..."):
                    result = answer_question_simple(
                        query=user_query,
                        textbook_id=current_textbook_id,
                        grade=student['grade'],
                        model_name=selected_model,
                        preprompt_mode=selected_mode
                    )
                    
                    st.write(result['answer'])
                    
                    # Проверяем наличие ошибки
                    has_error = result.get('error') is not None or result['answer'].startswith('[X]') or result['answer'].startswith('[!]') or 'Ошибка' in result['answer']
                    
                    # Показываем источники только если нет ошибки
                    if not has_error and result['sources']:
                        with st.expander("Источники"):
                            for source in result['sources']:
                                st.write(f"Страница: {source.get('page', 'Не указана')}")
                                if source.get('topic'):
                                    st.write(f"Тема: {source['topic']}")
                    
                    # Сохраняем сообщение в БД
                    message_id = add_message(
                        chat_id=chat_id,
                        student_query=user_query,
                        llm_response=result['answer']
                    )
                    
                    # Если пользователь согласился на практику, генерируем задание
                    if wants_practice and not has_error:
                        # Определяем уровень сложности на основе предыдущих оценок
                        complexity = get_next_complexity_level(chat_id)
                        
                        # Генерируем практическое задание
                        with st.spinner("Генерирую практическое задание..."):
                            task_result = generate_practice_task(
                                original_query=user_query,
                                original_solution=result['answer'],
                                textbook_id=current_textbook_id,
                                grade=student['grade'],
                                complexity_level=complexity,
                                model_name=selected_model,
                                preprompt_mode=selected_mode
                            )
                            
                            if not task_result.get('error') and task_result.get('task_text'):
                                # Сохраняем практическое задание
                                task_id = add_practice_task(
                                    chat_id=chat_id,
                                    original_message_id=message_id,
                                    task_text=task_result['task_text'],
                                    correct_answer=task_result['correct_answer'],
                                    complexity_level=complexity
                                )
                                
                                st.info(f"**Практическое задание:**\n\n{task_result['task_text']}\n\n**Подсказка:** {task_result.get('hint', '')}\n\nВведите ваш ответ выше.")
                            else:
                                st.warning("Не удалось сгенерировать практическое задание. Попробуйте позже.")
                    
                    # Проверяем, есть ли в ответе предложение практического задания (даже если пользователь не согласился)
                    # LLM должен всегда предлагать практику после решения
                    if not has_error and not wants_practice:
                        # Проверяем, есть ли в ответе предложение практики
                        answer_lower = result['answer'].lower()
                        if 'хочу' in answer_lower or 'попробовать' in answer_lower or 'похожее' in answer_lower:
                            # LLM уже предложил практику в ответе, ничего не делаем
                            pass
                    
                    # Кнопки для оценки (только если нет ошибки)
                    if not has_error:
                        st.markdown("---")
                        # Создаем палитру для оценки
                        rating_key = f"rating_palette_new_{message_id}"
                        st.markdown(f"""
                            <div id="{rating_key}" style="
                                border: 2px solid #e0e0e0;
                                border-radius: 8px;
                                padding: 1rem;
                                background-color: #f9f9f9;
                                margin: 1rem 0;
                            ">
                                <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: #333; font-size: 1.1rem;">ОЦЕНИТЕ ОТВЕТ</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        cols = st.columns(5)
                        for i, col in enumerate(cols, 1):
                            with col:
                                if st.button(
                                    "★" * i + "☆" * (5 - i),
                                    key=f"rate_new_{i}",
                                    use_container_width=True,
                                    help=f"Оценить на {i} из 5"
                                ):
                                    rate_message(message_id, i)
                                    st.rerun()
            
            st.rerun()


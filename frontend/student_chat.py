"""
Модуль чата для школьника
"""
import streamlit as st
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    authenticate_student, get_student, get_or_create_chat,
    get_chat_messages, add_message, rate_message, get_student_chats, get_chat
)
from rag_service import get_textbooks_list, answer_question_simple
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
        
        return model_names if model_names else ["qwen2.5:3b"]
    except:
        return ["qwen2.5:3b"]

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
        
        # Информация о школьнике
        st.caption(f"{student.get('full_name', 'Не указано')} • {student['grade']} класс")
        st.divider()
        
        # Список существующих чатов
        if existing_chats:
            st.subheader("Мои чаты", divider=False)
            for chat in existing_chats:
                textbook_name = textbook_options.get(chat['textbook_id'], chat['textbook_id'])
                chat_label = f"{textbook_name}"
                
                # Выделяем активный чат
                is_active = st.session_state.get('selected_chat_id') == chat['id']
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    chat_label,
                    key=f"chat_btn_{chat['id']}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state['selected_chat_id'] = chat['id']
                    st.rerun()
        
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
            # Проверяем, не существует ли уже такой чат
            existing_chat = None
            for chat in existing_chats:
                if (chat['textbook_id'] == selected_textbook_id and 
                    chat['subject'] == "математика" and 
                    chat['grade'] == student['grade']):
                    existing_chat = chat
                    break
            
            if existing_chat:
                st.warning("Чат с этим учебником уже существует")
                st.session_state['selected_chat_id'] = existing_chat['id']
                st.rerun()
            else:
                new_chat_id = get_or_create_chat(
                    student_id=student_id,
                    textbook_id=selected_textbook_id,
                    subject="математика",
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
    
    # Выбор модели
    col1, col2 = st.columns([3, 1])
    with col1:
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Модель",
            options=available_models,
            key="model_select"
        )
    with col2:
        is_connected, conn_msg = check_ollama_connection()
        if not is_connected:
            st.warning(f"{conn_msg}")
    
    # Загружаем историю сообщений
    messages = get_chat_messages(chat_id)
    
    # Отображаем историю чата
    for msg in messages:
        with st.chat_message("user"):
            st.write(msg['student_query'])
        
        with st.chat_message("assistant"):
            st.write(msg['llm_response'])
            
            # Проверяем, есть ли ошибка в ответе (по формату ответа)
            has_error = msg['llm_response'].startswith('❌') or msg['llm_response'].startswith('⚠️') or 'Ошибка' in msg['llm_response']
            
            # Показываем оценку, если есть
            if msg['rating']:
                st.caption(f"Оценка: {msg['rating']}/5")
                if msg['is_best_answer']:
                    st.success("Лучший ответ")
            
            # Кнопка оценки (только если нет ошибки и еще не оценено)
            if not msg['rating'] and not has_error:
                st.caption("Оцените ответ:")
                cols = st.columns(5)
                for i, col in enumerate(cols, 1):
                    with col:
                        if st.button(f"{i}", key=f"rate_{msg['id']}_{i}"):
                            rate_message(msg['id'], i)
                            st.rerun()
    
    # Поле для нового вопроса
    user_query = st.chat_input("Задайте вопрос по учебнику...")
    
    if user_query:
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
                    model_name=selected_model
                )
                
                st.write(result['answer'])
                
                # Проверяем наличие ошибки
                has_error = result.get('error') is not None or result['answer'].startswith('❌') or result['answer'].startswith('⚠️') or 'Ошибка' in result['answer']
                
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
                
                # Кнопки для оценки (только если нет ошибки)
                if not has_error:
                    st.caption("Оцените ответ:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols, 1):
                        with col:
                            if st.button(f"{i}", key=f"rate_new_{i}"):
                                rate_message(message_id, i)
                                st.rerun()
        
        st.rerun()


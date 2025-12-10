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
    get_chat_messages, add_message, rate_message
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
    
    st.title("Чат с репетитором")
    
    # Информация о школьнике
    with st.expander("Информация о школьнике", expanded=False):
        st.write(f"**ФИО:** {student.get('full_name', 'Не указано')}")
        st.write(f"**Класс:** {student['grade']}")
        st.write(f"**Логин:** {student['login']}")
    
    # Выбор учебника и предмета
    col1, col2 = st.columns(2)
    
    with col1:
        textbooks = get_textbooks_list()
        if not textbooks:
            st.error("Учебники не найдены. Проверьте конфигурацию.")
            return
        
        textbook_options = {tb['name']: tb['id'] for tb in textbooks}
        selected_textbook_name = st.selectbox(
            "Выберите учебник",
            options=list(textbook_options.keys()),
            key="textbook_select"
        )
        selected_textbook_id = textbook_options[selected_textbook_name]
    
    with col2:
        # Для MVP - только математика
        subject = st.selectbox(
            "Предмет",
            options=["математика"],
            disabled=True,
            key="subject_select"
        )
    
    # Выбор модели
    is_connected, conn_msg = check_ollama_connection()
    if not is_connected:
        st.warning(f"{conn_msg}. Убедитесь, что Ollama запущен.")
    
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Модель",
        options=available_models,
        key="model_select"
    )
    
    # Получаем или создаем чат
    chat_id = get_or_create_chat(
        student_id=student_id,
        textbook_id=selected_textbook_id,
        subject=subject,
        grade=student['grade']
    )
    
    # Загружаем историю сообщений
    messages = get_chat_messages(chat_id)
    
    # Отображаем историю чата
    st.markdown("---")
    st.subheader("История чата")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in messages:
            with st.chat_message("user"):
                st.write(msg['student_query'])
            
            with st.chat_message("assistant"):
                st.write(msg['llm_response'])
                
                # Показываем оценку, если есть
                if msg['rating']:
                    st.write(f"Оценка: {msg['rating']}/5")
                    if msg['is_best_answer']:
                        st.success("Лучший ответ")
                
                # Кнопка оценки (если еще не оценено)
                if not msg['rating']:
                    st.caption("Оцените ответ:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols, 1):
                        with col:
                            if st.button(f"{i}", key=f"rate_{msg['id']}_{i}"):
                                rate_message(msg['id'], i)
                                st.rerun()
    
    # Поле для нового вопроса
    st.markdown("---")
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
                    textbook_id=selected_textbook_id,
                    grade=student['grade'],
                    model_name=selected_model
                )
                
                st.write(result['answer'])
                
                # Показываем источники
                if result['sources']:
                    with st.expander("Источники"):
                        for source in result['sources']:
                            st.write(f"- Страница: {source.get('page', 'Не указана')}")
                            if source.get('topic'):
                                st.write(f"  Тема: {source['topic']}")
                
                # Сохраняем сообщение в БД
                message_id = add_message(
                    chat_id=chat_id,
                    student_query=user_query,
                    llm_response=result['answer']
                )
                
                # Кнопки для оценки
                st.caption("Оцените ответ:")
                cols = st.columns(5)
                for i, col in enumerate(cols, 1):
                    with col:
                        if st.button(f"{i}", key=f"rate_new_{i}"):
                            rate_message(message_id, i)
                            st.rerun()
        
        st.rerun()


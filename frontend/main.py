"""
Главный файл запуска приложения
"""
import streamlit as st
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import authenticate_parent, authenticate_student, create_parent
from student_chat import show_student_chat
from parent_dashboard import show_parent_dashboard

# Настройка страницы
st.set_page_config(
    page_title="Репетитор для школьников",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def show_login():
    """Отображает страницу входа"""
    
    # Центрированный заголовок
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Репетитор для школьников")
        st.markdown("---")
    
    # Выбор типа пользователя
    user_type = st.radio(
        "Тип пользователя:",
        ["Родитель", "Школьник"],
        horizontal=True,
        key="login_user_type"
    )
    
    st.markdown("---")
    
    if user_type == "Родитель":
        # Вкладки для логина и регистрации
        tab1, tab2 = st.tabs(["Вход", "Регистрация"])
        
        with tab1:
            st.subheader("Вход в систему")
            
            with st.form("parent_login"):
                username = st.text_input("Имя пользователя", placeholder="Введите имя пользователя")
                password = st.text_input("Пароль", type="password", placeholder="Введите пароль")
                submitted = st.form_submit_button("Войти", use_container_width=True)
                
                if submitted:
                    if not username or not password:
                        st.error("Заполните все поля")
                    else:
                        parent_id = authenticate_parent(username, password)
                        if parent_id:
                            st.session_state['parent_id'] = parent_id
                            st.session_state['user_type'] = 'parent'
                            st.success("Вход выполнен успешно")
                            st.rerun()
                        else:
                            st.error("Неверное имя пользователя или пароль")
        
        with tab2:
            st.subheader("Регистрация нового аккаунта")
            st.caption("После регистрации вы сможете добавить школьника и получить для него логин и пароль")
            
            with st.form("parent_register"):
                new_username = st.text_input("Имя пользователя", key="reg_username", placeholder="Придумайте имя пользователя")
                new_password = st.text_input("Пароль", type="password", key="reg_password", placeholder="Придумайте пароль")
                telegram = st.text_input("Telegram (необязательно)", key="reg_telegram", placeholder="@username")
                submitted_reg = st.form_submit_button("Зарегистрироваться", use_container_width=True)
                
                if submitted_reg:
                    if not new_username or not new_password:
                        st.error("Заполните имя пользователя и пароль")
                    else:
                        try:
                            parent_id = create_parent(new_username, new_password, telegram if telegram else None)
                            st.success("Регистрация успешна! Теперь вы можете войти.")
                            st.info("Перейдите на вкладку 'Вход' для входа в систему")
                        except ValueError as e:
                            st.error(str(e))
    
    else:  # Школьник
        st.subheader("Вход для школьника")
        st.caption("Используйте логин и пароль, полученные от родителя")
        
        with st.form("student_login"):
            login = st.text_input("Логин", placeholder="Введите логин")
            password = st.text_input("Пароль", type="password", placeholder="Введите пароль")
            submitted = st.form_submit_button("Войти", use_container_width=True)
            
            if submitted:
                if not login or not password:
                    st.error("Заполните все поля")
                else:
                    student_id = authenticate_student(login, password)
                    if student_id:
                        st.session_state['student_id'] = student_id
                        st.session_state['user_type'] = 'student'
                        st.success("Вход выполнен успешно")
                        st.rerun()
                    else:
                        st.error("Неверный логин или пароль")

def main():
    """Главная функция"""
    
    # Инициализация сессии
    if 'parent_id' not in st.session_state:
        st.session_state['parent_id'] = None
    if 'student_id' not in st.session_state:
        st.session_state['student_id'] = None
    if 'user_type' not in st.session_state:
        st.session_state['user_type'] = None
    if 'login_user_type' not in st.session_state:
        st.session_state['login_user_type'] = "Родитель"
    
    # Навигация
    if st.session_state['user_type'] == 'parent':
        # Кнопка выхода
        if st.sidebar.button("Выйти"):
            st.session_state['parent_id'] = None
            st.session_state['user_type'] = None
            st.rerun()
        
        show_parent_dashboard()
    
    elif st.session_state['user_type'] == 'student':
        # Кнопка выхода теперь в самом чате
        show_student_chat()
    
    else:
        show_login()

if __name__ == "__main__":
    main()


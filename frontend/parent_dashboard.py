"""
Модуль дашборда для родителя
"""
import streamlit as st
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    authenticate_parent, get_parent, get_students_by_parent,
    create_student, get_student_metrics, update_parent_telegram, delete_student
)
import pandas as pd

def show_parent_dashboard():
    """Отображает дашборд родителя"""
    
    # Проверка авторизации
    if 'parent_id' not in st.session_state:
        st.error("Вы не авторизованы. Пожалуйста, войдите в систему.")
        return
    
    parent_id = st.session_state['parent_id']
    parent = get_parent(parent_id)
    
    if not parent:
        st.error("Ошибка: данные родителя не найдены")
        return
    
    st.title("Дашборд родителя")
    
    # Информация о родителе
    with st.expander("Личный кабинет", expanded=False):
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**Имя пользователя:** {parent['username']}")
        
        with col2:
            # Telegram
            telegram = st.text_input(
                "Telegram",
                value=parent.get('telegram', '') or '',
                key="parent_telegram",
                placeholder="@username"
            )
        
        if st.button("Сохранить Telegram", type="primary"):
            if telegram:
                update_parent_telegram(parent_id, telegram)
                st.success("Telegram сохранен")
                st.rerun()
            else:
                st.warning("Введите Telegram")
    
    st.divider()
    
    # Добавление нового школьника
    st.subheader("Добавить школьника")
    st.caption("После добавления школьника вы получите логин и пароль для его входа в систему")
    
    with st.form("add_student_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input(
                "ФИО школьника *", 
                key="new_student_name", 
                placeholder="Иванов Иван Иванович",
                help="Обязательное поле"
            )
            username = st.text_input(
                "Логин (необязательно)", 
                key="new_student_username", 
                placeholder="Оставьте пустым для автогенерации", 
                help="Будет сгенерирован автоматически, если не указан"
            )
        
        with col2:
            # Выпадающий список для выбора класса с возможностью ручного ввода
            grade_options = [str(i) for i in range(1, 12)]  # От 1 до 11 класса
            grade_options.append("Ввести вручную")  # Добавляем опцию ручного ввода
            
            grade_choice = st.selectbox(
                "Класс *",
                options=grade_options,
                index=4,  # По умолчанию 5 класс (индекс 4)
                key="new_student_grade_select",
                help="Выберите класс из списка или введите вручную"
            )
            
            # Если выбрано "Ввести вручную", показываем поле для ввода
            if grade_choice == "Ввести вручную":
                grade = st.number_input(
                    "Введите класс",
                    min_value=1,
                    max_value=11,
                    value=5,
                    step=1,
                    key="new_student_grade_custom"
                )
            else:
                grade = int(grade_choice)
            
            telegram = st.text_input(
                "Telegram (необязательно)", 
                key="new_student_telegram", 
                placeholder="@username"
            )
        
        submitted = st.form_submit_button("Добавить школьника", type="primary", use_container_width=True)
        
        if submitted:
            if not full_name:
                st.error("Введите ФИО школьника")
            else:
                try:
                    result = create_student(
                        parent_id=parent_id,
                        full_name=full_name,
                        grade=grade,
                        username=username if username else None,
                        telegram=telegram if telegram else None
                    )
                    
                    st.success("Школьник успешно добавлен!")
                    with st.container():
                        st.markdown("### Данные для входа:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Логин:**")
                            st.code(result['login'], language=None)
                        with col2:
                            st.markdown(f"**Пароль:**")
                            st.code(result['password'], language=None)
                        st.warning("Сохраните эти данные! Они понадобятся для входа школьника в систему.")
                    # Не делаем rerun, чтобы показать credentials
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Ошибка при добавлении: {str(e)}")
    
    st.divider()
    
    # Список школьников
    st.subheader("Мои школьники")
    
    students = get_students_by_parent(parent_id)
    
    if not students:
        st.info("У вас пока нет добавленных школьников. Добавьте первого выше.")
    else:
        # Карточки школьников
        for idx, student in enumerate(students):
            with st.container():
                # Используем карточку с рамкой
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
                    
                    with col1:
                        st.markdown(f"### {student.get('full_name', 'Не указано')}")
                        st.markdown(f"Класс **{student['grade']}**")
                    
                    with col2:
                        st.markdown("**Логин:**")
                        st.code(student['login'], language=None)
                    
                    with col3:
                        st.markdown("**Пароль:**")
                        st.code(student['password'], language=None)
                    
                    with col4:
                        st.markdown("<br>", unsafe_allow_html=True)  # Отступ сверху
                        if st.button("Удалить", key=f"delete_{student['id']}", type="secondary", use_container_width=True):
                            if delete_student(student['id'], parent_id):
                                st.success("Школьник удален")
                                st.rerun()
                            else:
                                st.error("Ошибка при удалении")
                
                if idx < len(students) - 1:
                    st.divider()
    
    st.divider()
    
    # Метрики успеваемости
    if students:
        st.subheader("Метрики успеваемости")
        
        selected_student_id = st.selectbox(
            "Выберите школьника для просмотра метрик",
            options=[s['id'] for s in students],
            format_func=lambda x: next(s['full_name'] for s in students if s['id'] == x),
            key="metrics_student_select"
        )
        
        if selected_student_id:
            metrics = get_student_metrics(selected_student_id)
            selected_student = next(s for s in students if s['id'] == selected_student_id)
            
            st.markdown(f"### Прогресс: {selected_student.get('full_name', 'Школьник')}")
            
            # Основные метрики прогресса
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Средняя оценка",
                    f"{metrics['avg_rating']:.2f}",
                    delta=f"{metrics['rating_trend']:+.2f}" if metrics['rating_trend'] != 0 else None,
                    delta_color="normal" if metrics['rating_trend'] >= 0 else "inverse",
                    help="Общая средняя оценка ответов"
                )
            
            with col2:
                st.metric(
                    "Оценка за неделю",
                    f"{metrics['avg_rating_recent']:.2f}" if metrics['avg_rating_recent'] > 0 else "Нет данных",
                    help="Средняя оценка за последние 7 дней"
                )
            
            with col3:
                st.metric(
                    "Дней активности",
                    f"{metrics['active_days']}/30",
                    help="Количество дней с активностью за последние 30 дней"
                )
            
            with col4:
                st.metric(
                    "Дней с начала",
                    metrics['days_since_start'],
                    help="Количество дней с начала работы с системой"
                )
            
            # График прогресса
            if metrics['daily_activity']:
                st.markdown("#### Прогресс за последние 14 дней")
                activity_df = pd.DataFrame(metrics['daily_activity'])
                activity_df['date'] = pd.to_datetime(activity_df['date'])
                
                # График средней оценки по дням
                chart_data = activity_df.set_index('date')[['avg_rating']]
                st.line_chart(chart_data, y_label="Средняя оценка")
                
                # График активности
                st.markdown("#### Активность по дням")
                activity_chart = activity_df.set_index('date')[['count']]
                st.bar_chart(activity_chart, y_label="Количество вопросов")
            else:
                st.info("Пока нет данных для отображения графика прогресса")


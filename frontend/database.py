"""
Модуль для работы с базой данных (SQLite для MVP)
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"

def get_db_connection():
    """Создает подключение к БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Инициализирует структуру БД"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Таблица родителей
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            telegram TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Таблица школьников
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            full_name TEXT,
            grade INTEGER NOT NULL,
            login TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            telegram TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES parents(id)
        )
    """)
    
    # Таблица чатов (один чат = один учебник)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            textbook_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            grade INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    
    # Таблица сообщений в чате
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            student_query TEXT NOT NULL,
            llm_response TEXT NOT NULL,
            rating INTEGER,
            is_best_answer BOOLEAN DEFAULT 0,
            needs_review BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    
    # Таблица лучших ответов (для рейтинга)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS best_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            rating INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)
    
    # Таблица практических заданий
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            original_message_id INTEGER,
            task_text TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            student_answer TEXT,
            grade INTEGER,
            complexity_level INTEGER DEFAULT 1,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id),
            FOREIGN KEY (original_message_id) REFERENCES messages(id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Хеширует пароль"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_username() -> str:
    """Генерирует красивое русское имя пользователя"""
    import random
    # Русские имена для генерации
    first_names = ['Иван', 'Алексей', 'Дмитрий', 'Максим', 'Артем', 'Андрей', 'Сергей', 'Никита', 'Михаил', 'Даниил']
    last_names = ['Иванов', 'Петров', 'Сидоров', 'Смирнов', 'Кузнецов', 'Попов', 'Соколов', 'Лебедев', 'Козлов', 'Новиков']
    
    first = random.choice(first_names)
    last = random.choice(last_names)
    number = random.randint(100, 999)
    
    return f"{first}{last}{number}".lower()

def generate_password() -> str:
    """Генерирует пароль для школьника"""
    return secrets.token_urlsafe(8)

# === РОДИТЕЛИ ===

def create_parent(username: str, password: str, telegram: Optional[str] = None) -> int:
    """Создает родителя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    try:
        cursor.execute("""
            INSERT INTO parents (username, password_hash, telegram)
            VALUES (?, ?, ?)
        """, (username, password_hash, telegram))
        parent_id = cursor.lastrowid
        conn.commit()
        return parent_id
    except sqlite3.IntegrityError:
        raise ValueError("Пользователь с таким именем уже существует")
    finally:
        conn.close()

def authenticate_parent(username: str, password: str) -> Optional[int]:
    """Аутентифицирует родителя, возвращает ID или None"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    cursor.execute("""
        SELECT id FROM parents
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))
    
    result = cursor.fetchone()
    conn.close()
    
    return result['id'] if result else None

def get_parent(parent_id: int) -> Optional[Dict[str, Any]]:
    """Получает информацию о родителе"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM parents WHERE id = ?", (parent_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def update_parent_telegram(parent_id: int, telegram: str):
    """Обновляет Telegram родителя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE parents SET telegram = ? WHERE id = ?
    """, (telegram, parent_id))
    
    conn.commit()
    conn.close()

# === ШКОЛЬНИКИ ===

def create_student(parent_id: int, full_name: str, grade: int, username: Optional[str] = None, telegram: Optional[str] = None) -> Dict[str, str]:
    """Создает школьника и возвращает логин/пароль"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Генерируем username если не предоставлен
    if not username:
        username = generate_username()
        # Проверяем уникальность
        cursor.execute("SELECT id FROM students WHERE login = ?", (username,))
        while cursor.fetchone():
            username = generate_username()
            cursor.execute("SELECT id FROM students WHERE login = ?", (username,))
    
    # Проверяем уникальность предоставленного username
    else:
        cursor.execute("SELECT id FROM students WHERE login = ?", (username,))
        if cursor.fetchone():
            conn.close()
            raise ValueError("Пользователь с таким логином уже существует")
    
    password = generate_password()
    
    cursor.execute("""
        INSERT INTO students (parent_id, full_name, grade, login, password, telegram)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (parent_id, full_name, grade, username, password, telegram))
    
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        'student_id': student_id,
        'login': username,
        'password': password
    }

def authenticate_student(login: str, password: str) -> Optional[int]:
    """Аутентифицирует школьника, возвращает ID или None"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id FROM students
        WHERE login = ? AND password = ?
    """, (login, password))
    
    result = cursor.fetchone()
    conn.close()
    
    return result['id'] if result else None

def get_student(student_id: int) -> Optional[Dict[str, Any]]:
    """Получает информацию о школьнике"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_students_by_parent(parent_id: int) -> List[Dict[str, Any]]:
    """Получает всех школьников родителя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM students WHERE parent_id = ?", (parent_id,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def delete_student(student_id: int, parent_id: int) -> bool:
    """Удаляет школьника (только если он принадлежит родителю)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем, что школьник принадлежит родителю
    cursor.execute("SELECT id FROM students WHERE id = ? AND parent_id = ?", (student_id, parent_id))
    if not cursor.fetchone():
        conn.close()
        return False
    
    # Удаляем все связанные данные (каскадное удаление)
    # Сначала удаляем сообщения
    cursor.execute("""
        DELETE FROM messages 
        WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
    """, (student_id,))
    
    # Удаляем чаты
    cursor.execute("DELETE FROM chats WHERE student_id = ?", (student_id,))
    
    # Удаляем школьника
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    
    conn.commit()
    conn.close()
    return True

# === ЧАТЫ ===

def get_student_chats(student_id: int) -> List[Dict[str, Any]]:
    """Получает все чаты школьника"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM chats
        WHERE student_id = ?
        ORDER BY created_at DESC
    """, (student_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_or_create_chat(student_id: int, textbook_id: str, subject: str, grade: int) -> int:
    """Получает или создает чат для школьника и учебника"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем существующий чат
    cursor.execute("""
        SELECT id FROM chats
        WHERE student_id = ? AND textbook_id = ? AND subject = ? AND grade = ?
    """, (student_id, textbook_id, subject, grade))
    
    result = cursor.fetchone()
    
    if result:
        chat_id = result['id']
    else:
        # Создаем новый чат
        cursor.execute("""
            INSERT INTO chats (student_id, textbook_id, subject, grade)
            VALUES (?, ?, ?, ?)
        """, (student_id, textbook_id, subject, grade))
        chat_id = cursor.lastrowid
        conn.commit()
    
    conn.close()
    return chat_id

def create_chat(student_id: int, textbook_id: str, subject: str, grade: int) -> int:
    """Всегда создает новый чат (даже если уже есть чат с такими же параметрами)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Всегда создаем новый чат
    cursor.execute("""
        INSERT INTO chats (student_id, textbook_id, subject, grade)
        VALUES (?, ?, ?, ?)
    """, (student_id, textbook_id, subject, grade))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return chat_id

def delete_chat(chat_id: int, student_id: int) -> bool:
    """Удаляет чат (только если он принадлежит школьнику)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем, что чат принадлежит школьнику
    cursor.execute("SELECT id FROM chats WHERE id = ? AND student_id = ?", (chat_id, student_id))
    if not cursor.fetchone():
        conn.close()
        return False
    
    # Удаляем все сообщения чата
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    
    # Удаляем чат
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    
    conn.commit()
    conn.close()
    return True

def get_chat(chat_id: int) -> Optional[Dict[str, Any]]:
    """Получает информацию о чате"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_chat_messages(chat_id: int) -> List[Dict[str, Any]]:
    """Получает все сообщения чата"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM messages
        WHERE chat_id = ?
        ORDER BY created_at ASC
    """, (chat_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def add_message(chat_id: int, student_query: str, llm_response: str) -> int:
    """Добавляет сообщение в чат, возвращает ID сообщения"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO messages (chat_id, student_query, llm_response)
        VALUES (?, ?, ?)
    """, (chat_id, student_query, llm_response))
    
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return message_id

def rate_message(message_id: int, rating: int):
    """Оценивает сообщение (1-5)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE messages SET rating = ? WHERE id = ?
    """, (rating, message_id))
    
    # Если оценка 5, помечаем как лучший ответ и для проверки
    if rating == 5:
        cursor.execute("""
            UPDATE messages 
            SET is_best_answer = 1, needs_review = 1
            WHERE id = ?
        """, (message_id,))
        
        # Добавляем в таблицу лучших ответов
        cursor.execute("""
            INSERT INTO best_answers (message_id, rating)
            VALUES (?, ?)
        """, (message_id, rating))
    
    conn.commit()
    conn.close()

def add_practice_task(chat_id: int, original_message_id: int, task_text: str, correct_answer: str, complexity_level: int = 1) -> int:
    """Добавляет практическое задание, возвращает ID задания"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO practice_tasks (chat_id, original_message_id, task_text, correct_answer, complexity_level)
        VALUES (?, ?, ?, ?, ?)
    """, (chat_id, original_message_id, task_text, correct_answer, complexity_level))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return task_id

def get_pending_practice_task(chat_id: int) -> Optional[Dict[str, Any]]:
    """Получает незавершенное практическое задание для чата"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM practice_tasks
        WHERE chat_id = ? AND student_answer IS NULL
        ORDER BY created_at DESC
        LIMIT 1
    """, (chat_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def submit_practice_answer(task_id: int, student_answer: str, grade: int, feedback: str):
    """Сохраняет ответ ученика на практическое задание и оценку"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE practice_tasks
        SET student_answer = ?, grade = ?, feedback = ?, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (student_answer, grade, feedback, task_id))
    
    conn.commit()
    conn.close()

def get_next_complexity_level(chat_id: int) -> int:
    """Определяет следующий уровень сложности на основе последних оценок"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Получаем последние 3 оценки практических заданий
    cursor.execute("""
        SELECT grade, complexity_level FROM practice_tasks
        WHERE chat_id = ? AND grade IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 3
    """, (chat_id,))
    
    recent_grades = cursor.fetchall()
    conn.close()
    
    if not recent_grades:
        return 1  # Начинаем с уровня 1
    
    # Если последние оценки 4-5, увеличиваем сложность
    # Если 1-3, оставляем тот же уровень или уменьшаем
    avg_grade = sum(row['grade'] for row in recent_grades) / len(recent_grades)
    last_complexity = recent_grades[0]['complexity_level']
    
    if avg_grade >= 4.0 and last_complexity < 5:
        return min(last_complexity + 1, 5)
    elif avg_grade <= 2.0 and last_complexity > 1:
        return max(last_complexity - 1, 1)
    else:
        return last_complexity

def get_practice_tasks_for_student(student_id: int) -> List[Dict[str, Any]]:
    """Получает все практические задания школьника"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM practice_tasks
        WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
        AND student_answer IS NOT NULL
        ORDER BY completed_at DESC
    """, (student_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# === МЕТРИКИ ===

def get_student_metrics(student_id: int) -> Dict[str, Any]:
    """Получает метрики успеваемости школьника для отслеживания прогресса"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Средняя оценка (общая) - из сообщений и практических заданий
    cursor.execute("""
        SELECT AVG(rating) as avg_rating FROM (
            SELECT rating FROM messages
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND rating IS NOT NULL
            UNION ALL
            SELECT grade as rating FROM practice_tasks
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND grade IS NOT NULL
        )
    """, (student_id, student_id))
    avg_rating = cursor.fetchone()['avg_rating'] or 0
    
    # Средняя оценка за последние 7 дней
    cursor.execute("""
        SELECT AVG(rating) as avg_rating_recent FROM (
            SELECT rating FROM messages
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND rating IS NOT NULL
            AND created_at >= datetime('now', '-7 days')
            UNION ALL
            SELECT grade as rating FROM practice_tasks
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND grade IS NOT NULL
            AND completed_at >= datetime('now', '-7 days')
        )
    """, (student_id, student_id))
    avg_rating_recent = cursor.fetchone()['avg_rating_recent'] or 0
    
    # Средняя оценка за предыдущие 7 дней (для сравнения)
    cursor.execute("""
        SELECT AVG(rating) as avg_rating_previous FROM (
            SELECT rating FROM messages
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND rating IS NOT NULL
            AND created_at >= datetime('now', '-14 days')
            AND created_at < datetime('now', '-7 days')
            UNION ALL
            SELECT grade as rating FROM practice_tasks
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND grade IS NOT NULL
            AND completed_at >= datetime('now', '-14 days')
            AND completed_at < datetime('now', '-7 days')
        )
    """, (student_id, student_id))
    avg_rating_previous = cursor.fetchone()['avg_rating_previous'] or 0
    
    # Количество дней активности (за последние 30 дней)
    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(created_at)) as active_days
        FROM messages
        WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
        AND created_at >= datetime('now', '-30 days')
    """, (student_id,))
    active_days = cursor.fetchone()['active_days'] or 0
    
    # Количество дней с начала работы
    cursor.execute("""
        SELECT MIN(created_at) as first_message
        FROM messages
        WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
    """, (student_id,))
    first_message = cursor.fetchone()['first_message']
    
    days_since_start = 0
    if first_message:
        cursor.execute("""
            SELECT julianday('now') - julianday(?) as days
        """, (first_message,))
        days_since_start = int(cursor.fetchone()['days'] or 0)
    
    # Тенденция оценок (улучшение/ухудшение)
    rating_trend = 0
    if avg_rating_recent and avg_rating_previous:
        rating_trend = round(avg_rating_recent - avg_rating_previous, 2)
    
    # Активность по дням (последние 14 дней для графика)
    # Объединяем сообщения и практические задания
    cursor.execute("""
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as count,
            AVG(rating) as avg_rating_day
        FROM (
            SELECT created_at, rating FROM messages
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND created_at >= datetime('now', '-14 days')
            UNION ALL
            SELECT completed_at as created_at, grade as rating FROM practice_tasks
            WHERE chat_id IN (SELECT id FROM chats WHERE student_id = ?)
            AND completed_at >= datetime('now', '-14 days')
            AND completed_at IS NOT NULL
        )
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (student_id, student_id))
    daily_activity = []
    for row in cursor.fetchall():
        daily_activity.append({
            'date': row['date'],
            'count': row['count'],
            'avg_rating': round(row['avg_rating_day'] or 0, 2)
        })
    
    conn.close()
    
    return {
        'avg_rating': round(avg_rating, 2) if avg_rating else 0,
        'avg_rating_recent': round(avg_rating_recent, 2) if avg_rating_recent else 0,
        'rating_trend': rating_trend,
        'active_days': active_days,
        'days_since_start': days_since_start,
        'daily_activity': daily_activity
    }

# Инициализация БД при импорте
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
init_database()


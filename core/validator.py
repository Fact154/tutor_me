"""
Валидатор запросов пользователя для RAG системы
"""
import re
from typing import Dict, List, Tuple, Optional


class QueryValidator:
    """Валидатор запросов пользователя"""
    
    def __init__(self, chunks: List[Dict], doc_info: Dict):
        """
        Инициализация валидатора
        
        Args:
            chunks: Список чанков из учебника
            doc_info: Информация о документе
        """
        self.chunks = chunks
        self.doc_info = doc_info
        self.subject = doc_info.get('title', '').lower()
        
        # Извлекаем метаданные из chunks
        self.available_pages = self._extract_pages()
        self.available_exercises = self._extract_exercises()
        self.topics = self._extract_topics()
    
    def _extract_pages(self) -> set:
        """Извлекает все доступные страницы из chunks"""
        pages = set()
        for chunk in self.chunks:
            pages_str = chunk.get('pages', '')
            # Парсим "3-4", "5-6" и т.д.
            if '-' in pages_str:
                try:
                    start, end = map(int, pages_str.split('-'))
                    pages.update(range(start, end + 1))
                except ValueError:
                    continue
            elif pages_str.isdigit():
                pages.add(int(pages_str))
        return pages
    
    def _extract_exercises(self) -> Dict[int, List[int]]:
        """Извлекает доступные задания по страницам"""
        exercises = {}
        for chunk in self.chunks:
            pages_str = chunk.get('pages', '')
            chunk_exercises = chunk.get('exercises', [])
            
            if not chunk_exercises:
                continue
            
            if '-' in pages_str:
                try:
                    start, end = map(int, pages_str.split('-'))
                    for page in range(start, end + 1):
                        if page not in exercises:
                            exercises[page] = []
                        exercises[page].extend(chunk_exercises)
                except ValueError:
                    continue
            elif pages_str.isdigit():
                page = int(pages_str)
                if page not in exercises:
                    exercises[page] = []
                exercises[page].extend(chunk_exercises)
        
        # Удаляем дубликаты и сортируем
        for page in exercises:
            exercises[page] = sorted(list(set(exercises[page])))
        
        return exercises
    
    def _extract_topics(self) -> List[str]:
        """Извлекает все темы из учебника"""
        topics = []
        for chunk in self.chunks:
            if 'topic' in chunk:
                topics.append(chunk['topic'].lower())
            if 'keywords' in chunk:
                topics.extend([kw.lower() for kw in chunk['keywords']])
        return topics
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Валидирует запрос пользователя
        
        Блокируем ТОЛЬКО:
        1. Несуществующие страницы/задания (если явно указаны)
        2. Явно офтопик темы (другие предметы)
        
        Пропускаем:
        - Общие вопросы ("не понял", "объясни")
        - Вопросы по теории
        - Уточняющие вопросы
        
        Args:
            query: Запрос пользователя
            
        Returns:
            (is_valid, error_message) - True если запрос валиден, иначе False с сообщением об ошибке
        """
        query_lower = query.lower()
        
        # ✅ ПРОПУСКАЕМ: Общие фразы (просьбы о помощи)
        help_phrases = [
            'не понял', 'не понятно', 'объясни', 'помоги', 
            'как решить', 'что значит', 'расскажи', 'покажи',
            'не получается', 'не понимаю', 'сложно', 'трудно',
            'помощь', 'подскажи', 'разъясни', 'поясни'
        ]
        if any(phrase in query_lower for phrase in help_phrases):
            return True, None  # Пропускаем без проверок
        
        # ✅ ПРОПУСКАЕМ: Вопросы по теории (без конкретных заданий)
        theory_keywords = [
            'что такое', 'как называется', 'определение',
            'правило', 'формула', 'свойство', 'теорема',
            'понятие', 'термин', 'значение'
        ]
        if any(keyword in query_lower for keyword in theory_keywords):
            return True, None
        
        # 🔍 ПРОВЕРЯЕМ: Только если явно указаны страница/задание
        task_match = re.search(r'задан[иеья]+\s*[№#]?\s*(\d+)', query_lower)
        page_match = re.search(r'страниц[аеыу]+\s*[№#]?\s*(\d+)', query_lower)
        
        # Если НЕ указаны страница/задание - пропускаем
        if not task_match and not page_match:
            return True, None
        
        # Если указаны - проверяем существование
        if task_match and page_match:
            task_num = int(task_match.group(1))
            page_num = int(page_match.group(1))
            
            # Проверяем страницу
            if self.available_pages and page_num not in self.available_pages:
                max_page = max(self.available_pages)
                min_page = min(self.available_pages)
                return False, (
                    f"[!] Страница {page_num} не найдена в учебнике.\n\n"
                    f"Доступные страницы: {min_page}-{max_page}\n\n"
                    f"Попробуйте указать другую страницу."
                )
            
            # Проверяем задание на странице
            if page_num in self.available_exercises:
                if task_num not in self.available_exercises[page_num]:
                    available = ', '.join(map(str, self.available_exercises[page_num]))
                    return False, (
                        f"[!] Задание {task_num} не найдено на странице {page_num}.\n\n"
                        f"Доступные задания на этой странице: {available}\n\n"
                        f"Попробуйте выбрать одно из них."
                    )
        
        elif page_match:
            page_num = int(page_match.group(1))
            if self.available_pages and page_num not in self.available_pages:
                max_page = max(self.available_pages)
                min_page = min(self.available_pages)
                return False, (
                    f"[!] Страница {page_num} не найдена в учебнике.\n\n"
                    f"Доступные страницы: {min_page}-{max_page}"
                )
        
        # ⚠️ ПРОВЕРЯЕМ: Явный офтопик (только очень явные случаи)
        if 'математика' in self.subject:
            # Блокируем только явные запросы по другим предметам
            offtopic_subjects = [
                'война', 'битва', 'революция',  # История
                'стихотворение', 'поэма', 'роман', 'писатель',  # Литература
                'клетка', 'организм', 'животное', 'растение',  # Биология
                'материк', 'океан', 'страна', 'столица',  # География
                'атом', 'молекула', 'реакция', 'элемент',  # Химия/Физика
                'english', 'английский язык', 'перевод'  # Английский
            ]
            
            # Проверяем только если есть явное упоминание другого предмета
            # И НЕТ математических терминов
            math_terms = ['число', 'задач', 'решени', 'вычисл', 'формул', 'уравнен']
            has_math_terms = any(term in query_lower for term in math_terms)
            
            if not has_math_terms:
                for offtopic in offtopic_subjects:
                    if offtopic in query_lower:
                        return False, (
                            f"[!] Ваш вопрос не относится к математике.\n\n"
                            f"Пожалуйста, задайте вопрос по учебнику математики."
                        )
        
        # Все проверки пройдены
        return True, None
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по доступным данным"""
        return {
            'total_pages': len(self.available_pages),
            'page_range': f"{min(self.available_pages)}-{max(self.available_pages)}" if self.available_pages else "N/A",
            'total_exercises': sum(len(exs) for exs in self.available_exercises.values()),
            'pages_with_exercises': len(self.available_exercises),
            'topics_count': len(set(self.topics))
        }


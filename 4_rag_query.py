#!/usr/bin/env python3
"""
RAG система для ответов на вопросы
"""

import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from config import Config
from typing import List, Dict

class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(Config.DB_DIR))
        self.ollama_model = Config.OLLAMA_MODEL
    
    def search_relevant_chunks(
        self,
        query: str,
        subject: str,
        grade: int,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Ищет релевантные chunks для запроса
        """
        # Получаем коллекцию
        collection_name = f"{Config.CHROMA_COLLECTION_PREFIX}_{subject}_{grade}"
        
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            raise ValueError(f"Коллекция '{collection_name}' не найдена. Запустите 3_create_embeddings.py")
        
        # Создаём embedding для запроса
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Поиск
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Форматируем результаты
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return chunks
    
    def create_prompt(self, query: str, chunks: List[Dict], grade: int, subject: str) -> str:
        """
        Создаёт промпт для LLM
        """
        # Контекст из chunks
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            context_parts.append(
                f"[Источник {idx}]\n"
                f"Учебник: {metadata.get('textbook_title', 'Неизвестно')}\n"
                f"Страница: {metadata.get('page', '?')}\n"
                f"Содержание:\n{chunk['document']}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""Ты — репетитор для школьника {grade} класса по предмету "{subject}".

КОНТЕКСТ ИЗ УЧЕБНИКА:
{context}

ВОПРОС УЧЕНИКА:
{query}

ПРАВИЛА ОТВЕТА:
1. Используй ТОЛЬКО информацию из предоставленного контекста
2. Объясняй на уровне {grade} класса
3. Приводи примеры из учебника, если есть
4. Если в контексте нет ответа, честно скажи об этом
5. Укажи источник (страницу учебника)
6. Объясняй пошагово

ОТВЕТ:"""
        
        return prompt
    
    def answer_question(
        self,
        query: str,
        subject: str,
        grade: int,
        n_results: int = 3,
        verbose: bool = False
    ) -> Dict:
        """
        Полный цикл: поиск + генерация ответа
        """
        # 1. Поиск релевантных chunks
        if verbose:
            print(f"\n🔍 Поиск информации в учебнике ({subject}, {grade} класс)...")
        
        chunks = self.search_relevant_chunks(query, subject, grade, n_results)
        
        if verbose:
            print(f"✓ Найдено {len(chunks)} релевантных фрагментов:")
            for idx, chunk in enumerate(chunks, 1):
                print(f"  {idx}. {chunk['metadata'].get('textbook_title', 'Неизвестно')}, "
                      f"стр. {chunk['metadata'].get('page', '?')} "
                      f"(релевантность: {1 - chunk['distance']:.2f})")
        
        # 2. Создание промпта
        prompt = self.create_prompt(query, chunks, grade, subject)
        
        if verbose:
            print(f"\n💬 Генерация ответа (модель: {self.ollama_model})...")
        
        # 3. Генерация ответа через Ollama
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt
            )
            
            answer = response['response']
            
        except Exception as e:
            answer = f"Ошибка при генерации ответа: {str(e)}"
        
        # 4. Формируем результат
        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'textbook': chunk['metadata'].get('textbook_title', 'Неизвестно'),
                    'page': chunk['metadata'].get('page', '?'),
                    'relevance': 1 - chunk['distance']
                }
                for chunk in chunks
            ],
            'metadata': {
                'subject': subject,
                'grade': grade,
                'model': self.ollama_model,
                'chunks_used': len(chunks)
            }
        }
        
        return result


def interactive_mode():
    """
    Интерактивный режим для тестирования
    """
    print("=== RAG Система для учебников ===")
    print("Доступные предметы: математика, история")
    print("Доступные классы: 5, 6, 7")
    print("Для выхода введите 'exit'\n")
    
    rag = RAGSystem()
    
    # Запрашиваем параметры
    subject = input("Предмет: ").strip().lower()
    grade = int(input("Класс: ").strip())
    
    print(f"\n✓ Работаем с: {subject}, {grade} класс")
    print("Задавайте вопросы по учебнику:\n")
    
    while True:
        query = input(f"\n[{subject} {grade}] Вопрос: ").strip()
        
        if query.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
        
        if not query:
            continue
        
        try:
            result = rag.answer_question(
                query=query,
                subject=subject,
                grade=grade,
                verbose=True
            )
            
            print(f"\n{'='*60}")
            print("ОТВЕТ:")
            print(f"{'='*60}")
            print(result['answer'])
            print(f"\n{'='*60}")
            print("ИСТОЧНИКИ:")
            for idx, source in enumerate(result['sources'], 1):
                print(f"  {idx}. {source['textbook']}, стр. {source['page']} "
                      f"(релевантность: {source['relevance']:.2%})")
            print(f"{'='*60}")
            
        except ValueError as e:
            print(f"\n❌ Ошибка: {e}")
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка: {e}")


def test_query(subject: str, grade: int, query: str):
    """
    Тестовый запрос (для отладки)
    """
    rag = RAGSystem()
    
    result = rag.answer_question(
        query=query,
        subject=subject,
        grade=grade,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТ ЗАПРОСА")
    print(f"{'='*60}")
    print(f"Вопрос: {result['query']}")
    print(f"\nОтвет:\n{result['answer']}")
    print(f"\nИсточники:")
    for source in result['sources']:
        print(f"  - {source['textbook']}, стр. {source['page']}")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Режим тестового запроса
        # Пример: python 4_rag_query.py математика 5 "Как сложить дроби?"
        if len(sys.argv) >= 4:
            subject = sys.argv[1]
            grade = int(sys.argv[2])
            query = " ".join(sys.argv[3:])
            test_query(subject, grade, query)
        else:
            print("Использование: python 4_rag_query.py <предмет> <класс> <вопрос>")
            print("Пример: python 4_rag_query.py математика 5 'Как сложить дроби?'")
    else:
        # Интерактивный режим
        interactive_mode()
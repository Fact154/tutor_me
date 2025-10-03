#!/usr/bin/env python3
"""
Создание embeddings и загрузка в ChromaDB
"""

import json
from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import Config

class EmbeddingManager:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        print(f"Загрузка модели embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Инициализация ChromaDB
        self.client = chromadb.PersistentClient(path=str(Config.DB_DIR))
    
    def create_text_for_embedding(self, chunk: Dict) -> str:
        """
        Создаёт текстовое представление chunk'а для embedding
        """
        metadata = chunk['metadata']
        content = chunk['content']
        
        text_parts = [
            f"Предмет: {metadata['subject']}",
            f"Класс: {metadata['grade']}",
            f"Страница: {metadata['page']}",
        ]
        
        if 'topic' in metadata:
            text_parts.append(f"Тема: {metadata['topic']}")
        
        if 'chapter' in metadata:
            text_parts.append(f"Глава: {metadata['chapter']}")
        
        # Добавляем содержимое
        text_parts.append(f"Содержание: {content['text'][:500]}")  # Первые 500 символов
        
        # Для математики добавляем формулы
        if 'formulas' in content and content['formulas']:
            text_parts.append(f"Формулы: {', '.join(content['formulas'][:3])}")
        
        # Для истории добавляем даты и личности
        if 'dates' in metadata and metadata['dates']:
            text_parts.append(f"Даты: {', '.join(metadata['dates'][:5])}")
        
        if 'historical_figures' in metadata and metadata['historical_figures']:
            text_parts.append(f"Исторические личности: {', '.join(metadata['historical_figures'][:3])}")
        
        return '\n'.join(text_parts)
    
    def create_embeddings_for_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        """
        Создаёт embeddings для списка chunks
        """
        texts = [self.create_text_for_embedding(chunk) for chunk in chunks]
        
        print(f"Создание embeddings для {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        return embeddings.tolist()
    
    def store_in_chromadb(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        collection_name: str
    ):
        """
        Сохраняет chunks с embeddings в ChromaDB
        """
        # Создаём или получаем коллекцию
        try:
            collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' уже существует. Добавление данных...")
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"Textbook collection: {collection_name}"}
            )
            print(f"Создана новая коллекция: '{collection_name}'")
        
        # Подготовка данных
        ids = [chunk['chunk_id'] for chunk in chunks]
        documents = [self.create_text_for_embedding(chunk) for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Загрузка батчами
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Загрузка в ChromaDB"):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        
        print(f"✓ Загружено {len(ids)} chunks в коллекцию '{collection_name}'")

def process_chunks_file(chunks_file: Path, embedding_manager: EmbeddingManager):
    """
    Обрабатывает файл с chunks
    """
    print(f"\nОбработка файла: {chunks_file.name}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    metadata = data['metadata']
    
    # Название коллекции
    collection_name = f"{Config.CHROMA_COLLECTION_PREFIX}_{metadata['subject']}_{metadata['grade']}"
    
    # Создаём embeddings
    embeddings = embedding_manager.create_embeddings_for_chunks(chunks)
    
    # Сохраняем в ChromaDB
    embedding_manager.store_in_chromadb(chunks, embeddings, collection_name)

if __name__ == "__main__":
    embedding_manager = EmbeddingManager()
    
    # Обрабатываем все файлы chunks
    chunks_files = list(Config.STRUCTURED_DIR.glob("*_chunks.json"))
    
    if not chunks_files:
        print("Нет файлов chunks для обработки!")
        print("Сначала запустите 2_structure_data.py")
        exit(1)
    
    for chunks_file in chunks_files:
        process_chunks_file(chunks_file, embedding_manager)
    
    print("\n✓ Все данные загружены в ChromaDB")
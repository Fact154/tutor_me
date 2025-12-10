#!/usr/bin/env python3
"""
Тестирование валидатора запросов
"""
from core.validator import QueryValidator
from core import rag

# Загружаем данные
print("Загрузка данных учебника...")
chunks_data, error, doc_info = rag.load_chunks()

if not chunks_data:
    print(f"Ошибка загрузки: {error}")
    exit(1)

print(f"Загружено {len(chunks_data)} чанков")
print(f"Учебник: {doc_info.get('title', 'Неизвестно')}")
print()

# Создаем валидатор
validator = QueryValidator(chunks_data, doc_info)

# Получаем статистику
stats = validator.get_statistics()
print("=== Статистика ===")
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

# Тестовые запросы
test_queries = [
    # Валидные запросы (должны пройти)
    ("Я не понял задание", True),
    ("Помоги решить", True),
    ("Что такое дроби?", True),
    ("Объясни подробнее", True),
    ("Задание 5 на странице 6", True),  # Существует
    
    # Невалидные запросы (должны быть заблокированы)
    ("Задание 1 на странице 999", False),  # Несуществующая страница
    ("Задание 99 на странице 5", False),   # Несуществующее задание
    ("Расскажи про Вторую мировую войну", False),  # Офтопик
]

print("=== Тестирование ===")
passed = 0
failed = 0

for query, expected_valid in test_queries:
    is_valid, error_message = validator.validate_query(query)
    
    status = "✓" if is_valid == expected_valid else "✗"
    result = "PASS" if is_valid == expected_valid else "FAIL"
    
    print(f"{status} {result}: '{query}'")
    print(f"   Ожидалось: {'валиден' if expected_valid else 'невалиден'}")
    print(f"   Получено: {'валиден' if is_valid else 'невалиден'}")
    
    if not is_valid and error_message:
        print(f"   Сообщение: {error_message[:100]}...")
    
    print()
    
    if is_valid == expected_valid:
        passed += 1
    else:
        failed += 1

print("=== Результаты ===")
print(f"Пройдено: {passed}/{len(test_queries)}")
print(f"Провалено: {failed}/{len(test_queries)}")

if failed == 0:
    print("\n✓ Все тесты пройдены успешно!")
else:
    print(f"\n✗ Провалено {failed} тестов")


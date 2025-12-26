"""
Скрипт для обучения NLP модели.
Запускается отдельно при изменении данных интентов.
"""
import sys
sys.path.append('.')

from app.bot.nlp_engine import NLPEngine
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser(description='Обучение NLP модели для чат-бота')
    parser.add_argument('--data', default='app/data/intents.json',
                       help='Путь к файлу с интентами')
    parser.add_argument('--output', default='app/data/models/nlp_model.joblib',
                       help='Путь для сохранения модели')
    
    args = parser.parse_args()
    
    print(f"Загрузка данных из {args.data}")
    nlp = NLPEngine(args.data)
    
    print("Обучение модели...")
    nlp.train()
    
    print(f"Сохранение модели в {args.output}")
    nlp.save_model(args.output)
    
    print("Обучение завершено!")
    print(f"Количество интентов: {len(nlp.responses)}")
    print(f"Примеры интентов: {list(nlp.responses.keys())[:5]}")

if __name__ == "__main__":
    main()
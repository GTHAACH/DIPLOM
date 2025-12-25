import json
import numpy as np
from typing import Dict, List, Tuple
import nltk
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NLPEngine:
    """Класс для обработки естественного языка и классификации намерений"""
    
    def __init__(self, intents_path: str):
        self.stemmer = SnowballStemmer("russian")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=0.001,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        self.classifier = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.classifier)
        self.intents = self._load_intents(intents_path)
        self.tags = []
        self.responses = {}
        
    def _load_intents(self, path: str) -> Dict:
        """Загрузка интентов из JSON файла"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Подготовка данных для обучения
        samples = []
        self.tags = []
        
        for intent in data['intents']:
            tag = intent['tag']
            self.responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                samples.append(pattern)
                self.tags.append(tag)
        
        return {
            'samples': samples,
            'tags': self.tags,
            'full_data': data['intents']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста: приведение к нижнему регистру, стемминг"""
        tokens = nltk.word_tokenize(text.lower())
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens 
                         if token.isalnum()]
        return ' '.join(stemmed_tokens)
    
    def train(self):
        """Обучение модели классификации намерений"""
        print("Обучение модели NLP...")
        
        # Предобработка всех примеров
        processed_samples = [
            self.preprocess_text(sample) 
            for sample in self.intents['samples']
        ]
        
        # Обучение пайплайна
        self.pipeline.fit(processed_samples, self.intents['tags'])
        print(f"Модель обучена. Количество классов: {len(set(self.intents['tags']))}")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Предсказание намерения пользователя"""
        processed_text = self.preprocess_text(text)
        
        try:
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            max_prob_idx = np.argmax(probabilities)
            predicted_tag = self.pipeline.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            return predicted_tag, confidence
        except:
            # Если что-то пошло не так (например, слово не из словаря)
            return "unknown", 0.0
    
    def get_response(self, intent_tag: str) -> str:
        """Получение случайного ответа для определенного интента"""
        if intent_tag in self.responses:
            import random
            return random.choice(self.responses[intent_tag])
        return "Извините, я не совсем понимаю. Можете переформулировать?"
    
    def save_model(self, path: str):
        """Сохранение обученной модели"""
        joblib.dump(self.pipeline, path)
        print(f"Модель сохранена в {path}")
    
    def load_model(self, path: str):
        """Загрузка обученной модели"""
        if os.path.exists(path):
            self.pipeline = joblib.load(path)
            print(f"Модель загружена из {path}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.bot.nlp_engine import NLPEngine
from app.bot.core import ChatBotCore
from app.config import settings

app = FastAPI(
    title="Bank ChatBot API",
    description="API для чат-бота банка с NLP",
    version="1.0.0"
)

# Инициализация компонентов
nlp_engine = NLPEngine("app/data/intents.json")
# Обучение модели при старте (в реальности - отдельный процесс)
nlp_engine.train()
bot_core = ChatBotCore(nlp_engine)

class MessageRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None

@app.post("/chat", response_model=MessageResponse)
async def process_chat_message(request: MessageRequest):
    """
    Обработка сообщения от пользователя.
    
    - **user_id**: уникальный идентификатор пользователя
    - **message**: текстовое сообщение от пользователя
    - **session_id**: идентификатор сессии (опционально)
    """
    try:
        # Обработка сообщения через ядро бота
        response_text = bot_core.process_message(request.user_id, request.message)
        
        # Получаем информацию о текущем интенте
        session = bot_core.get_or_create_session(request.user_id)
        intent_info = None
        confidence_info = None
        
        if session.current_intent:
            intent, confidence = nlp_engine.predict_intent(request.message)
            intent_info = intent
            confidence_info = confidence
        
        return MessageResponse(
            response=response_text,
            session_id=request.user_id,  # В реальности - уникальный ID сессии
            intent=intent_info,
            confidence=confidence_info
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки сообщения: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "service": "bank_chatbot",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/intents")
async def get_available_intents():
    """Получение списка поддерживаемых интентов"""
    return {
        "intents": list(nlp_engine.responses.keys()),
        "count": len(nlp_engine.responses)
    }

if __name__ == "__main__":
    # Запуск сервера для разработки
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
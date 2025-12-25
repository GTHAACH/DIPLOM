from typing import Dict, Optional
from enum import Enum
import datetime

class DialogState(Enum):
    """Состояния диалога (State Machine)"""
    START = "start"
    AUTHENTICATION = "authentication"
    MAIN_MENU = "main_menu"
    PROCESSING_REQUEST = "processing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    END = "end"

class UserSession:
    """Класс для хранения состояния сессии пользователя"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state = DialogState.START
        self.context = {}
        self.last_activity = datetime.datetime.now()
        self.auth_attempts = 0
        self.current_intent = None
    
    def update_state(self, new_state: DialogState):
        """Обновление состояния диалога"""
        self.state = new_state
        self.last_activity = datetime.datetime.now()
    
    def add_to_context(self, key: str, value):
        """Добавление данных в контекст"""
        self.context[key] = value
    
    def is_authenticated(self) -> bool:
        """Проверка аутентификации пользователя"""
        return self.context.get('authenticated', False)

class ChatBotCore:
    """Основной класс управления диалогом"""
    
    def __init__(self, nlp_engine):
        self.nlp = nlp_engine
        self.sessions: Dict[str, UserSession] = {}
        # Правила переходов между состояниями
        self.state_transitions = {
            DialogState.START: self._handle_start,
            DialogState.AUTHENTICATION: self._handle_auth,
            DialogState.MAIN_MENU: self._handle_main_menu,
            DialogState.PROCESSING_REQUEST: self._handle_processing
        }
    
    def get_or_create_session(self, user_id: str) -> UserSession:
        """Получение или создание сессии пользователя"""
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id)
        return self.sessions[user_id]
    
    def process_message(self, user_id: str, message: str) -> str:
        """Основной метод обработки входящего сообщения"""
        session = self.get_or_create_session(user_id)
        
        # Очистка старых сессий (простой таймаут)
        self._cleanup_old_sessions()
        
        # Получаем обработчик текущего состояния
        handler = self.state_transitions.get(session.state)
        if handler:
            return handler(session, message)
        
        return "Произошла ошибка. Пожалуйста, начните заново."
    
    def _handle_start(self, session: UserSession, message: str) -> str:
        """Обработка начального состояния"""
        session.update_state(DialogState.AUTHENTICATION)
        return ("Добро пожаловать в чат-бот банка 'Финанс'! "
                "Для начала введите ваш ID клиента (6 цифр):")
    
    def _handle_auth(self, session: UserSession, message: str) -> str:
        """Обработка аутентификации"""
        if message.isdigit() and len(message) == 6:
            # Здесь должна быть проверка через API банка
            # Сейчас - заглушка
            if self._check_client_id(message):
                session.add_to_context('client_id', message)
                session.add_to_context('authenticated', True)
                session.update_state(DialogState.MAIN_MENU)
                return (f"Добрый день, клиент {message}! "
                        "Чем могу помочь?\n"
                        "• Узнать баланс\n"
                        "• Заблокировать карту\n"
                        "• Курс валют\n"
                        "• Контакты отделений")
            else:
                session.auth_attempts += 1
                if session.auth_attempts >= 3:
                    session.update_state(DialogState.END)
                    return "Слишком много попыток. Обратитесь в отделение."
                return "Неверный ID. Попробуйте еще раз:"
        return "ID должен состоять из 6 цифр. Попробуйте еще раз:"
    
    def _handle_main_menu(self, session: UserSession, message: str) -> str:
        """Обработка основного меню с помощью NLP"""
        if not session.is_authenticated():
            session.update_state(DialogState.AUTHENTICATION)
            return "Требуется аутентификация. Введите ваш ID:"
        
        # Используем NLP для определения намерения
        intent, confidence = self.nlp.predict_intent(message)
        session.current_intent = intent
        
        if confidence < 0.4:  # Низкая уверенность
            return "Уточните, пожалуйста, что вы хотите сделать?"
        
        # Обработка конкретных интентов
        if intent == "balance_inquiry":
            # Здесь интеграция с API банка
            balance = self._get_balance_from_api(session.context['client_id'])
            return f"На вашем счете: {balance} руб."
        
        elif intent == "card_block":
            session.update_state(DialogState.WAITING_CONFIRMATION)
            return "Вы уверены, что хотите заблокировать карту? (да/нет)"
        
        elif intent == "exchange_rate":
            rates = self._get_exchange_rates()
            return f"Курс ЦБ на сегодня:\n{rates}"
        
        # Стандартный ответ из NLP
        return self.nlp.get_response(intent)
    
    def _handle_processing(self, session: UserSession, message: str) -> str:
        """Обработка состояния выполнения операции"""
        # Заглушка для демонстрации
        session.update_state(DialogState.MAIN_MENU)
        return "Операция выполнена успешно! Что еще вас интересует?"
    
    def _check_client_id(self, client_id: str) -> bool:
        """Заглушка проверки ID (в реальности - запрос к API)"""
        # В реальном проекте здесь будет HTTP-запрос к системе банка
        return len(client_id) == 6 and client_id.isdigit()
    
    def _get_balance_from_api(self, client_id: str) -> float:
        """Заглушка получения баланса"""
        # В реальном проекте - интеграция с Core Banking System
        return 45678.50
    
    def _get_exchange_rates(self) -> str:
        """Заглушка получения курсов валют"""
        return "USD: 90.50 руб.\nEUR: 98.75 руб.\nCNY: 12.45 руб."
    
    def _cleanup_old_sessions(self, timeout_minutes: int = 30):
        """Очистка старых сессий"""
        now = datetime.datetime.now()
        to_remove = []
        
        for user_id, session in self.sessions.items():
            delta = now - session.last_activity
            if delta.total_seconds() > timeout_minutes * 60:
                to_remove.append(user_id)
        
        for user_id in to_remove:
            del self.sessions[user_id]
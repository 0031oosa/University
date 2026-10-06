# Agro Scoring API

REST API на FastAPI для оценки риска сельскохозяйственных предприятий.
Лабораторная работа №1 по дисциплине «Современные интеллектуальные сетевые сервисы».

## Запуск

```bash
python -m venv venv
venv\Scripts\activate     
pip install -r requirements.txt
uvicorn main:app --reload
```

После запуска:
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json

## Endpoints

| Метод | Путь | Назначение |
|---|---|---|
| GET | /health | Проверка работоспособности API |
| GET | /model-info | Информация о модели |
| POST | /predict | Оценка риска хозяйства |
| GET | /predictions | Список прогнозов (`limit`, `risk_level`) |
| GET | /predictions/{request_id} | Прогноз по ID |

## Структура

```
main.py       # endpoints, middleware
schemas.py    # Pydantic-модели
model.py      # расчёт риска (заглушка вместо ML-модели)
services.py   # бизнес-логика и постпроцессинг
storage.py    # хранилище прогнозов в памяти
```

## Пример запроса

```json
{
  "farm_id": "FARM-001",
  "region": "Krasnodar",
  "crop_type": "wheat",
  "area_ha": 2500,
  "temperature_avg": 24.3,
  "precipitation_mm": 320,
  "payment_delay_days": 45,
  "previous_defaults": 1,
  "debt": 6500000
}
```

Допустимые регионы: `Krasnodar`, `Rostov`, `Stavropol`.
Для проверки ответа 503 установите `MODEL_READY = False` в `model.py`.
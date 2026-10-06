import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status

import agro_api.model as model
import agro_api.services as services
import agro_api.storage as storage
from agro_api.schemas import (
    FarmRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="Agro Scoring API",
    description="REST API для оценки риска сельскохозяйственных предприятий.",
    version="1.0.0",
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.perf_counter() - start, 6))
    return response


@app.get("/health", response_model=HealthResponse,
         summary="Проверка состояния API",
         description="Используется для проверки того, что REST API запущен и отвечает.")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse,
         summary="Информация о модели",
         description="Возвращает название, версию, тип и текущее состояние модели.")
def model_info():
    return {
        "model_name": model.MODEL_NAME,
        "model_version": model.MODEL_VERSION,
        "model_type": model.MODEL_TYPE,
        "status": "ready" if model.MODEL_READY else "unavailable",
    }


@app.post("/predict", response_model=PredictionResponse,
          status_code=status.HTTP_200_OK,
          summary="Оценить риск хозяйства",
          description="Принимает характеристики хозяйства, выполняет валидацию, "
                      "инференс модели и возвращает оценку риска.")
def predict(request: FarmRequest):
    if not model.MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable",
        )
    if request.region not in services.ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown region: {request.region}. "
                   f"Allowed regions: {sorted(services.ALLOWED_REGIONS)}",
        )
    return services.make_prediction(request)


# Этот маршрут должен идти ДО /predictions/{request_id}
@app.get("/predictions", response_model=List[PredictionResponse],
         summary="Получить список прогнозов",
         description="Возвращает список выполненных прогнозов. Поддерживает "
                     "ограничение количества и фильтрацию по уровню риска.")
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100,
                       description="Максимальное количество результатов"),
    risk_level: Optional[str] = Query(default=None,
                                      description="Фильтр: low, medium или high"),
):
    if risk_level is not None and risk_level not in services.ALLOWED_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="risk_level must be 'low', 'medium' or 'high'",
        )
    return storage.list_all(risk_level, limit)


@app.get("/predictions/{request_id}", response_model=PredictionResponse,
         summary="Получить прогноз по request_id",
         description="Возвращает сохранённый прогноз по его уникальному идентификатору.")
def get_prediction(request_id: str):
    result = storage.get(request_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
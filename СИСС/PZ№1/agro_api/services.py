import logging
import uuid

import agro_api.model as model
import agro_api.storage as storage
from agro_api.schemas import FarmRequest

logger = logging.getLogger(__name__)

ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
ALLOWED_LEVELS = {"low", "medium", "high"}


def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: str) -> str:
    if level == "low":
        return "Стандартное рассмотрение"
    if level == "medium":
        return "Требуется дополнительная проверка"
    return "Высокий риск. Требуется ручное рассмотрение"


def make_prediction(data: FarmRequest) -> dict:
    """Инференс -> постпроцессинг -> сохранение результата."""
    logger.info("Prediction request received | farm_id=%s", data.farm_id)

    score = model.calculate_risk(data)
    level = get_risk_level(score)
    request_id = str(uuid.uuid4())

    result = {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": model.MODEL_VERSION,
    }
    storage.save(request_id, result)

    logger.info(
        "Prediction completed | request_id=%s | farm_id=%s | risk_score=%s | risk_level=%s",
        request_id, data.farm_id, score, level,
    )
    return result
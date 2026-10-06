from typing import Optional

# Временное хранилище в памяти (позже можно заменить на SQLite/PostgreSQL)
predictions: dict[str, dict] = {}


def save(request_id: str, result: dict) -> None:
    predictions[request_id] = result


def get(request_id: str) -> Optional[dict]:
    return predictions.get(request_id)


def list_all(risk_level: Optional[str] = None, limit: int = 10) -> list[dict]:
    values = list(predictions.values())
    if risk_level is not None:
        values = [v for v in values if v["risk_level"] == risk_level]
    return values[:limit]
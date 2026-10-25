# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №2
# ============================================================

import asyncio
import logging
import math
import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiosqlite
import httpx

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ============================================================
# 1. ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. КОНСТАНТЫ МОДЕЛИ
# ============================================================

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

# Имитация состояния модели.
# Если False, endpoints инференса возвращаем HTTP 503.
MODEL_READY = True


# ============================================================
# LLM / VIREONIX
# ============================================================

# Vireonix предоставляет OpenAI-совместимый endpoint
# без API-ключа и авторизации.
LLM_BASE_URL = "https://vireonix.ai/v1"
LLM_MODEL = "auto"

# Таймаут 
LLM_TIMEOUT_SECONDS = 60.0

# Ограничения 
LLM_DEFAULT_MAX_TOKENS = 300
LLM_DEFAULT_TEMPERATURE = 0.3


# ============================================================
# 3. НАСТРОЙКА BATCHING / QUEUE
# ============================================================

# Максимальное число объектов в одном batch.
MAX_BATCH_SIZE = 8

# Максимальное время ожидания накопления dynamic batch.
MAX_BATCH_WAIT_MS = 50

# Максимальное количество запросов в очереди.
QUEUE_MAX_SIZE = 100

# Максимальное время ожидания результата инференса.
INFERENCE_TIMEOUT_SECONDS = 2.0


# ============================================================
# 4. ИМИТАЦИЯ ВРЕМЕНИ РАБОТЫ МОДЕЛИ
# ============================================================

# Предположим:
# 20 ms — фиксированный overhead одного вызова модели;
# 3 ms — вычисления на один объект.

MODEL_FIXED_OVERHEAD_SECONDS = 0.020
MODEL_PER_ITEM_SECONDS = 0.003


# ============================================================
# 5. НАСТРОЙКИ SQLITE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "agro_scoring.db"

# Если SQLite временно занята другой транзакцией,
# соединение будет ожидать до 5 секунд.
DB_BUSY_TIMEOUT_MS = 5000


# ============================================================
# 6. БИЗНЕС-СПРАВОЧНИКИ
# ============================================================

ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol",
}


# SQLite strftime('%w', ...) возвращает:
# 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
}


# ============================================================
# 7. PYDANTIC-МОДЕЛИ
# ============================================================

class FarmRequest(BaseModel):
    """Входные данные для оценки риска одного хозяйства."""

    farm_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор хозяйства",
    )

    region: str = Field(
        ...,
        min_length=1,
        description="Регион",
    )

    crop_type: str = Field(
        ...,
        min_length=1,
        description="Основная сельскохозяйственная культура",
    )

    area_ha: float = Field(
        ...,
        gt=0,
        description="Площадь посевов, га",
    )

    temperature_avg: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Средняя температура, °C",
    )

    precipitation_mm: float = Field(
        ...,
        ge=0,
        description="Количество осадков, мм",
    )

    payment_delay_days: int = Field(
        ...,
        ge=0,
        description="Количество дней просрочки платежа",
    )

    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Количество предыдущих дефолтов",
    )

    debt: float = Field(
        ...,
        ge=0,
        description="Текущая задолженность",
    )


class PredictionResponse(BaseModel):
    """Ответ для одного прогноза."""

    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


# ============================================================
# 8. МОДЕЛИ ДЛЯ ЯВНОГО BATCH REQUEST
# ============================================================

class BatchRequest(BaseModel):
    """Явный batch, который клиент отправляет одним HTTP-запросом."""

    items: List[FarmRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Список объектов для пакетного инференса",
    )


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: List[BatchItemResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: Optional[str]
    prev_page: Optional[str]


class PaginatedPredictionsResponse(BaseModel):
    data: List[PredictionResponse]
    meta: PaginationMeta



# ============================================================
# МОДЕЛИ ДЛЯ LLM
# ============================================================

class LLMChatRequest(BaseModel):
    """
    Запрос к внешней LLM.

    Не передавайте сюда реальные
    конфиденциальные банковские или персональные данные.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст запроса к LLM",
    )

    model: str = Field(
        default=LLM_MODEL,
        min_length=1,
        description="Модель Vireonix. Для демо используется auto",
    )

    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        ge=1,
        le=2000,
    )


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: Optional[str]
    answer: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    """
    Комбинированный ответ:
    1. детерминированный результат агроскоринга;
    2. текстовое объяснение, сформированное LLM.
    """

    prediction: PredictionResponse
    llm: LLMChatResponse


# ============================================================
# 9. SQLITE: ПОЛУЧЕНИЕ СОЕДИНЕНИЯ
# ============================================================

@asynccontextmanager
async def get_db():
    """
    Открывает отдельное асинхронное соединение с SQLite
    на одну логическую операцию.

    """

    db = await aiosqlite.connect(
        str(DATABASE_PATH),
        timeout=DB_BUSY_TIMEOUT_MS / 1000,
    )

    db.row_factory = aiosqlite.Row

    # busy_timeout задается для каждого нового соединения.
    await db.execute(
        f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS};"
    )

    try:
        yield db
    finally:
        await db.close()


# ============================================================
# 10. SQLITE: ИНИЦИАЛИЗАЦИЯ БАЗЫ
# ============================================================

async def init_db() -> None:
    """
    Создает таблицу predictions при старте приложения.

    WAL улучшает совместную работу чтения и записи.
    """

    async with get_db() as db:
        await db.execute(
            "PRAGMA journal_mode=WAL;"
        )

        await db.execute(
            "PRAGMA synchronous=NORMAL;"
        )

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_request_id
            ON predictions(request_id);
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_risk_level
            ON predictions(risk_level);
            """
        )

        await db.commit()

    logger.info(
        "SQLite initialized | path=%s",
        DATABASE_PATH,
    )


# ============================================================
# 11. SQLITE: СОХРАНЕНИЕ ОДНОГО ПРОГНОЗА
# ============================================================

async def save_prediction(
    result: dict,
    source: str,
) -> None:
    """
    Сохраняет один prediction.

    request_id не используется как PRIMARY KEY:
    один HTTP batch-request может породить несколько прогнозов
    с одним и тем же request_id.
    """

    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result["request_id"],
                    result["farm_id"],
                    result["risk_score"],
                    result["risk_level"],
                    result["recommendation"],
                    result["model_version"],
                    source,
                ),
            )

            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Database write error | request_id=%s",
            result.get("request_id"),
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 12. SQLITE: СОХРАНЕНИЕ BATCH ОДНОЙ ТРАНЗАКЦИЕЙ
# ============================================================

async def save_predictions_batch(
    request_id: str,
    items: List[dict],
) -> None:
    """
    Сохраняет весь batch через executemany()
    и один COMMIT.
    """

    rows = [
        (
            request_id,
            item["farm_id"],
            item["risk_score"],
            item["risk_level"],
            item["recommendation"],
            item["model_version"],
            "batch",
        )
        for item in items
    ]

    try:
        async with get_db() as db:
            await db.executemany(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )

            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Batch database write error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 13. SQLITE: СПИСОК ПРОГНОЗОВ
# ============================================================

async def get_predictions_from_db(
    limit: int,
    risk_level: Optional[str] = None,
) -> List[dict]:
    """Возвращает историю прогнозов из SQLite."""

    try:
        async with get_db() as db:
            if risk_level is None:
                cursor = await db.execute(
                    """
                    SELECT
                        request_id,
                        farm_id,
                        risk_score,
                        risk_level,
                        recommendation,
                        model_version
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT
                        request_id,
                        farm_id,
                        risk_score,
                        risk_level,
                        recommendation,
                        model_version
                    FROM predictions
                    WHERE risk_level = ?
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (
                        risk_level,
                        limit,
                    ),
                )

            rows = await cursor.fetchall()
            await cursor.close()

        return [
            dict(row)
            for row in rows
        ]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 14. SQLITE: ПАГИНАЦИЯ ПО ДНЮ НЕДЕЛИ
# ============================================================

async def get_predictions_by_weekday_page(
    weekday_number: str,
    page: int,
    per_page: int,
) -> tuple[List[dict], int, int]:
    """
    Возвращает одну страницу прогнозов для выбранного дня недели.

    Используется классическая offset-pagination:
   
    Пример:
        page=1, per_page=5 -> LIMIT 5 OFFSET 0
        page=2, per_page=5 -> LIMIT 5 OFFSET 5
    """

    offset = (page - 1) * per_page

    try:
        async with get_db() as db:
            # 1. Получаем записи текущей страницы.
            cursor = await db.execute(
                """
                SELECT
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version
                FROM predictions
                WHERE strftime('%w', created_at) = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
                """,
                (
                    weekday_number,
                    per_page,
                    offset,
                ),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            # 2. Считаем общее количество записей этого дня недели.
            cursor = await db.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE strftime('%w', created_at) = ?;
                """,
                (weekday_number,),
            )

            row = await cursor.fetchone()
            await cursor.close()

        total_items = row[0]

        total_pages = (
            math.ceil(total_items / per_page)
            if total_items > 0
            else 0
        )

        return (
            [dict(row) for row in rows],
            total_items,
            total_pages,
        )

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | weekday=%s",
            weekday_number,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 15. SQLITE: ДЕМО ДЛЯ ПАГИНАЦИИ
# ============================================================

async def seed_weekday_predictions(
    count: int = 70,
) -> int:
    """
    Создает демонстрационные predictions за разные дни.

    created_at распределяется по последним 21 дням,
    поэтому в базе появляются записи для всех дней недели.

    """

    rows = []

    for i in range(count):
        request_id = str(uuid.uuid4())
        farm_id = f"DEMO-{i + 1:03d}"

        risk_score = round(
            min(0.1 + (i % 9) * 0.1, 1.0),
            2,
        )

        risk_level = get_risk_level(risk_score)

        rows.append(
            (
                request_id,
                farm_id,
                risk_score,
                risk_level,
                get_recommendation(risk_level),
                MODEL_VERSION,
                "seed",
                f"-{i % 21} days",
            )
        )

    try:
        async with get_db() as db:
            await db.executemany(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    datetime('now', ?)
                );
                """,
                rows,
            )

            await db.commit()

        return count

    except aiosqlite.Error as exc:
        logger.exception(
            "Demo seed database error"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 16. SQLITE: ПОИСК ПО request_id
# ============================================================

async def get_predictions_by_request_id(
    request_id: str,
) -> List[dict]:
    """
    Возвращает все прогнозы, связанные с HTTP request_id.

    Для /predict и /predict-queued обычно будет одна строка.
    Для /batch-predict — несколько строк.
    """

    try:
        async with get_db() as db:
            cursor = await db.execute(
                """
                SELECT
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version
                FROM predictions
                WHERE request_id = ?
                ORDER BY id;
                """,
                (request_id,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

        return [
            dict(row)
            for row in rows
        ]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


# ============================================================
# 15. ЛОГИКА ОЦЕНКИ РИСКА
# ============================================================

def calculate_risk(
    data: FarmRequest,
) -> float:
    """
    Имитация ML-модели.
    """

    score = 0.1

    if data.payment_delay_days > 30:
        score += 0.3

    if data.previous_defaults > 0:
        score += 0.3

    if data.debt > 5_000_000:
        score += 0.2

    if data.precipitation_mm < 100:
        score += 0.1

    return round(
        min(score, 1.0),
        2,
    )


# ============================================================
# 16. ПОСТПРОЦЕССИНГ
# ============================================================

def get_risk_level(
    score: float,
) -> str:
    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


def get_recommendation(
    level: str,
) -> str:
    if level == "low":
        return "Стандартное рассмотрение"

    if level == "medium":
        return "Требуется дополнительная проверка"

    return "Высокий риск. Требуется ручное рассмотрение"


# ============================================================
# 17. BUSINESS VALIDATION
# ============================================================

def validate_business_rules(
    data: FarmRequest,
) -> None:
    """
    Pydantic проверяет типы, обязательность и диапазоны.
    Здесь проверяются правила предметной области.
    """

    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {data.region}. "
                f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
            ),
        )


# ============================================================
# 18. ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
# ============================================================

def make_prediction_result(
    data: FarmRequest,
    score: float,
    request_id: str,
) -> dict:
    level = get_risk_level(
        score
    )

    return {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": MODEL_VERSION,
    }


# ============================================================
# 19. LLM API: VIREONIX
# ============================================================

async def call_vireonix_llm(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Асинхронный запрос к Vireonix.

    FastAPI endpoints в этом проекте асинхронные, поэтому
    используется httpx.AsyncClient + await, а не requests.post().

    Vireonix использует OpenAI-совместимый формат:
        POST /v1/chat/completions
    """

    url = f"{LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник сервиса агроскоринга. "
                    "Отвечай кратко и понятно. "
                    "Не изменяй переданный risk_score и risk_level. "
                    "Не придумывай причины, которых нет во входных данных."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()

    try:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
            },
            json=payload,
        )

    except httpx.TimeoutException as exc:
        logger.exception(
            "LLM timeout | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timeout",
        ) from exc

    except httpx.RequestError as exc:
        logger.exception(
            "LLM connection error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is unavailable",
        ) from exc

    latency_ms = (
        time.perf_counter()
        - started
    ) * 1000

    if response.status_code != status.HTTP_200_OK:
        logger.error(
            "LLM upstream error | request_id=%s | status=%s | body=%s",
            request_id,
            response.status_code,
            response.text[:500],
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "LLM provider returned an error "
                f"(HTTP {response.status_code})"
            ),
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned invalid JSON",
        ) from exc

    choices = data.get(
        "choices",
        [],
    )

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain choices",
        )

    choice = choices[0]

    message = choice.get(
        "message",
        {},
    )

    content = message.get(
        "content"
    )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain message content",
        )

    return {
        "request_id": request_id,
        "requested_model": model,
        "returned_model": data.get("model"),
        "answer": content,
        "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }


def build_scoring_explanation_prompt(
    prediction: dict,
) -> str:
    """
    Формирует промпт только из результата
    скорингового алгоритма.

    Не отправляем внешний farm_id и исходные
    финансовые признаки в публичный LLM gateway.
    """

    return (
        "Сформируй краткое объяснение для оператора банка "
        "по результату алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n"
        "1. 2-4 предложения.\n"
        "2. Не меняй категорию риска.\n"
        "3. Не придумывай дополнительные факторы риска.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )


# ============================================================
# 20. ИМИТАЦИЯ INFERENCE ENGINE
# ============================================================

# Один lock имитирует один экземпляр модели /
# один вычислительный ресурс.

model_lock = asyncio.Lock()


async def model_predict_single(
    data: FarmRequest,
) -> float:
    """Одиночный вызов модели."""

    async with model_lock:
        await asyncio.sleep(
            MODEL_FIXED_OVERHEAD_SECONDS
            + MODEL_PER_ITEM_SECONDS
        )

        return calculate_risk(
            data
        )


async def model_predict_batch(
    items: List[FarmRequest],
) -> List[float]:
    """Один пакетный вызов модели."""

    async with model_lock:
        await asyncio.sleep(
            MODEL_FIXED_OVERHEAD_SECONDS
            + MODEL_PER_ITEM_SECONDS * len(items)
        )

        return [
            calculate_risk(item)
            for item in items
        ]


# ============================================================
# 21. ОЧЕРЕДЬ ДЛЯ DYNAMIC BATCHING
# ============================================================

inference_queue = asyncio.Queue(
    maxsize=QUEUE_MAX_SIZE
)


# ============================================================
# 22. ЭЛЕМЕНТ ОЧЕРЕДИ
# ============================================================

@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


# ============================================================
# 23. BATCH WORKER
# ============================================================

async def batch_worker() -> None:
    """
    Фоновый worker:

    1. Берет первый запрос.
    2. Копит дополнительные запросы до MAX_BATCH_SIZE
       или до истечения MAX_BATCH_WAIT_MS.
    3. Выполняет один batch inference.
    4. Возвращает каждому HTTP-запросу его score.
    """

    logger.info(
        "Batch worker started"
    )

    while True:
        first_task = await inference_queue.get()

        if first_task.future.cancelled():
            inference_queue.task_done()
            continue

        batch = [
            first_task
        ]

        deadline = (
            time.perf_counter()
            + MAX_BATCH_WAIT_MS / 1000
        )

        # ----------------------------------------------------
        # Накопление dynamic batch
        # ----------------------------------------------------

        while len(batch) < MAX_BATCH_SIZE:
            remaining_time = (
                deadline
                - time.perf_counter()
            )

            if remaining_time <= 0:
                break

            try:
                task = await asyncio.wait_for(
                    inference_queue.get(),
                    timeout=remaining_time,
                )

                if task.future.cancelled():
                    inference_queue.task_done()
                    continue

                batch.append(
                    task
                )

            except asyncio.TimeoutError:
                break

        logger.info(
            "Dynamic batch formed | size=%s | queue_size=%s",
            len(batch),
            inference_queue.qsize(),
        )

        try:
            scores = await asyncio.wait_for(
                model_predict_batch(
                    [
                        task.data
                        for task in batch
                    ]
                ),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )

            for task, score in zip(
                batch,
                scores,
            ):
                if not task.future.done():
                    task.future.set_result(
                        score
                    )

        except Exception as exc:
            logger.exception(
                "Batch inference failed"
            )

            for task in batch:
                if not task.future.done():
                    task.future.set_exception(
                        exc
                    )

        finally:
            # На каждый queue.get() должен быть task_done().
            for _ in batch:
                inference_queue.task_done()


# ============================================================
# 24. LIFESPAN FASTAPI
# ============================================================

# асинхронный контекстный менеджер lifespan
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    При старте:
    1. Инициализируем SQLite.
    2. Запускаем batch worker.

    При остановке:
    корректно завершаем worker.
    """

    await init_db()

    # Один переиспользуемый HTTP-клиент для внешнего LLM API.
    app.state.llm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            LLM_TIMEOUT_SECONDS
        ),
    )

    worker = asyncio.create_task(
        batch_worker()
    )

    app.state.batch_worker = worker

    yield

    worker.cancel()

    try:
        await worker
    except asyncio.CancelledError:
        pass

    await app.state.llm_client.aclose()


# ============================================================
# 25. FASTAPI APP
# ============================================================

# инициализируем экземпляр приложения
app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Agro Scoring API: request_id, SQLite/aiosqlite, "
        "single inference, batching и интеграция с Vireonix LLM."
    ),
    version="4.0.0",
    lifespan=lifespan,
)


# ============================================================
# 26. REQUEST_ID + PROCESS TIME MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    """
    request_id:
    - принимается из X-Request-ID;
    - если отсутствует, сервер создает UUID;
    - сохраняется в request.state;
    - возвращается клиенту в X-Request-ID.

    Также измеряется полное время HTTP-запроса.
    """

    start_time = time.perf_counter()

    request_id = request.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        request_id = str(
            uuid.uuid4()
        )

    request.state.request_id = request_id

    logger.info(
        "Request started | "
        "request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(
            request
        )

    except Exception:
        logger.exception(
            "Unhandled error | request_id=%s",
            request_id,
        )

        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    process_time = (
        time.perf_counter()
        - start_time
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time"
    ] = str(
        round(
            process_time,
            6,
        )
    )

    logger.info(
        "Request completed | "
        "request_id=%s | status=%s | time=%.4f",
        request_id,
        response.status_code,
        process_time,
    )

    return response


# ============================================================
# 27. ОБРАБОТЧИК HTTPException
# ============================================================

@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id,
        },
        headers=exc.headers,
    )


# ============================================================
# 28. ОБРАБОТЧИК PYDANTIC / FASTAPI VALIDATION
# ============================================================

@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(
                exc.errors()
            ),
            "request_id": request_id,
        },
    )


# ============================================================
# 29. GET /health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
)
async def health():
    return {
        "status": "ok",
    }


# ============================================================
# 30. GET /model-info
# ============================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
)
async def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": (
            "ready"
            if MODEL_READY
            else "unavailable"
        ),
    }


# ============================================================
# 31. LLM CHAT
# POST /llm/chat
# ============================================================

@app.post(
    "/llm/chat",
    response_model=LLMChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Прямой учебный запрос к Vireonix LLM",
    description=(
        "Отправляет prompt в OpenAI-совместимый Vireonix API. "
        "API-ключ и Authorization header не используются. "
        "Не передавайте реальные конфиденциальные данные."
    ),
)
async def llm_chat(
    data: LLMChatRequest,
    request: Request,
):
    request_id = (
        request.state.request_id
    )

    return await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=data.prompt,
        request_id=request_id,
        model=data.model,
        temperature=data.temperature,
        max_tokens=data.max_tokens,
    )


# ============================================================
# 32. PREDICT + LLM EXPLANATION
# POST /predict-with-llm
# ============================================================

@app.post(
    "/predict-with-llm",
    response_model=PredictWithLLMResponse,
    status_code=status.HTTP_200_OK,
    summary="Агроскоринг с текстовым объяснением LLM",
    description=(
        "Сначала выполняет детерминированный агроскоринг и сохраняет "
        "результат в SQLite. Затем Vireonix LLM формирует только "
        "текстовое объяснение результата. LLM не определяет risk_score."
    ),
)
async def predict_with_llm(
    data: FarmRequest,
    request: Request,
):
    """
    Внешняя LLM не используется для расчета risk_score.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(
        data
    )

    request_id = (
        request.state.request_id
    )

    try:
        score = await asyncio.wait_for(
            model_predict_single(
                data
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    prediction = make_prediction_result(
        data,
        score,
        request_id,
    )

    # Сначала сохраняем детерминированный prediction.
    await save_prediction(
        prediction,
        source="single+llm",
    )

    # В публичный LLM endpoint отправляем только уже рассчитанный
    # результат скоринга, без farm_id и исходных финансовых признаков.
    prompt = build_scoring_explanation_prompt(
        prediction
    )

    llm_result = await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=prompt,
        request_id=request_id,
        model=LLM_MODEL,
        temperature=LLM_DEFAULT_TEMPERATURE,
        max_tokens=LLM_DEFAULT_MAX_TOKENS,
    )

    return {
        "prediction": prediction,
        "llm": llm_result,
    }


# ============================================================
# 33. SINGLE INFERENCE
# POST /predict
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Одиночный inference",
)
async def predict(
    data: FarmRequest,
    request: Request,
):
    """
    Один HTTP-request -> один inference -> одна запись SQLite.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(
        data
    )

    request_id = (
        request.state.request_id
    )

    try:
        score = await asyncio.wait_for(
            model_predict_single(
                data
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    result = make_prediction_result(
        data,
        score,
        request_id,
    )

    # Важно:
    # HTTP 200 возвращается только после успешного COMMIT.
    await save_prediction(
        result,
        source="single",
    )

    return result


# ============================================================
# 34. EXPLICIT BATCH INFERENCE
# POST /batch-predict
# ============================================================

@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Явный пакетный inference",
)
async def batch_predict(
    batch: BatchRequest,
    request: Request,
):
    """
    Клиент сам передает несколько хозяйств одним HTTP-запросом.

    Все результаты batch сохраняются в SQLite одной транзакцией:
    executemany() + один commit().
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    for item in batch.items:
        validate_business_rules(
            item
        )

    request_id = (
        request.state.request_id
    )

    start_time = time.perf_counter()

    try:
        scores = await asyncio.wait_for(
            model_predict_batch(
                batch.items
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Batch inference timeout",
        )

    response_items = []

    for data, score in zip(
        batch.items,
        scores,
    ):
        level = get_risk_level(
            score
        )

        response_items.append(
            {
                "farm_id": data.farm_id,
                "risk_score": score,
                "risk_level": level,
                "recommendation": get_recommendation(
                    level
                ),
                "model_version": MODEL_VERSION,
            }
        )

    # Сохраняем весь batch одной транзакцией.
    await save_predictions_batch(
        request_id=request_id,
        items=response_items,
    )

    processing_time_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return {
        "request_id": request_id,
        "batch_size": len(
            batch.items
        ),
        "processing_time_ms": round(
            processing_time_ms,
            2,
        ),
        "items": response_items,
    }


# ============================================================
# 35. DYNAMIC BATCHING
# POST /predict-queued
# ============================================================

@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Inference через очередь и dynamic batching",
)
async def predict_queued(
    data: FarmRequest,
    request: Request,
):
    """
    С точки зрения клиента это одиночный HTTP-request.

    Сервер помещает его в asyncio.Queue и может объединить
    несколько запросов в один вызов модели.

    После получения score каждый запрос отдельно сохраняет
    свой результат в SQLite.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(
        data
    )

    request_id = (
        request.state.request_id
    )

    # --------------------------------------------------------
    # УПРАВЛЕНИЕ ПОТОКОМ ЗАПРОСОВ BACKPRESSURE
    # --------------------------------------------------------

    if inference_queue.full():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Inference queue is full. "
                "Try again later."
            ),
            headers={
                "Retry-After": "1",
            },
        )

    loop = asyncio.get_running_loop()

    future = loop.create_future()

    task = InferenceTask(
        data=data,
        request_id=request_id,
        future=future,
    )

    try:
        inference_queue.put_nowait(
            task
        )

    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference queue is full",
            headers={
                "Retry-After": "1",
            },
        )

    logger.info(
        "Request queued | "
        "request_id=%s | queue_size=%s",
        request_id,
        inference_queue.qsize(),
    )

    try:
        score = await asyncio.wait_for(
            future,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        if not future.done():
            future.cancel()

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    result = make_prediction_result(
        data,
        score,
        request_id,
    )

    await save_prediction(
        result,
        source="queued",
    )

    return result


# ============================================================
# 36. GET /queue-status
# ============================================================

@app.get(
    "/queue-status",
    summary="Состояние очереди и настройки batching",
)
async def queue_status():
    return {
        "queue_size": inference_queue.qsize(),
        "queue_max_size": QUEUE_MAX_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_batch_wait_ms": MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
    }


# ============================================================
# 37. GET /predictions
# ============================================================

@app.get(
    "/predictions",
    response_model=List[PredictionResponse],
    summary="Получить историю прогнозов из SQLite",
)
async def get_predictions(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    risk_level: Optional[str] = Query(
        default=None,
        description="low, medium или high",
    ),
):
    if risk_level is not None:
        allowed_levels = {
            "low",
            "medium",
            "high",
        }

        if risk_level not in allowed_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "risk_level must be "
                    "low, medium or high"
                ),
            )

    return await get_predictions_from_db(
        limit=limit,
        risk_level=risk_level,
    )


# ============================================================
# 38. GET /requests/{request_id}/predictions
# ============================================================

@app.get(
    "/requests/{request_id}/predictions",
    response_model=List[PredictionResponse],
    summary="Все прогнозы, созданные одним HTTP request",
)
async def get_request_predictions(
    request_id: str,
):
    """
    Для single / queued обычно возвращает один элемент.
    Для explicit batch может вернуть несколько элементов.
    """

    results = await get_predictions_by_request_id(
        request_id
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    return results


# ============================================================
# 39. GET /predictions/by-weekday
# ПАГИНАЦИЯ + ФИЛЬТР ПО ДНЮ НЕДЕЛИ
# ============================================================

@app.get(
    "/predictions/by-weekday",
    response_model=PaginatedPredictionsResponse,
    summary="Прогнозы по дню недели с пагинацией",
)
async def get_predictions_by_weekday(
    request: Request,

    weekday: str = Query(
        ...,
        description=(
            "День недели: monday, tuesday, wednesday, "
            "thursday, friday, saturday, sunday"
        ),
    ),

    page: int = Query(
        default=1,
        ge=1,
        description="Номер страницы",
    ),

    per_page: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Количество записей на странице",
    ),
):
    """
    Пример:
        GET /predictions/by-weekday?weekday=monday&page=1&per_page=5

    Вернет первые 5 записей для понедельника.
    """

    weekday = weekday.lower()

    if weekday not in WEEKDAY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "weekday must be one of: "
                "monday, tuesday, wednesday, thursday, "
                "friday, saturday, sunday"
            ),
        )

    (
        data,
        total_items,
        total_pages,
    ) = await get_predictions_by_weekday_page(
        weekday_number=WEEKDAY_MAP[weekday],
        page=page,
        per_page=per_page,
    )

    # Если записи существуют, но номер страницы слишком большой.
    if total_pages > 0 and page > total_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Page {page} does not exist. "
                f"Total pages: {total_pages}"
            ),
        )

    # Если записей нет вообще, разрешаем только page=1.
    if total_pages == 0 and page > 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data for this weekday",
        )

    next_page = None

    if page < total_pages:
        next_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page + 1,
                per_page=per_page,
            )
        )

    prev_page = None

    if page > 1 and total_pages > 0:
        prev_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page - 1,
                per_page=per_page,
            )
        )

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": next_page,
            "prev_page": prev_page,
        },
    }


# ============================================================
# 40. POST /demo/seed-weekdays
#ЗАПОЛНЕНИЕ БД ДАННЫМИ ЗА РАЗНЫЕ ДНИ
# ============================================================

@app.post(
    "/demo/seed-weekdays",
    summary="Создать демонстрационные predictions за разные дни",
)
async def seed_demo_weekdays(
    count: int = Query(
        default=70,
        ge=7,
        le=1000,
        description="Количество демонстрационных записей",
    ),
):
    """
    Создает данные за последние 21 день, чтобы можно было
    протестировать выборку по дням недели и пагинацию.
    """

    created = await seed_weekday_predictions(
        count=count
    )

    return {
        "status": "ok",
        "created": created,
        "message": (
            "Demo predictions created. "
            "Use /predictions/by-weekday for pagination."
        ),
    }


# ============================================================
# 41. GET /predictions/{request_id}
# ============================================================

@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить одиночный прогноз по request_id",
)
async def get_prediction(
    request_id: str,
):
    """
    Сохраняет обратную совместимость старого API.

    Если request_id относится к batch и в БД несколько строк,
    возвращается 409 и предлагается использовать endpoint:
    /requests/{request_id}/predictions
    """

    results = await get_predictions_by_request_id(
        request_id
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions"
            ),
        )

    return results[0]


@app.get("/")
async def root():
    return {
        "message": "Agro Scoring API is running",
        "docs": "/docs",
        "health": "/health"
    }

# Запуск
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

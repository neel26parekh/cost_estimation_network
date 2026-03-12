from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from .config import CORS_ORIGINS, MODEL_MAX_AGE_DAYS, RECENT_PREDICTIONS_LIMIT, ensure_directories
from .drift import load_latest_drift_report
from .logger import get_logger
from .monitoring import append_prediction_log, read_recent_prediction_logs, summarize_prediction_logs
from .predict import load_metadata, load_model, predict_price
from .schemas import PredictionLogEntry, PredictionRequest, PredictionResponse
from .security import authorize_request

logger = get_logger(__name__)

app = FastAPI(title="Laptop Price Predictor API", version="1.0.0")
ensure_directories()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service_unavailable(exc: FileNotFoundError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Top-level probe endpoints (no auth, no versioning — for load balancers)
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "laptop-price-predictor-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    """Health check with optional model staleness warning."""
    result: dict = {"status": "ok"}
    try:
        metadata = load_metadata()
        trained_at = datetime.fromisoformat(metadata["trained_at_utc"])
        age_days = (datetime.now(UTC) - trained_at).days
        result["model_age_days"] = age_days
        if age_days > MODEL_MAX_AGE_DAYS:
            result["warning"] = f"Model is {age_days} days old (threshold: {MODEL_MAX_AGE_DAYS})"
    except (FileNotFoundError, KeyError):
        pass
    return result


@app.get("/ready")
def readiness() -> dict[str, str]:
    try:
        load_metadata()
        load_model()
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Versioned API router — /v1/
# ---------------------------------------------------------------------------

v1 = APIRouter(prefix="/v1")


@v1.get("/metadata")
def metadata(request: Request, _: None = Depends(authorize_request)) -> dict:
    try:
        return load_metadata()
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@v1.get("/predictions/recent", response_model=list[PredictionLogEntry])
def recent_predictions(
    request: Request,
    limit: int = Query(default=RECENT_PREDICTIONS_LIMIT, ge=1, le=100),
    _: None = Depends(authorize_request),
) -> list[dict]:
    return read_recent_prediction_logs(limit=limit)


@v1.get("/monitoring/drift")
def latest_drift_report(request: Request, _: None = Depends(authorize_request)) -> dict:
    try:
        return load_latest_drift_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@v1.get("/monitoring/summary")
def monitoring_summary(request: Request, _: None = Depends(authorize_request)) -> dict:
    return summarize_prediction_logs()


@v1.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest, _: None = Depends(authorize_request)) -> PredictionResponse:
    started_at = perf_counter()
    request_id = uuid4().hex
    try:
        metadata_payload = load_metadata()
        predicted_price = predict_price(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc

    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    append_prediction_log(
        {
            "request_id": request_id,
            "model_name": metadata_payload["model_name"],
            "model_version": metadata_payload["model_version"],
            "predicted_price_inr": round(predicted_price, 2),
            "latency_ms": latency_ms,
            "features": payload.model_dump(),
        }
    )
    logger.info("Prediction request_id=%s price=%.2f latency=%.2fms", request_id, predicted_price, latency_ms)

    return PredictionResponse(
        predicted_price_inr=round(predicted_price, 2),
        model_name=metadata_payload["model_name"],
        model_version=metadata_payload["model_version"],
        request_id=request_id,
        latency_ms=latency_ms,
    )


app.include_router(v1)

# ---------------------------------------------------------------------------
# Backward-compatible unversioned aliases for existing clients
# ---------------------------------------------------------------------------

@app.get("/metadata")
def metadata_compat(request: Request, _: None = Depends(authorize_request)) -> dict:
    return metadata(request, _)


@app.get("/predictions/recent", response_model=list[PredictionLogEntry])
def recent_predictions_compat(
    request: Request,
    limit: int = Query(default=RECENT_PREDICTIONS_LIMIT, ge=1, le=100),
    _: None = Depends(authorize_request),
) -> list[dict]:
    return recent_predictions(request, limit, _)


@app.get("/monitoring/drift")
def latest_drift_report_compat(request: Request, _: None = Depends(authorize_request)) -> dict:
    return latest_drift_report(request, _)


@app.get("/monitoring/summary")
def monitoring_summary_compat(request: Request, _: None = Depends(authorize_request)) -> dict:
    return monitoring_summary(request, _)


@app.post("/predict", response_model=PredictionResponse)
def predict_compat(
    request: Request,
    payload: PredictionRequest,
    _: None = Depends(authorize_request),
) -> PredictionResponse:
    return predict(request, payload, _)

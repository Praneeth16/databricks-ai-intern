"""FastAPI application for the Databricks AI Intern web interface."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.agent import router as agent_router
from routes.auth import router as auth_router

# Load .env from project root (parent directory)
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting backend...")
    # MLflow Tracing flushes traces server-side; KPIs come from Lakeview
    # over system.mlflow.traces + system.serving.endpoint_usage.
    try:
        import lakebase
        from session_manager import session_manager
        lakebase.init(session_manager.config)
    except Exception as e:
        logger.warning("Lakebase init skipped: %s", e)

    yield

    logger.info("Shutting down backend...")
    try:
        import lakebase
        lakebase.shutdown()
    except Exception as e:
        logger.debug("Lakebase shutdown suppressed: %s", e)
    try:
        from session_manager import session_manager
        for sid, agent_session in list(session_manager.sessions.items()):
            sess = agent_session.session
            if sess.config.save_sessions:
                try:
                    sess.save_trajectory_local()
                except Exception as e:
                    logger.warning("Failed to flush session %s: %s", sid, e)
    except Exception as e:
        logger.warning("Lifespan final-flush skipped: %s", e)


app = FastAPI(
    title="Databricks AI Intern",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# Apps runtime signal — same one dependencies.py uses.
_APPS_MODE = bool(
    os.environ.get("DATABRICKS_APP_NAME") or os.environ.get("DATABRICKS_WORKSPACE_ID")
)

# CORS: wide open for local dev (Vite on another port); in Apps mode the
# frontend is same-origin, so restrict to what it actually uses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"] if _APPS_MODE else ["*"],
    allow_headers=["content-type", "authorization"] if _APPS_MODE else ["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)

# Serve static files (frontend build) in production
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Serving static files from {static_path}")
else:
    logger.info("No static directory found, running in API-only mode")


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "Databricks AI Intern API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

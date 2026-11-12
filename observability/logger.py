"""
Observability & Structured Logging Module.

Features:
  - Structured JSON log formatting for auditability and production ingestion.
  - OpenTelemetry-compatible tracing spans (agent step latency, tool executions).
  - Trace export to JSONL logs (`logs/agent_traces.jsonl`).
  - Swarm execution performance and telemetry summary metrics.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TRACE_LOG_FILE = os.path.join(LOGS_DIR, "agent_traces.jsonl")


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "agent"):
            log_obj["agent"] = record.agent
        if hasattr(record, "step"):
            log_obj["step"] = record.step
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        if hasattr(record, "data"):
            log_obj["data"] = record.data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


# Configure dedicated trace logger
_trace_logger = logging.getLogger("swarm.trace")
_trace_logger.setLevel(logging.INFO)

# File handler for JSONL traces
_file_handler = logging.FileHandler(TRACE_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(StructuredJSONFormatter())
if not any(isinstance(h, logging.FileHandler) for h in _trace_logger.handlers):
    _trace_logger.addHandler(_file_handler)


def log_agent_event(
    agent: str,
    step: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
    level: int = logging.INFO,
) -> None:
    """Log structured agent event to JSONL trace log."""
    extra = {
        "agent": agent,
        "step": step,
        "data": data,
        "latency_ms": latency_ms,
    }
    if error:
        extra["error"] = error
        level = logging.ERROR
    _trace_logger.log(level, message, extra=extra)


@contextmanager
def trace_span(agent: str, step: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
    """Context manager for tracing execution spans with latency tracking."""
    start_time = time.time()
    span_data: Dict[str, Any] = {"agent": agent, "step": step, "metadata": metadata or {}}
    try:
        yield span_data
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        span_data["latency_ms"] = elapsed_ms
        log_agent_event(agent, step, f"Span '{step}' completed", data=span_data, latency_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        span_data["latency_ms"] = elapsed_ms
        span_data["error"] = str(e)
        log_agent_event(agent, step, f"Span '{step}' failed: {e}", error=str(e), latency_ms=elapsed_ms)
        raise e


def generate_trace_summary(agent_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an observability summary from agent logs:
      - Total steps
      - Per-agent latency breakdown
      - Total tool calls
      - Error counts
    """
    agent_latencies_ms: Dict[str, float] = {}
    steps_by_agent: Dict[str, int] = {}
    tool_calls_count = 0
    error_count = 0

    for log in agent_logs:
        agent = log.get("agent", "Unknown")
        steps_by_agent[agent] = steps_by_agent.get(agent, 0) + 1

        if log.get("latency_ms"):
            agent_latencies_ms[agent] = round(
                agent_latencies_ms.get(agent, 0.0) + log["latency_ms"], 2
            )
        if "tool" in log.get("step", "").lower() or log.get("step") == "tool_invoke":
            tool_calls_count += 1
        if log.get("error"):
            error_count += 1

    summary: Dict[str, Any] = {
        "total_log_entries": len(agent_logs),
        "agent_latencies_ms": agent_latencies_ms,
        "tool_calls_count": tool_calls_count,
        "error_count": error_count,
        "steps_by_agent": steps_by_agent,
    }
    return summary


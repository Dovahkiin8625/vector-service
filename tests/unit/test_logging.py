import io
import json
import logging

import structlog

from vector_service.core.logging import setup_logging, request_id_var


def _capture_log_output(format: str = "json", level: str = "INFO"):
    buf = io.StringIO()
    setup_logging(format=format, level=level, stream=buf)
    return buf


def test_json_output_is_valid_json():
    buf = _capture_log_output("json")
    log = structlog.get_logger("test")
    log.info("hello", foo="bar")
    line = buf.getvalue().strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["event"] == "hello"
    assert obj["foo"] == "bar"
    assert obj["level"] == "info"
    assert "ts" in obj


def test_console_output_is_text():
    buf = _capture_log_output("console")
    log = structlog.get_logger("test")
    log.info("hi", k="v")
    assert "hi" in buf.getvalue()


def test_request_id_included_when_set():
    buf = _capture_log_output("json")
    request_id_var.set("req_abc")
    log = structlog.get_logger("test")
    log.info("with-id")
    obj = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert obj["request_id"] == "req_abc"


def test_log_level_respected():
    """Verify level filtering via rendered stderr.

    Note: we cannot use pytest's `caplog` here because `setup_logging`
    clears root handlers (production behavior to avoid duplicate output).
    Instead we capture stderr, which is where structlog renders.
    """
    buf = io.StringIO()
    setup_logging(format="console", level="WARNING", stream=buf)
    log = structlog.get_logger("test_lvl")
    log.info("should-be-filtered")
    log.warning("should-appear")
    rendered = buf.getvalue()
    assert "should-be-filtered" not in rendered
    assert "should-appear" in rendered

"""Outputs for violations, periodic metrics and lifecycle events.

    stdout   JSON lines on stdout
    file     JSON lines appended to `path`
    webhook  POST JSON to `url`
    agent    edge-ops Edge Agent HTTP API (POST /runtimes/{id}/events and /telemetry)
    mqtt     publish to `{topic_prefix}/violation|metrics|event` (needs paho-mqtt)
    kafka    produce to `topic` (needs kafka-python)

Network sinks send from a background thread so a slow endpoint never stalls inference; when their queue is full,
the oldest message is dropped.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ppe_runtime.ppe.constants import EVENT_VIOLATION
from ppe_runtime.ppe.models import Violation
from ppe_runtime.utils.config import SinkConfig
from ppe_runtime.utils.metrics import TELEMETRY_KEYS

logger = logging.getLogger(__name__)


class Sink(ABC):
    def on_violation(self, violation: Violation) -> None:
        self.send({"type": "violation", "ts": time.time(), **violation.to_dict()})

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        self.send({"type": "metrics", "ts": time.time(), **snapshot})

    def on_event(self, event_type: str, message: str, severity: str = "INFO", data: dict | None = None) -> None:
        self.send(
            {
                "type": "event",
                "ts": time.time(),
                "event_type": event_type,
                "message": message,
                "severity": severity,
                "data": data or {},
            }
        )

    @abstractmethod
    def send(self, record: dict[str, Any]) -> None: ...

    def close(self) -> None:
        pass


class StdoutSink(Sink):
    def __init__(self, metrics: bool = False) -> None:
        self._metrics = metrics

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        if self._metrics:
            super().on_metrics(snapshot)

    def send(self, record: dict[str, Any]) -> None:
        print(json.dumps(record, default=str), file=sys.stdout, flush=True)


class FileSink(Sink):
    def __init__(self, path: str, metrics: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self._metrics = metrics
        self._lock = threading.Lock()

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        if self._metrics:
            super().on_metrics(snapshot)

    def send(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


class BackgroundSink(Sink):
    """Queues records and delivers them on a worker thread via `deliver()`; failures are logged, never raised."""

    def __init__(self, max_queue: int = 1000) -> None:
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._worker, name=f"{type(self).__name__}", daemon=True)
        self._thread.start()

    def send(self, record: dict[str, Any]) -> None:
        while True:
            try:
                self._queue.put_nowait(record)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def _worker(self) -> None:
        while (record := self._queue.get()) is not None:
            try:
                self.deliver(record)
            except Exception as exc:  # network sinks must not kill the runtime
                logger.warning("%s delivery failed: %s", type(self).__name__, exc)

    @abstractmethod
    def deliver(self, record: dict[str, Any]) -> None: ...

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)


class WebhookSink(BackgroundSink):
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0, metrics: bool = False):
        import requests

        self._session = requests.Session()
        self._session.headers.update(headers or {})
        self._url, self._timeout, self._metrics = url, timeout, metrics
        super().__init__()

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        if self._metrics:
            super().on_metrics(snapshot)

    def deliver(self, record: dict[str, Any]) -> None:
        self._session.post(self._url, json=record, timeout=self._timeout).raise_for_status()


class AgentSink(BackgroundSink):
    """Reports to the local edge-ops Edge Agent: violations/lifecycle as events, metrics as `RuntimeTelemetry`."""

    def __init__(
        self, agent_url: str, runtime_id: str, node_id: str, api_key: str | None = None, timeout: float = 5.0
    ) -> None:
        import requests

        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key
        self._base = f"{agent_url.rstrip('/')}/runtimes/{runtime_id}"
        self._runtime_id, self._node_id, self._timeout = runtime_id, node_id, timeout
        super().__init__()

    def on_violation(self, violation: Violation) -> None:
        missing = ", ".join(sorted(violation.missing))
        who = f"track {violation.track_id}" if violation.track_id is not None else "person"
        self.on_event(
            EVENT_VIOLATION, f"PPE violation: {who} missing {missing}", violation.severity, violation.to_dict()
        )

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        body = {k: v for k, v in snapshot.items() if k in TELEMETRY_KEYS}
        body.update(runtime_id=self._runtime_id, node_id=self._node_id)
        self.send({"_path": "telemetry", **body})

    def on_event(self, event_type: str, message: str, severity: str = "INFO", data: dict | None = None) -> None:
        self.send(
            {"_path": "events", "event_type": event_type, "message": message, "severity": severity, "data": data or {}}
        )

    def deliver(self, record: dict[str, Any]) -> None:
        path = record.pop("_path")
        self._session.post(f"{self._base}/{path}", json=record, timeout=self._timeout).raise_for_status()


class MqttSink(BackgroundSink):
    def __init__(
        self,
        host: str,
        port: int = 1883,
        topic_prefix: str = "ppe",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
        client_id: str = "",
    ) -> None:
        import paho.mqtt.client as mqtt

        try:
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except AttributeError:  # paho-mqtt < 2
            self._client = mqtt.Client(client_id=client_id)
        if username:
            self._client.username_pw_set(username, password)
        self._client.connect_async(host, port)
        self._client.loop_start()
        self._prefix, self._qos = topic_prefix.rstrip("/"), qos
        super().__init__()

    def deliver(self, record: dict[str, Any]) -> None:
        self._client.publish(f"{self._prefix}/{record['type']}", json.dumps(record, default=str), qos=self._qos)

    def close(self) -> None:
        super().close()
        self._client.loop_stop()
        self._client.disconnect()


class KafkaSink(BackgroundSink):
    def __init__(self, bootstrap_servers: str | list[str], topic: str = "ppe-events", **producer_kwargs) -> None:
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda r: json.dumps(r, default=str).encode("utf-8"),
            **producer_kwargs,
        )
        self._topic = topic
        super().__init__()

    def deliver(self, record: dict[str, Any]) -> None:
        self._producer.send(self._topic, record)

    def close(self) -> None:
        super().close()
        self._producer.flush(timeout=5)
        self._producer.close()


_SINKS: dict[str, type[Sink]] = {
    "stdout": StdoutSink,
    "file": FileSink,
    "webhook": WebhookSink,
    "agent": AgentSink,
    "mqtt": MqttSink,
    "kafka": KafkaSink,
}


def create_sink(cfg: SinkConfig) -> Sink:
    return _SINKS[cfg.type](**cfg.options)


class MultiSink(Sink):
    """Fans every call out to several sinks; one failing sink does not affect the others."""

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks

    def _each(self, method: str, *args) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args)
            except Exception:
                logger.exception("%s.%s failed", type(sink).__name__, method)

    def on_violation(self, violation: Violation) -> None:
        self._each("on_violation", violation)

    def on_metrics(self, snapshot: dict[str, Any]) -> None:
        self._each("on_metrics", snapshot)

    def on_event(self, event_type: str, message: str, severity: str = "INFO", data: dict | None = None) -> None:
        self._each("on_event", event_type, message, severity, data)

    def send(self, record: dict[str, Any]) -> None:
        self._each("send", record)

    def close(self) -> None:
        self._each("close")

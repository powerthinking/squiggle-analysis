"""Event detection modules."""

from .change_point import (
    EventEntropyMetrics,
    compute_event_entropy,
    detect_events,
)

__all__ = [
    "EventEntropyMetrics",
    "compute_event_entropy",
    "detect_events",
]

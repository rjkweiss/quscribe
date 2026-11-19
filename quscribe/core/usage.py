"""
Token and latency accounting for model calls, used in cost and latency reports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class UsageRecord:
    """
    Usage of a single model call.

    Attributes:
        model_name: The model that served the call.
        input_tokens: Prompt tokens billed for the call.
        output_tokens: Completion tokens billed for the call.
        latency_seconds: Time the call took.
    """

    model_name: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class UsageSummary:
    """
    Aggregated usage over a group of model calls.

    Attributes:
        call_count: Number of model calls.
        input_tokens: Total prompt tokens.
        output_tokens: Total completion tokens.
        model_latency_seconds: Total time spent waiting for models.
    """

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_seconds: float = 0.0

    @classmethod
    def from_records(cls, records: Iterable[UsageRecord]) -> UsageSummary:
        """
        Aggregate individual call records.

        Args:
            records: The records to aggregate.

        Returns:
            UsageSummary: Totals across all records.
        """
        record_list = list(records)
        return cls(
            call_count=len(record_list),
            input_tokens=sum(record.input_tokens for record in record_list),
            output_tokens=sum(record.output_tokens for record in record_list),
            model_latency_seconds=sum(
                record.latency_seconds for record in record_list),
        )


@dataclass(frozen=True)
class ModelPricing:
    """
    Per-token prices, manually provided since provider prices change over time.

    Attributes:
        input_cost_per_million_tokens: Price in USD per million prompt tokens.
        output_cost_per_million_tokens: Price in USD per million completion
                                        tokens.
    """

    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float

    def cost_of(self, usage: UsageSummary) -> float:
        """
        Compute the cost of aggregated usage.

        Args:
            usage: The usage to price.

        Returns:
            float: The cost in USD.
        """
        return (
            usage.input_tokens * self.input_cost_per_million_tokens
            + usage.output_tokens * self.output_cost_per_million_tokens
        ) / TOKENS_PER_MILLION


class UsageTracker:
    """
    Append-only log of model calls, shared by every model in one pipeline.

    Callers take a ``mark`` before a unit of work and ``summarize`` from that
    mark afterwards, yielding per-circuit usage without resetting shared state.
    """

    def __init__(self) -> None:
        """
        Create an empty tracker.
        """
        self._records: list[UsageRecord] = []

    def record(self, usage_record: UsageRecord) -> None:
        """
        Append the usage of one model call.

        Args:
            usage_record: The call to record.

        Returns:
            None.
        """
        self._records.append(usage_record)

    def mark(self) -> int:
        """
        Return a position marker for later use with ``summarize``.

        Returns:
            int: The number of calls recorded so far.
        """
        return len(self._records)

    def summarize(self, since_mark: int = 0) -> UsageSummary:
        """
        Aggregate the calls recorded since a marker.

        Args:
            since_mark: A value previously returned by ``mark``;
                        0 covers every call.

        Returns:
            UsageSummary: Totals for the calls after the marker.
        """
        return UsageSummary.from_records(self._records[since_mark:])

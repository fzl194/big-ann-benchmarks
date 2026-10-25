"""Opt-in debt-triggered REINDEX and current-run-count search budgeting.

Only current catalog row counts and the adapter's committed live count are
used. No runbook look-ahead, query answers, or GT participate in decisions.
"""
from fractions import Fraction
import math
import time

from psycopg import sql

from .module import Gsvector, QueryCardinalityError


class ReindexPlanner:
    def __init__(self, ratio=0, min_rows=1000000, cooldown=16):
        if (type(ratio) not in (int, float) or
                (type(ratio) is float and not math.isfinite(ratio)) or
                (ratio != 0 and ratio <= 1)):
            raise ValueError("reindex_ratio must be zero or a finite number greater than one")
        for name, value in (("min_rows", min_rows), ("cooldown", cooldown)):
            if type(value) is not int or not 1 <= value <= 2147483647:
                raise ValueError(f"reindex {name} must be a positive bounded integer")
        self.ratio = Fraction(str(ratio))
        self.min_rows, self.cooldown = min_rows, cooldown
        self.reset()

    def reset(self):
        self.tick = self.due = 0

    def plan(self, physical_rows, live_rows):
        if any(type(n) is not int or n < 0 for n in (physical_rows, live_rows)):
            raise ValueError("nonnegative integer row counts required")
        self.tick += 1
        if (not self.ratio or not live_rows or physical_rows < self.min_rows or
                self.tick < self.due or
                physical_rows * self.ratio.denominator < live_rows * self.ratio.numerator):
            return False
        self.due = self.tick + self.cooldown
        return True


def query_budget(arguments, run_count, run_floor):
    result = dict(arguments)
    if type(run_count) is not int or run_count < 0:
        raise ValueError("invalid run count")
    if type(run_floor) is not int or not 0 <= run_floor <= 64:
        raise ValueError("query_run_floor must be an integer in 0..64")
    if not run_floor:
        return result
    if "max_k" not in arguments:
        raise ValueError("adaptive query budgeting requires an explicit max_k")
    divisor = max(1, run_count)
    for name, upper in (("search_l", 100000), ("max_k", 32768)):
        base = arguments[name]
        if type(base) is not int or not 1 <= base <= upper:
            raise ValueError(f"invalid base {name}")
        result[name] = min(upper, max(base, (base * run_floor + divisor - 1) // divisor))
    return result


class AdaptiveGsvector(Gsvector):
    def __init__(self, metric, index_params):
        params = dict(index_params)
        ratio = params.pop("reindex_ratio", 0)
        minimum = params.pop("reindex_min_rows", 1000000)
        cooldown = params.pop("reindex_cooldown", 16)
        floor = params.pop("query_run_floor", 0)
        retries = params.pop("query_rebuild_retries", 0)
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("query_rebuild_retries must be an integer in 0..2")
        self.query_rebuild_retries = retries
        self.reindex_planner = ReindexPlanner(ratio, minimum, cooldown)
        query_budget({"search_l": 200, "max_k": 100}, 1, floor)
        self.query_run_floor = floor
        self._pending_reindexes = []
        self._budget_context = None
        self._query_started_at = None
        self._query_attempts = []
        super().__init__(metric, params)
        self.params.update({k: v for k, v in index_params.items() if k not in params})

    def _new_workload(self):
        self.reindex_planner.reset()
        self._pending_reindexes = []
        self._budget_context = None
        super()._new_workload()

    def _maintenance(self):
        super()._maintenance()
        if not self.index_ready or not self.reindex_planner.ratio:
            return
        before = self._runs()
        physical = sum(r[1] for r in before)
        if not self.reindex_planner.plan(physical, self.live_count):
            return
        self._perform_reindex(before, "debt_ratio")

    def _perform_reindex(self, before, reason):
        physical = sum(r[1] for r in before)
        started = time.perf_counter()
        # These idle readers will lose their old run handles at REINDEX
        # publication anyway. Release their mappings before the bulk build.
        self._close_query_pool()
        self.conn.execute(sql.SQL("REINDEX INDEX {}.{}").format(
            sql.Identifier(self.schema), sql.Identifier("ann_idx")))
        elapsed = time.perf_counter() - started
        after = self._runs()
        self.maintenance_totals["reindex_s"] = self.maintenance_totals.get("reindex_s", 0) + elapsed
        self.maintenance_totals["reindex_calls"] = self.maintenance_totals.get("reindex_calls", 0) + 1
        self._pending_reindexes.append({
            "reason": reason,
            "maintenance_tick": self.reindex_planner.tick,
            "graph_rows_before": physical, "graph_rows_after": sum(r[1] for r in after),
            "live_rows": self.live_count, "runs_before": len(before), "runs_after": len(after),
            "elapsed_s": elapsed,
        })
        if self.reclaim_planner is not None:
            self.reclaim_planner.reset()
        # The base maintenance already closed its writer if requested. A
        # subsequent REINDEX has its own build allocations to release.
        self._trim_allocator()
        self._reconnect_writer()

    def query(self, X, k):
        nominal = self.query_args
        self._query_started_at = time.perf_counter()
        self._query_attempts = []
        try:
            for attempt in range(self.query_rebuild_retries + 1):
                count = len(self._runs()) if self.query_run_floor else 0
                effective = query_budget(nominal, count, self.query_run_floor)
                self._budget_context = {
                    "run_count": count, "run_floor": self.query_run_floor,
                    "nominal_query_args": dict(nominal),
                } if self.query_run_floor else None
                self.query_args = effective
                receipt = {"attempt": attempt, "query_args": dict(effective), "status": "running"}
                self._query_attempts.append(receipt)
                try:
                    return super().query(X, k)
                except QueryCardinalityError as exc:
                    receipt.update(status="shortfall", found=exc.found, expected=exc.expected,
                                   completed_queries=exc.completed_queries_total)
                    if attempt == self.query_rebuild_retries:
                        raise
                    before = self._runs()
                    if sum(r[1] for r in before) <= self.live_count:
                        raise  # no deletion debt to repair; don't mask other defects
                    self._perform_reindex(before, "query_shortfall")
        finally:
            self.query_args = nominal
            self._budget_context = None
            self._query_started_at = None
            self._query_attempts = []

    def _record(self, operation, **data):
        if operation in ("insert", "delete", "replace", "search") and self._pending_reindexes:
            data["maintenance_reindexes"] = self._pending_reindexes
            self._pending_reindexes = []
        if operation == "search" and self._budget_context is not None:
            data["budget_plan"] = self._budget_context
        if operation == "search" and self._query_started_at is not None:
            data["elapsed_s"] = time.perf_counter() - self._query_started_at
            data["maintenance_totals"] = dict(self.maintenance_totals)
            self._query_attempts[-1].update(status="pass", completed_queries=data["queries"])
            data["query_attempts"] = [dict(item) for item in self._query_attempts]
            data["executed_ann_queries"] = sum(item["completed_queries"] for item in self._query_attempts)
        super()._record(operation, **data)

"""Bounded caller-side scheduling; not an estimate of reclaimability or speed."""

from fractions import Fraction
import math


class ReclaimThreshold:
    """Require both an absolute floor and a fraction of planned input rows."""

    def __init__(self, min_dead=1, min_dead_fraction=0):
        if type(min_dead) is not int or not 1 <= min_dead <= 2147483647:
            raise ValueError("reclaim_min_dead must be an integer in 1..2147483647")
        if (type(min_dead_fraction) not in (int, float) or
                not 0 <= min_dead_fraction <= 1 or not math.isfinite(min_dead_fraction)):
            raise ValueError("reclaim_min_dead_fraction must be a finite number in 0..1")
        self.min_dead = min_dead
        # Interpret the configured decimal, not the float's binary expansion:
        # e.g. ceil(0.07 * 100) must be 7, not 8 from floating multiplication.
        self.fraction = Fraction(str(min_dead_fraction))

    def minimum(self, rows):
        if type(rows) is not int or not 1 <= rows <= 2147483647:
            raise ValueError("reclaim input rows must be an integer in 1..2147483647")
        numerator = rows * self.fraction.numerator
        proportional = (numerator + self.fraction.denominator - 1) // self.fraction.denominator
        return max(self.min_dead, proportional)


class ReclaimPlanner:
    def __init__(self, max_rows, max_runs=1, cooldown=8):
        for name, value, upper in (("max_rows", max_rows, 2147483647),
                                   ("max_runs", max_runs, 256),
                                   ("cooldown", cooldown, 1000000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"reclaim {name} must be an integer in 1..{upper}")
        self.max_rows = max_rows
        self.max_runs = max_runs
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.tick = 0
        self.due = {}

    def plan(self, runs, fresh_ids=()):
        """Charge every selected physical row, including no-op attempts.

        Older due times take precedence over new arrivals. State is local to
        this workload and is rebuilt from authoritative run lists each round.
        Missing/oversized runs are not converted into an unlimited operation.
        """
        current = {}
        for run_id, rows, *_ in runs:
            if type(run_id) not in (str, int):
                raise ValueError("invalid run identity")
            run_id = int(run_id)
            if run_id <= 0 or type(rows) is not int or rows < 0 or run_id in current:
                raise ValueError("invalid or duplicate run metadata")
            current[run_id] = rows
        self.tick += 1
        self.due = {run_id: due for run_id, due in self.due.items() if run_id in current}
        for run_id in current:
            self.due.setdefault(run_id, self.tick)
        for run_id in fresh_ids:
            if int(run_id) in current:
                self.defer(int(run_id))
        remaining = self.max_rows
        jobs, oversized = [], []
        for run_id in sorted(current, key=lambda r: (self.due[r], r)):
            rows = current[run_id]
            if rows > self.max_rows:
                oversized.append({"run_id": run_id, "input_rows": rows})
                continue
            if not rows or self.due[run_id] > self.tick:
                continue
            if len(jobs) >= self.max_runs or rows > remaining:
                continue
            jobs.append((run_id, rows))
            remaining -= rows
            self.defer(run_id)
        return jobs, oversized

    def defer(self, run_id):
        self.due[run_id] = self.tick + self.cooldown

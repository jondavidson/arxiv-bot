# universe.py
from __future__ import annotations
import polars as pl
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Any, Iterable


# ───────────────────────── Config dataclasses ──────────────────────────
@dataclass
class IndexSpec:
    name: str
    constituents_path: str  # parquet with [symbol, effective_from, effective_to]

    @classmethod
    def load(cls, name: str, cfg: Dict[str, Any]) -> "IndexSpec":
        return cls(name=name, constituents_path=cfg["constituentsPath"])


@dataclass
class ThresholdRule:
    lookback_days: int
    min_avg: float


@dataclass
class StabilityRule:
    add_days: int = 0
    remove_days: int = 0


@dataclass
class UniverseSpec:
    name: str
    base_index: str
    liquidity: ThresholdRule
    market_cap: ThresholdRule
    stability: StabilityRule

    @classmethod
    def load(cls, name: str, cfg: Dict[str, Any]) -> "UniverseSpec":
        liq = ThresholdRule(**cfg["liquidity"])
        mcap = ThresholdRule(**cfg["marketCap"])
        stab = StabilityRule(**cfg.get("stableDays", {}))
        return cls(name=name, base_index=cfg["baseIndex"],
                   liquidity=liq, market_cap=mcap, stability=stab)


# ───────────────────────── Universe class ──────────────────────────
class Universe:
    """
    Polars DataFrame with columns [symbol, date] and a boolean column named after
    the universe (True = in universe).
    """
    def __init__(self, name: str, df: pl.DataFrame):
        self.name = name
        self.df = df.sort(["symbol", "date"])

    # -------- set ops --------------------------------------------------
    def _binary_op(self, other: "Universe", how: str, new_name: str) -> "Universe":
        left = self.df.rename({self.name: "flag"})
        right = other.df.rename({other.name: "flag"})
        merged = left.join(right, on=["symbol", "date"], how="outer")
        if how == "union":
            flag = (pl.col(self.name).fill_false() | pl.col(other.name).fill_false())
        elif how == "intersection":
            flag = (pl.col(self.name).fill_false() & pl.col(other.name).fill_false())
        elif how == "diff":
            flag = (pl.col(self.name).fill_false() & (~pl.col(other.name).fill_false()))
        else:
            raise ValueError(how)
        return Universe(new_name, merged.with_columns(flag.alias(new_name))[["symbol", "date", new_name]])

    def __or__(self, other: "Universe"):  return self._binary_op(other, "union", f"{self.name}_OR_{other.name}")
    def __and__(self, other: "Universe"): return self._binary_op(other, "intersection", f"{self.name}_AND_{other.name}")
    def __sub__(self, other: "Universe"): return self._binary_op(other, "diff", f"{self.name}_MINUS_{other.name}")

    # -------- io helpers ----------------------------------------------
    def save(self, path: str): self.df.write_parquet(path)
    @classmethod
    def load(cls, path: str):   # name is inferred from file stem
        df = pl.read_parquet(path)
        name = [c for c in df.columns if c not in {"symbol", "date"}][0]
        return cls(name, df)


# ───────────── Build universe membership from config ─────────────

def compute_universe(spec: UniverseSpec,
                     index_specs: Dict[str, IndexSpec],
                     daily_liq_df: pl.DataFrame,
                     daily_mcap_df: pl.DataFrame,
                     start: date,
                     end: date) -> Universe:
    """
    Returns Universe object with column <spec.name>=bool
    daily_liq_df  : [symbol, date, liquidity]
    daily_mcap_df : [symbol, date, mcap]
    """
    # 1) base index membership table
    index_df = pl.read_parquet(index_specs[spec.base_index].constituents_path)
    # explode date ranges to daily rows within [start,end]
    index_df = (
        index_df
        .with_columns([
            pl.date_range(
                pl.col("effective_from").clip(start, end),
                pl.col("effective_to").clip(start, end),
                "1d",
                eager=True
            ).alias("date")
        ])
        .explode("date")
        .select(["symbol", "date"])
        .unique()
    )

    # 2) join daily liquidity + mcap
    df = (index_df
          .join(daily_liq_df, on=["symbol", "date"])
          .join(daily_mcap_df, on=["symbol", "date"]))

    # 3) compute rolling metrics per symbol
    def _filter_by(rule: ThresholdRule, col: str) -> pl.Series:
        return (
            pl.col(col)
            .rolling_mean(rule.lookback_days, min_periods=rule.lookback_days)
            .over("symbol")
            .alias(f"ok_{col}")
            .cast(pl.Float64) >= rule.min_avg
        )

    df = df.with_columns([
        _filter_by(spec.liquidity, "liquidity"),
        _filter_by(spec.market_cap, "mcap")
    ])
    df = df.with_columns(((pl.col("ok_liquidity") & pl.col("ok_mcap")).alias("raw_member")))

    # 4) apply stability rule (anti-flicker)
    stab = spec.stability
    if stab.add_days or stab.remove_days:
        df = (df
              .sort(["symbol", "date"])
              .with_columns([
                  # rolling window of True counts
                  pl.col("raw_member")
                    .rolling_sum(stab.add_days or 1, min_periods=1)
                    .over("symbol")
                    .alias("add_ok"),
                  pl.col("raw_member")
                    .rolling_sum(stab.remove_days or 1, min_periods=1, center=False, closed="right")
                    .over("symbol")
                    .alias("rem_ok"),
              ])
              .with_columns(
                  ((pl.col("add_ok") > 0) & (pl.col("rem_ok") > 0)).alias(spec.name)
              )
              .select(["symbol", "date", spec.name])
        )
    else:
        df = df.select(["symbol", "date", pl.col("raw_member").alias(spec.name)])

    return Universe(spec.name, df)


# ------------------------ EXAMPLE ---------------------------------
if __name__ == "__main__":
    # tiny fake inputs
    idx = {"SP500": IndexSpec.load("SP500", {"constituentsPath": "sp500_const.parquet"})}
    spec = UniverseSpec.load("TRADING_US", {
        "baseIndex": "SP500",
        "liquidity":  { "lookbackDays": 30, "minAvg": 1_000_000 },
        "marketCap":  { "lookbackDays": 30, "minAvg": 100_000_000 },
        "stableDays": { "addDays": 5, "removeDays": 5 }
    })
    liq = pl.DataFrame({"symbol": [], "date": [], "liquidity": []})
    mcap = pl.DataFrame({"symbol": [], "date": [], "mcap": []})

    uni = compute_universe(spec, idx, liq, mcap,
                           start=date(2025, 1, 1), end=date(2025, 12, 31))
    print(uni.df.head())


# tooling/introspect.py
from __future__ import annotations
import subprocess, re, sys, json, pathlib, shlex

ARG_PAT = re.compile(r"^\\s*(--\\w[\\w-]*)")

def list_args(script_path: str) -> list[str]:
    """Return list of CLI flags supported by a script (argparse/click/typer)."""
    try:
        out = subprocess.check_output(
            [sys.executable, script_path, "--help"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return ARG_PAT.findall(out)
    except Exception as exc:
        print(f"[WARN] {script_path}: {exc}")
        return []

def crawl_scripts(root="scripts") -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for p in pathlib.Path(root).rglob("*.py"):
        mapping[str(p)] = list_args(str(p))
    return mapping

if __name__ == "__main__":
    result = crawl_scripts()
    print(json.dumps(result, indent=2))


from dask.distributed import SSHCluster, Client
from dask import delayed

##############################################################################
# 1 Create the cluster and connect a client
##############################################################################
cluster = SSHCluster(
    hosts=["host1", "host2"],               # ← replace with real hostnames
    connect_options={"username": "me"},     # any SSH parameters you need
    worker_options={"nthreads": 2},         # (optional) per-worker kwargs
)
client = Client(cluster)

# Handy: check the canonical addresses the scheduler knows about
for addr, w in client.scheduler_info()["workers"].items():
    print(f"{w['name']} → {addr}")

##############################################################################
# 2 Define a small dependent pipeline with dask.delayed
##############################################################################
@delayed
def task1(x):
    print("Running task1 on host1")
    return x + 1

@delayed
def task2(y):
    print("Running task2 on host1")
    return y * 10

@delayed
def task3(a, b):
    print("Running task3 on host2")
    return a + b

# Build the graph
a  = task1(7)        # host1
b  = task2(a)        # host1, depends on task1
out = task3(a, b)    # host2, depends on tasks 1 & 2

##############################################################################
# 3 Pin each delayed task to its host and compute the final result
##############################################################################
workers_map = {
    a:  "host1",     # could also be 'tcp://host1:8786'
    b:  "host1",
    out: "host2",
}

future = client.compute(out, workers=workers_map)
result = future.result()
print("Pipeline result:", result)



import dask

with dask.annotate(workers="host1"):
    a = task1(7)
    b = task2(a)

with dask.annotate(workers="host2"):
    out = task3(a, b)

result = out.compute()

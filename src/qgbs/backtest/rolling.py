"""滚动窗口划分（含 purging / embargo，防标签泄漏）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    """滚动窗口规格。"""

    train: str  # 如 "3y"
    valid: str  # 如 "6m"
    test: str  # 如 "1m"
    purge: str  # 如 "5d"，标签重叠剔除
    embargo: str  # 如 "2d"


@dataclass(frozen=True)
class Split:
    """一次滚动划分：各段互不重迭，train/valid 与 test 之间有 purge+embargo。"""

    split_id: int
    train: pd.DatetimeIndex
    valid: pd.DatetimeIndex
    test: pd.DatetimeIndex


def parse_duration(s: str) -> pd.DateOffset:
    """解析窗口字符串：后缀 y/m/w/d 分别对应 年/月/周/日（如 "3y"、"6m"、"5d"）。"""
    unit, n = s[-1], int(s[:-1])
    match unit:
        case "y":
            return pd.DateOffset(years=n)
        case "m":
            return pd.DateOffset(months=n)
        case "w":
            return pd.DateOffset(weeks=n)
        case "d":
            return pd.DateOffset(days=n)
        case _:
            raise ValueError(f"无法解析窗口长度: {s!r}（后缀须为 y/m/w/d）")


def rolling_splits(dates: pd.DatetimeIndex, spec: WindowSpec) -> Iterator[Split]:
    """生成定长滚动 (train, valid, test) 划分序列（方案书 §7.2）。

    训练窗为**定长滚动窗**：每折训练窗口长度恒等于 ``spec.train``，随测试窗
    按 ``spec.test`` 步进整体前移（首折 train 起点恰为 dates[0]）；valid 末端
    与 test 起点之间剔除 purge+embargo 长度的样本（标签重叠剔除 + 隔离带）。
    各折 test 窗位置与旧口径一致：第 k 折 test 起点 = dates[0] + train + valid
    + purge + embargo + k·test。
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    train_off = parse_duration(spec.train)
    valid_off = parse_duration(spec.valid)
    test_off = parse_duration(spec.test)
    purge_off = parse_duration(spec.purge)
    embargo_off = parse_duration(spec.embargo)

    origin = dates[0]
    split_id = 0
    while True:
        test_start = origin + train_off + valid_off + purge_off + embargo_off + split_id * test_off
        test_end = test_start + test_off
        if test_start > dates[-1]:
            break
        gap_start = test_start - purge_off - embargo_off  # purge+embargo 隔离带起点
        train_start = gap_start - valid_off - train_off  # 定长训练窗起点，随测试窗前移
        train = dates[(dates >= train_start) & (dates < gap_start - valid_off)]
        valid = dates[(dates >= gap_start - valid_off) & (dates < gap_start)]
        test = dates[(dates >= test_start) & (dates < test_end)]
        if len(train) == 0 or len(test) == 0:
            break
        yield Split(split_id=split_id, train=train, valid=valid, test=test)
        split_id += 1

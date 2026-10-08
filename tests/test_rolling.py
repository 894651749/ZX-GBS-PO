"""rolling_splits 定长滚动窗语义测试（方案书 §7.2：训练窗定长滚动，非锚定扩张）。"""

from __future__ import annotations

import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits

# 日频日期 + 日单位窗口：日历长度与样本数一一对应，便于断言恒定长度
DATES = pd.date_range("2020-01-01", periods=400, freq="D")
SPEC = WindowSpec(train="30d", valid="10d", test="5d", purge="2d", embargo="1d")


def test_train_window_length_constant() -> None:
    """每折 train 长度恒定（定长滚动），且窗口整体随测试窗前移。"""
    splits = list(rolling_splits(DATES, SPEC))
    assert len(splits) > 2
    # 定长：每折训练窗恒等于 spec.train（30 天）
    assert {len(s.train) for s in splits} == {30}
    # 非锚定扩张：各折 train 起点不同且单调后移，步进 = spec.test
    starts = [s.train[0] for s in splits]
    assert len(set(starts)) == len(starts)
    assert starts == sorted(starts)
    step = starts[1] - starts[0]
    assert step == pd.Timedelta(days=5)
    assert all(b - a == step for a, b in zip(starts, starts[1:]))
    # 首折 train 起点 = 样本首日（与旧口径衔接，test 窗位置不变）
    assert splits[0].train[0] == DATES[0]


def test_valid_length_and_purge_embargo_gap() -> None:
    """valid 长度恒定；valid 末端与 test 起点之间保留 purge+embargo 隔离带。"""
    splits = list(rolling_splits(DATES, SPEC))
    assert {len(s.valid) for s in splits} == {10}
    for s in splits:
        # 三段互不重迭且顺序相接：train → valid →（隔离带）→ test
        assert s.train[-1] < s.valid[0] < s.valid[-1] < s.test[0]
        # valid 与 train 紧邻（无缺口）
        assert s.valid[0] - s.train[-1] == pd.Timedelta(days=1)
        # 隔离带：2d purge + 1d embargo → valid 末端与 test 起点相隔 4 个日频点
        gap_days = (s.test[0] - s.valid[-1]).days
        assert gap_days == 2 + 1 + 1


def test_no_split_when_data_insufficient() -> None:
    """数据长度不足一个完整窗口时不产生任何划分。"""
    short = pd.date_range("2020-01-01", periods=40, freq="D")
    assert list(rolling_splits(short, SPEC)) == []

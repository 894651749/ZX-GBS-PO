"""Parquet 分区存储与数据版本管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataVersion:
    """一次数据落盘的版本标识，用于实验可复现性记录。"""

    version_id: str
    created_at: str  # ISO 时间戳
    source: str  # 数据来源描述
    checksum: str  # 内容校验和


class ParquetStore:
    """按 (数据集, 日期分区) 组织的 Parquet 存储。

    目录约定：``{root}/{dataset}/date={yyyy-mm-dd}/part.parquet``；
    每次写入记录数据版本，供实验 runner 追踪。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write_partition(self, dataset: str, df: pd.DataFrame, partition: str) -> DataVersion:
        """写入一个分区并返回其数据版本。"""
        raise NotImplementedError

    def read_partition(self, dataset: str, partition: str) -> pd.DataFrame:
        """读取指定分区。"""
        raise NotImplementedError

    def list_partitions(self, dataset: str) -> list[str]:
        """列出数据集下所有分区键。"""
        raise NotImplementedError

    def current_version(self, dataset: str) -> DataVersion:
        """返回数据集当前版本。"""
        raise NotImplementedError

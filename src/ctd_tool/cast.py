from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class QCMessage:
    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    status: str = "ok"
    warnings: list[QCMessage] = field(default_factory=list)
    errors: list[QCMessage] = field(default_factory=list)

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(QCMessage(level="warning", code=code, message=message))
        if self.status == "ok":
            self.status = "warning"

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(QCMessage(level="error", code=code, message=message))
        self.status = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "warnings": [m.__dict__ for m in self.warnings],
            "errors": [m.__dict__ for m in self.errors],
        }


@dataclass
class CastHeader:
    source_path: Path
    start_time: str | None
    nquan: int | None
    nvalues: int | None
    interval_label: str | None
    interval_value: float | None
    bad_flag: float | None
    variables: dict[int, str]
    variable_long_names: dict[str, str] = field(default_factory=dict)
    variable_units: dict[str, str | None] = field(default_factory=dict)
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CastRecord:
    header: CastHeader
    data: pd.DataFrame
    canonical_map: dict[str, str]
    is_sv_only: bool
    report: ValidationReport = field(default_factory=ValidationReport)

    @property
    def cast_id(self) -> str:
        return self.header.source_path.stem

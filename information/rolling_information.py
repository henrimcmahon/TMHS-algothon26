from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RollingInformationData:
    matrices: FloatArray
    names: list[str]
    window_ends: IntArray
    window_size: int
    lag: int

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices, dtype=np.float64)
        window_ends = np.asarray(self.window_ends, dtype=np.int64)

        if matrices.ndim != 3:
            raise ValueError(
                "matrices must have shape "
                "(n_windows, n_tickers, n_tickers)."
            )

        number_of_windows, rows, columns = matrices.shape

        if rows != columns:
            raise ValueError("Each information matrix must be square.")

        if rows != len(self.names):
            raise ValueError(
                f"Matrix contains {rows} tickers, but "
                f"{len(self.names)} ticker names were supplied."
            )

        if len(window_ends) != number_of_windows:
            raise ValueError(
                "window_ends must contain one value per matrix."
            )

        if self.window_size < 1:
            raise ValueError("window_size must be positive.")

        if self.lag < 0:
            raise ValueError("lag cannot be negative.")

        if not np.all(np.isfinite(matrices)):
            raise ValueError("matrices contain non-finite values.")

        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "window_ends", window_ends)
        object.__setattr__(
            self,
            "names",
            [str(name) for name in self.names],
        )

    @property
    def number_of_windows(self) -> int:
        return int(self.matrices.shape[0])

    @property
    def number_of_tickers(self) -> int:
        return int(self.matrices.shape[1])

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            matrices=self.matrices,
            names=np.asarray(self.names, dtype=np.str_),
            window_ends=self.window_ends,
            window_size=np.asarray(self.window_size, dtype=np.int64),
            lag=np.asarray(self.lag, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> RollingInformationData:
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        with np.load(input_path, allow_pickle=False) as archive:
            available_keys = set(archive.files)

            matrix_key = cls._first_existing_key(
                available_keys,
                (
                    "matrices",
                    "rolling_matrices",
                    "rolling_mi",
                    "mi_matrices",
                    "rolling_information",
                    "rolling_lagged_information",
                ),
            )

            name_key = cls._first_existing_key(
                available_keys,
                (
                    "names",
                    "tickers",
                    "symbols",
                ),
            )

            end_key = cls._first_existing_key(
                available_keys,
                (
                    "window_ends",
                    "end_days",
                    "indices",
                ),
            )

            if matrix_key is None:
                raise KeyError(
                    "Could not locate the rolling matrices. "
                    f"Archive keys: {sorted(available_keys)}"
                )

            matrices = np.asarray(
                archive[matrix_key],
                dtype=np.float64,
            )

            if name_key is None:
                names = [
                    f"Ticker {index}"
                    for index in range(matrices.shape[1])
                ]
            else:
                names = archive[name_key].astype(str).tolist()

            if end_key is None:
                window_ends = np.arange(
                    matrices.shape[0],
                    dtype=np.int64,
                )
            else:
                window_ends = np.asarray(
                    archive[end_key],
                    dtype=np.int64,
                )

            window_size = cls._load_scalar(
                archive,
                available_keys,
                "window_size",
                default=0,
            )

            lag = cls._load_scalar(
                archive,
                available_keys,
                "lag",
                default=1,
            )

        # Older archives may not contain window_size.
        if window_size < 1:
            window_size = 1

        return cls(
            matrices=matrices,
            names=names,
            window_ends=window_ends,
            window_size=window_size,
            lag=lag,
        )

    @staticmethod
    def _first_existing_key(
        available_keys: set[str],
        candidates: tuple[str, ...],
    ) -> str | None:
        return next(
            (
                key
                for key in candidates
                if key in available_keys
            ),
            None,
        )

    @staticmethod
    def _load_scalar(
        archive: np.lib.npyio.NpzFile,
        available_keys: set[str],
        key: str,
        default: int,
    ) -> int:
        if key not in available_keys:
            return default

        return int(np.asarray(archive[key]).item())
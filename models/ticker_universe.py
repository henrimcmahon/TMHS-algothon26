from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ticker import Ticker


PriceMatrix = NDArray[np.float64]
TickerSelection = str | Sequence[str] | None
TickerResult = Ticker | list[Ticker]


class TickerUniverse:
    """
    Loads and manages all instruments contained in prices.txt.

    The source file has:
        - one ticker per column;
        - one day per row.

    The Algothon evaluator uses the opposite orientation:
        - one ticker per row;
        - one day per column.

    This class exposes both formats where useful.
    """

    def __init__(self, file_path: str | Path = "prices.txt") -> None:
        self.file_path = Path(file_path)

        self._price_frame = self._load_price_frame()
        self._tickers = self._instantiate_all_tickers()

        self._tickers_by_symbol = {
            ticker.symbol: ticker
            for ticker in self._tickers
        }

    def _load_price_frame(self) -> pd.DataFrame:
        """
        Load prices.txt.

        The regular expression separator supports one or more whitespace
        characters between values.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Price file not found: {self.file_path.resolve()}"
            )

        frame = pd.read_csv(
            self.file_path,
            sep=r"\s+",
            dtype=np.float64,
        )

        if frame.empty:
            raise ValueError("Price file contains no price observations")

        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate ticker symbols found: {duplicates}"
            )

        if frame.isna().any().any():
            raise ValueError("Price file contains missing values")

        if (frame <= 0).any().any():
            raise ValueError("Price file contains non-positive prices")

        return frame

    def _instantiate_all_tickers(self) -> list[Ticker]:
        """Create one Ticker object for each column in prices.txt."""
        return [
            Ticker(
                symbol=symbol,
                index=index,
                prices=self._price_frame[symbol].to_numpy(
                    dtype=np.float64,
                    copy=True,
                ),
            )
            for index, symbol in enumerate(self._price_frame.columns)
        ]

    def instantiate(
        self,
        selection: TickerSelection = None,
    ) -> TickerResult:
        """
        Return all tickers, one ticker, or a selected list of tickers.

        Args:
            selection:
                None:
                    Return every Ticker object.

                str:
                    Return one Ticker object.

                Sequence[str]:
                    Return a list containing the requested Ticker objects.

        Examples:
            universe.instantiate()
            universe.instantiate("ALGO")
            universe.instantiate(["ALGO", "AENO", "LSST"])
        """
        if selection is None:
            return list(self._tickers)

        if isinstance(selection, str):
            return self.get(selection)

        if isinstance(selection, Sequence):
            return [self.get(symbol) for symbol in selection]

        raise TypeError(
            "selection must be None, a ticker symbol, "
            "or a sequence of ticker symbols"
        )

    def get(self, symbol: str) -> Ticker:
        """Return one ticker by symbol."""
        normalized_symbol = symbol.strip().upper()

        try:
            return self._tickers_by_symbol[normalized_symbol]
        except KeyError as error:
            available = ", ".join(self.symbols)

            raise KeyError(
                f"Unknown ticker '{symbol}'. "
                f"Available tickers: {available}"
            ) from error

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return all ticker symbols in their original column order."""
        return tuple(ticker.symbol for ticker in self._tickers)

    @property
    def number_of_tickers(self) -> int:
        return len(self._tickers)

    @property
    def number_of_days(self) -> int:
        return len(self._price_frame)

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return a copy in source-file orientation.

        Shape:
            (number_of_days, number_of_tickers)
        """
        return self._price_frame.copy()

    def as_price_matrix(self) -> PriceMatrix:
        """
        Return prices in Algothon evaluator orientation.

        Shape:
            (number_of_tickers, number_of_days)

        This matches the shape of prcSoFar.
        """
        return self._price_frame.to_numpy(
            dtype=np.float64,
            copy=True,
        ).T

    def __len__(self) -> int:
        return len(self._tickers)

    def __iter__(self) -> Iterator[Ticker]:
        return iter(self._tickers)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False

        return symbol.strip().upper() in self._tickers_by_symbol

    def __getitem__(self, symbol: str) -> Ticker:
        return self.get(symbol)
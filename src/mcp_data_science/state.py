import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataStore:
    """Holds named DataFrames in memory with a 'current' active DataFrame.
    Also stores trained ML models."""

    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}
        self._current: str = ""
        self._models: dict[str, dict] = {}

    @property
    def current_name(self) -> str:
        return self._current

    @current_name.setter
    def current_name(self, name: str) -> None:
        if name not in self._frames:
            raise KeyError(f"No dataframe named '{name}'. Available: {list(self._frames.keys())}")
        self._current = name

    def add(self, name: str, df: pd.DataFrame, set_current: bool = True) -> None:
        self._frames[name] = df
        if set_current or not self._current:
            self._current = name
        logger.info("Stored dataframe '%s' (%d rows, %d cols)", name, df.shape[0], df.shape[1])

    def get(self, name: str = "") -> pd.DataFrame:
        resolved = name if name else self._current
        if not resolved:
            raise ValueError("No dataframe loaded. Use load_csv first.")
        if resolved not in self._frames:
            raise KeyError(f"No dataframe named '{resolved}'. Available: {list(self._frames.keys())}")
        return self._frames[resolved]

    def set(self, name: str, df: pd.DataFrame) -> None:
        if not name:
            name = self._current
        self._frames[name] = df

    def resolve_name(self, name: str = "") -> str:
        resolved = name if name else self._current
        if not resolved:
            raise ValueError("No dataframe loaded. Use load_csv first.")
        return resolved

    def list_names(self) -> list[str]:
        return list(self._frames.keys())

    def remove(self, name: str) -> None:
        if name in self._frames:
            del self._frames[name]
            if self._current == name:
                self._current = next(iter(self._frames), "")

    def copy(self, source: str, target: str) -> None:
        if source not in self._frames:
            raise KeyError(f"No dataframe named '{source}'. Available: {list(self._frames.keys())}")
        self._frames[target] = self._frames[source].copy()
        logger.info("Copied '%s' -> '%s'", source, target)

    # ── Model storage ───────────────────────────────────────────────

    def add_model(self, name: str, model_info: dict) -> None:
        self._models[name] = model_info
        logger.info("Stored model '%s' (type=%s)", name, model_info.get("type", "unknown"))

    def get_model(self, name: str) -> dict:
        if name not in self._models:
            raise KeyError(f"No model named '{name}'. Available: {list(self._models.keys())}")
        return self._models[name]

    def list_model_names(self) -> list[str]:
        return list(self._models.keys())

    def remove_model(self, name: str) -> None:
        if name in self._models:
            del self._models[name]

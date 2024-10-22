from abc import ABC, abstractmethod
from numpy import ndarray
from typing import List
from rank2plan.types import Pair, Matrix


class LpUnderlying(ABC):
    @abstractmethod
    def fit(self, X_tilde: Matrix, pairs: List[Pair], save_state=False) -> None:
        pass

    @abstractmethod
    def predict(self, X: Matrix) -> ndarray:
        pass

    @abstractmethod
    def weights(self) -> ndarray:
        pass

    @abstractmethod
    def refit_with_C_value(self, X_tilde: Matrix, C: float, save_state=False) -> None:
        pass

    @abstractmethod
    def clear_state(self) -> None:
        pass

    @property
    @abstractmethod
    def C(self) -> float:
        pass

    @C.setter
    @abstractmethod
    def C(self, value: float) -> None:
        pass

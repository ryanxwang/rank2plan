from abc import ABC, abstractmethod
from typing import List, Optional
from rank2plan.types import Pair
from numpy import ndarray


class Model(ABC):
    @abstractmethod
    def fit(self, X: ndarray, pairs: List[Pair]) -> None:
        """Fit the model to the data.

        Args:
            X (ndarray): The data matrix, shape (n_samples, n_features).
            pairs (List[Pair]): The list of ranking pairs, shape (n_pairs,).
        """
        pass

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """Pointwise evaluation of the model, where lower values are better.

        Args:
            X (ndarray): The data matrix, shape (n_samples, n_features).

        Returns:
            npt.ArrayLike: The pointwise evaluation of the model, shape
            (n_samples,).
        """
        pass

    @abstractmethod
    def weights(self) -> Optional[ndarray]:
        """Get the weights of the model.

        Returns:
            ndarray: The weights of the model, shape (n_features,). None if the
            model has not been fitted.
        """
        pass

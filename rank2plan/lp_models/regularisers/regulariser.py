from abc import ABC, abstractmethod


class Regulariser(ABC):
    @abstractmethod
    def get_C_value(
        self, main_objective: float, regularisation_objective: float
    ) -> float:
        """Register the main and regularisation objectives and get the new C
        value.

        Args:
            main_objective (float): The main objective value, not accounting for
            regularisation (C value)

            regularisation_objective (float): The regularisation objective value

        Returns:
            float: New value for C
        """
        pass

from rank2plan.lp_models.regularisers import Regulariser


class DynamicRegulariser(Regulariser):
    def __init__(self, target: float):
        self.target = target

    def get_C_value(
        self, main_objective: float, regularisation_objective: float
    ) -> float:
        raise NotImplementedError("Dynamic regularisation not implemented yet")

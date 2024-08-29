from rank2plan.lp_models.regularisers import Regulariser


class ConstantRegulariser(Regulariser):
    def __init__(self, C: float):
        self.C = C

    def get_C_value(
        self, main_objective: float, regularisation_objective: float
    ) -> float:
        return self.C

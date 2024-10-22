from rank2plan import Pair, LossType, PenalisationType, Matrix
from rank2plan.lp_models.constraint_column_generation.smoothing_hinge_loss import (
    loop_smoothing_hinge_loss_columns_samples_restricted,
    loop_smoothing_hinge_loss_samples_restricted,
)
from rank2plan.lp_models.objective_values import (
    compute_main_objective,
    compute_regularisation_objective,
    compute_overall_objective,
)
from rank2plan.lp_models.utils import sparseLpDot
from rank2plan.lp_models.lp_underlying import LpUnderlying
from pulp import LpSolver, LpProblem, LpMinimize, LpVariable, lpSum, lpDot, LpConstraint
from typing import List, Optional, Tuple
from numpy import ndarray
import numpy as np
from time import time
import logging
import random
import math
from dataclasses import dataclass
from scipy.sparse import csr_matrix, issparse

LOGGER = logging.getLogger(__name__)


@dataclass
class FitState:
    N: int
    P: int
    # we don't save X_tilde here to save memory
    pairs: List[Pair]
    problem: LpProblem
    constraint_indices: List[int]
    feature_indices: List[int]


class ConstraintColumnModel(LpUnderlying):
    def __init__(
        self, solver: LpSolver, C: float, tol: float, no_feature_sampling=False
    ) -> None:
        """Don't use this directly, use LpModel instead."""
        if solver.mip == True:
            raise ValueError("solver must be configured for LP, use mip=False")
        self.solver = solver
        self._C = C
        self.tol = tol
        self.no_feature_sampling = no_feature_sampling
        self.state: Optional[FitState] = None
        self._weights: Optional[ndarray] = None

    def clear_state(self) -> None:
        self.state = None

    @property
    def C(self) -> float:
        return self._C

    @C.setter
    def C(self, value: float) -> None:
        self._C = value

    def fit(self, X_tilde: Matrix, pairs: List[Pair], save_state=False) -> None:
        start = time()
        if self.no_feature_sampling:
            constraint_indices = _init_constraint_sampling_smoothing(
                X_tilde, pairs, self._C
            )
            assert X_tilde.shape is not None
            feature_indices = list(range(X_tilde.shape[1]))
        else:
            constraint_indices, feature_indices = (
                _init_constraint_column_sampling_smoothing(X_tilde, pairs, self._C)
            )
        LOGGER.info(
            f"Initial constraint and column sampling done in {time() - start:.3f} seconds"
        )

        problem = self._build_subproblem(
            X_tilde, pairs, constraint_indices, feature_indices, None
        )

        assert X_tilde.shape is not None
        N, P = X_tilde.shape
        self.state = FitState(N, P, pairs, problem, constraint_indices, feature_indices)
        self._fit(X_tilde)
        if not save_state:
            self.state = None

    def refit_with_C_value(self, X_tilde: Matrix, C: float, save_state=False) -> None:
        # this X_tilde must be the same as the one used in the initial fit, we
        # don't save it to save memory
        assert self.state is not None
        LOGGER.info(f"Refitting with C value changed from {self._C} to {C}")
        self._C = C
        self.state.problem = self._rebuild_subproblem_objective(self.state.problem)
        self._fit(X_tilde)
        if not save_state:
            self.state = None

    def predict(self, X: Matrix) -> ndarray:
        assert self._weights is not None
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights

    def _fit(self, X_tilde: Matrix) -> None:
        assert isinstance(X_tilde, np.ndarray) or isinstance(X_tilde, csr_matrix)
        is_sparse = issparse(X_tilde)
        assert self.state is not None
        N, P = self.state.N, self.state.P
        features_to_check = list(set(range(P)) - set(self.state.feature_indices))
        constraint_to_check = list(set(range(N)) - set(self.state.constraint_indices))
        gs = np.array([pair.gap * pair.sample_weight for pair in self.state.pairs])

        cur_iter = 0
        while True:
            cur_iter += 1

            start = time()
            self.state.problem.solve(self.solver)
            LOGGER.info(f"Subproblem solved in {time() - start:.3f} seconds")
            LOGGER.info(f"Current objective: {self.state.problem.objective.value()}")  # type: ignore

            if (
                len(self.state.feature_indices) == P
                and len(self.state.constraint_indices) == N
            ):
                LOGGER.info("Finishing as all features and constraints are used")
                break

            constraints: List[LpConstraint] = [
                self.state.problem.constraints[_constraint_name(i)]
                for i in self.state.constraint_indices
            ]
            dual_values = np.array([constraint.pi for constraint in constraints])

            variable_dict = self.state.problem.variablesDict()
            beta_plus = [
                variable_dict[_beta_plus_name(feature_id)]
                for feature_id in self.state.feature_indices
            ]
            beta_minus = [
                variable_dict[_beta_minus_name(feature_id)]
                for feature_id in self.state.feature_indices
            ]
            beta = np.array(
                [
                    beta_plus[i].varValue - beta_minus[i].varValue
                    for i in range(len(self.state.feature_indices))
                ]
            )

            # Reduced costs for features
            X_tilde_reduced: Matrix = X_tilde[self.state.constraint_indices, :][  # type: ignore
                :, features_to_check
            ]
            reduced_costs_features: ndarray = (1 / self._C) * np.ones(
                len(features_to_check)
            ) - np.abs(
                np.dot(X_tilde_reduced.T, dual_values)
                if not is_sparse
                else (X_tilde_reduced.T.dot(dual_values))
            )
            violated_features = np.array(features_to_check)[
                reduced_costs_features < -self.tol
            ]

            if violated_features.shape[0] > 400:
                indices = np.argsort(reduced_costs_features)[:400]
                violated_features = np.array(features_to_check)[indices]

            X_tilde_reduced: Matrix = X_tilde[:, self.state.feature_indices][  # type: ignore
                constraint_to_check, :
            ]
            reduced_costs_constraints: ndarray = (
                gs[constraint_to_check] - np.dot(X_tilde_reduced, beta)  # type: ignore
                if not is_sparse
                else gs[constraint_to_check] - X_tilde_reduced.dot(beta)
            )
            violated_constraints = np.array(constraint_to_check)[
                reduced_costs_constraints > self.tol
            ]

            if len(violated_constraints) == 0 and len(violated_features) == 0:
                LOGGER.info("Finishing as no violated constraints or features")
                break

            if len(violated_features) > 0:
                LOGGER.info(f"Adding {len(violated_features)} features")
                LOGGER.info(
                    f"Most negative feature reduced cost: {np.min(reduced_costs_features)}"
                )
                self.state.problem = self._add_features_to_subproblem(
                    X_tilde,
                    self.state.pairs,
                    self.state.problem,
                    violated_features,
                    self.state.constraint_indices,
                )
                self.state.feature_indices += list(violated_features)
                features_to_check = list(
                    set(range(P)) - set(self.state.feature_indices)
                )

                variable_dict = self.state.problem.variablesDict()
                beta_plus += [
                    variable_dict[_beta_plus_name(feature_id)]
                    for feature_id in violated_features
                ]
                beta_minus += [
                    variable_dict[_beta_minus_name(feature_id)]
                    for feature_id in violated_features
                ]

            if len(violated_constraints) > 0:
                LOGGER.info(f"Adding {len(violated_constraints)} constraints")
                LOGGER.info(
                    f"Most violated constraint: {np.max(reduced_costs_constraints)}"
                )
                self.state.problem = self._add_constraints_to_subproblem(
                    X_tilde,
                    self.state.pairs,
                    self.state.problem,
                    violated_constraints,
                    self.state.feature_indices,
                    beta_plus,
                    beta_minus,
                )
                self.state.constraint_indices.extend(violated_constraints)
                constraint_to_check = list(
                    set(constraint_to_check) - set(violated_constraints)
                )

        variable_dict = self.state.problem.variablesDict()
        beta_plus = np.array(
            [
                (
                    variable_dict[_beta_plus_name(feature_id)].varValue
                    if feature_id in self.state.feature_indices
                    else 0
                )
                for feature_id in range(P)
            ]
        )
        beta_minus = np.array(
            [
                (
                    variable_dict[_beta_minus_name(feature_id)].varValue
                    if feature_id in self.state.feature_indices
                    else 0
                )
                for feature_id in range(P)
            ]
        )
        self._weights = beta_plus - beta_minus
        LOGGER.info(
            f"Finished constraint and column generation in {cur_iter} iterations, using {time() - start:.3f} seconds"
        )
        LOGGER.info(
            f"Final support size {len(self.state.feature_indices)}, of which {len(np.where(self._weights != 0)[0])} are non-zero"
        )
        LOGGER.info(f"Pulp objective: {self.state.problem.objective.value()}")  # type: ignore
        LOGGER.info(
            f"Overall objective: {compute_overall_objective(X_tilde, self.state.pairs, self._weights, self._C)}"
        )
        LOGGER.info(
            f"Main objective: {compute_main_objective(X_tilde, self.state.pairs, self._weights)}"
        )
        LOGGER.info(
            f"Regularisation objective: {compute_regularisation_objective(self._weights)}"
        )

    def _build_subproblem(
        self,
        X_tilde: Matrix,
        pairs: List[Pair],
        constraint_indices: List[int],
        feature_indices: List[int],
        warm_start: Optional[ndarray],
    ) -> LpProblem:
        start = time()
        assert X_tilde.shape is not None
        N, P = X_tilde.shape
        is_sparse = issparse(X_tilde)
        N_constraints = len(constraint_indices)
        P_features = len(feature_indices)
        sample_weights = [
            pairs[constraint_id].sample_weight for constraint_id in constraint_indices
        ]
        LOGGER.info(
            f"Subproblem with {N_constraints} constraints and {P_features} features"
        )

        problem = LpProblem("ConstraintColumnGenerationSubproblem", LpMinimize)

        # Hinge loss
        xi = [
            LpVariable(
                _xi_name(constraint_id, pairs[constraint_id].i, pairs[constraint_id].j),
                lowBound=0,
            )
            for constraint_id in constraint_indices
        ]

        # Beta
        beta_plus = [
            LpVariable(_beta_plus_name(feature_id), lowBound=0)
            for feature_id in feature_indices
        ]
        beta_minus = [
            LpVariable(_beta_minus_name(feature_id), lowBound=0)
            for feature_id in feature_indices
        ]

        problem += (
            self._C * lpDot(sample_weights, xi) + lpSum(beta_plus) + lpSum(beta_minus)
        )

        # The selected constraints using the selected features
        X_tilde_reduced: Matrix = X_tilde[:, feature_indices]  # type: ignore
        for i, constraint_id in enumerate(constraint_indices):
            pair = pairs[constraint_id]
            if is_sparse:
                problem.addConstraint(
                    (
                        sparseLpDot(X_tilde_reduced.getrow(constraint_id), beta_plus)  # type: ignore
                        - sparseLpDot(X_tilde_reduced.getrow(constraint_id), beta_minus)  # type: ignore
                        >= pair.gap * pair.sample_weight - pair.sample_weight * xi[i]
                    ),
                    name=f"constraint_{constraint_id}",
                )
            else:
                problem.addConstraint(
                    (
                        lpDot(X_tilde_reduced[constraint_id], beta_plus)
                        - lpDot(X_tilde_reduced[constraint_id], beta_minus)
                        >= pair.gap * pair.sample_weight - pair.sample_weight * xi[i]
                    ),
                    name=f"constraint_{constraint_id}",
                )

        if warm_start is not None:
            for i, feature_id in enumerate(feature_indices):
                beta_plus[i].setInitialValue(max(0, warm_start[feature_id]))
                beta_minus[i].setInitialValue(max(0, -warm_start[feature_id]))

        LOGGER.info(f"Subproblem built in {time() - start:.3f} seconds")
        return problem

    def _add_features_to_subproblem(
        self,
        X_tilde: Matrix,
        pairs: List[Pair],
        problem: LpProblem,
        violated_features: ndarray,
        current_constraints: List[int],
    ) -> LpProblem:
        is_sparse = issparse(X_tilde)
        # add new features
        added_beta_plus = []
        added_beta_minus = []

        for feature_id in violated_features:
            added_beta_plus.append(LpVariable(_beta_plus_name(feature_id), lowBound=0))
            added_beta_minus.append(
                LpVariable(_beta_minus_name(feature_id), lowBound=0)
            )

        X_tilde_reduced: Matrix = X_tilde[:, violated_features]  # type: ignore
        problem.addVariables(added_beta_plus)
        problem.addVariables(added_beta_minus)
        for constraint_id in current_constraints:
            old_constraint: LpConstraint = problem.constraints[
                _constraint_name(constraint_id)
            ]
            if is_sparse:
                updated_constraint = (
                    old_constraint
                    + sparseLpDot(
                        X_tilde_reduced.getrow(constraint_id), added_beta_plus  # type: ignore
                    )
                    - sparseLpDot(
                        X_tilde_reduced.getrow(constraint_id), added_beta_minus  # type: ignore
                    )
                )
            else:
                updated_constraint = (
                    old_constraint
                    + lpDot(X_tilde_reduced[constraint_id], added_beta_plus)
                    - lpDot(X_tilde_reduced[constraint_id], added_beta_minus)
                )
            problem.constraints[_constraint_name(constraint_id)] = updated_constraint
            problem.modifiedConstraints.append(updated_constraint)

        problem.setObjective(
            problem.objective + lpSum(added_beta_plus) + lpSum(added_beta_minus)
        )
        LOGGER.info("Features added and objective updated")

        return problem

    def _add_constraints_to_subproblem(
        self,
        X_tilde: Matrix,
        pairs: List[Pair],
        problem: LpProblem,
        violated_constraints: ndarray,
        current_features: List[int],
        beta_plus: List[LpVariable],
        beta_minus: List[LpVariable],
    ):
        is_sparse = issparse(X_tilde)
        xis_to_add = []
        sample_weights_to_add = []
        X_reduced: Matrix = X_tilde[:, current_features]  # type: ignore

        for violated_constraint in violated_constraints:
            pair = pairs[violated_constraint]
            xi_violated = LpVariable(
                f"xi_{violated_constraint}_{pair.i}_{pair.j}", lowBound=0
            )
            if is_sparse:
                problem.addConstraint(
                    (
                        sparseLpDot(X_reduced.getrow(violated_constraint), beta_plus)  # type: ignore
                        - sparseLpDot(X_reduced.getrow(violated_constraint), beta_minus)  # type: ignore
                        >= pair.gap * pair.sample_weight
                        - pair.sample_weight * xi_violated
                    ),
                    name=_constraint_name(violated_constraint),
                )
            else:
                problem.addConstraint(
                    (
                        lpDot(X_reduced[violated_constraint], beta_plus)
                        - lpDot(X_reduced[violated_constraint], beta_minus)
                        >= pair.gap * pair.sample_weight
                        - pair.sample_weight * xi_violated
                    ),
                    name=_constraint_name(violated_constraint),
                )
            xis_to_add.append(xi_violated)
            sample_weights_to_add.append(pair.sample_weight)

        problem.setObjective(
            problem.objective + lpDot(xis_to_add, sample_weights_to_add) * self._C
        )
        LOGGER.info("Constraints added and objective updated")

        return problem

    def _rebuild_subproblem_objective(self, problem: LpProblem) -> LpProblem:
        assert self.state is not None
        variables_dict = problem.variablesDict()
        beta_plus = [
            variables_dict[_beta_plus_name(feature_id)]
            for feature_id in self.state.feature_indices
        ]
        beta_minus = [
            variables_dict[_beta_minus_name(feature_id)]
            for feature_id in self.state.feature_indices
        ]
        xis = [
            variables_dict[
                _xi_name(
                    constraint_id,
                    self.state.pairs[constraint_id].i,
                    self.state.pairs[constraint_id].j,
                )
            ]
            for constraint_id in self.state.constraint_indices
        ]
        sample_weights = [
            self.state.pairs[constraint_id].sample_weight
            for constraint_id in self.state.constraint_indices
        ]
        problem.setObjective(
            self._C * lpDot(xis, sample_weights) + lpSum(beta_plus) + lpSum(beta_minus)
        )
        LOGGER.info("Rebuilt subproblem objective")
        return problem


def _beta_plus_name(feature_id: int) -> str:
    return f"beta_plus_{feature_id}"


def _beta_minus_name(feature_id: int) -> str:
    return f"beta_minus_{feature_id}"


def _xi_name(constraint_id: int, i: int, j: int) -> str:
    return f"xi_{constraint_id}_{i}_{j}"


def _constraint_name(constraint_id: int) -> str:
    return f"constraint_{constraint_id}"


def _init_constraint_column_sampling_smoothing(
    X_tilde: Matrix, pairs: List[Pair], C: float
) -> Tuple[List[int], List[int]]:
    pairs_np = np.array(pairs)
    assert X_tilde.shape is not None
    N, P = X_tilde.shape
    is_sparse = issparse(X_tilde)
    if is_sparse:
        rho = (1 / C) / np.max(abs(X_tilde).sum(axis=0))
    else:
        assert isinstance(X_tilde, np.ndarray)
        rho = (1 / C) / np.max(np.sum(np.abs(X_tilde), axis=0))

    N0 = 5 * int(math.sqrt(N))
    P0 = 5 * int(math.sqrt(P))

    if N0 >= N or P0 >= P:
        LOGGER.warning(
            "Using no samples and features for initial sampling, as N or P is too small"
        )
        return [], []

    old_beta_averaged = np.ones(P)
    beta_averaged = np.zeros(P)
    delta_l2_diff_mean = 1e8

    k = 0
    while delta_l2_diff_mean > 5e-1 and k < int(N / N0):
        k += 1
        LOGGER.info(f"Sampling iteration {k}")
        LOGGER.info(f"Difference in l2 norm: {delta_l2_diff_mean}")

        subset = np.sort(random.sample(range(N), N0))
        X_tilde_reduced: Matrix = X_tilde[subset]  # type: ignore
        pairs_reduced = pairs_np[subset]

        # Correlation screening
        argsort_columns = np.argsort(
            np.abs(
                np.sum(X_tilde_reduced, axis=0)  # type: ignore
                if not is_sparse
                else X_tilde_reduced.sum(axis=0)
            )
        )
        index_columns = np.ravel(argsort_columns[::-1][:P0])

        X_tilde_reduced: Matrix = X_tilde_reduced[:, index_columns]  # type: ignore

        if is_sparse:
            alpha_sample = rho * np.max(abs(X_tilde_reduced).sum(axis=0))
        else:
            assert isinstance(X_tilde_reduced, np.ndarray)
            alpha_sample = rho * np.max(np.sum(np.abs(X_tilde_reduced), axis=0))

        tau_max = 0.2
        n_loop = 20
        n_iter = 100
        beta_smoothing_reduced = loop_smoothing_hinge_loss_columns_samples_restricted(
            LossType.Hinge,
            PenalisationType.L1,
            X_tilde_reduced,
            pairs_reduced,
            alpha_sample,
            tau_max,
            n_loop,
            n_iter,
        )

        beta_sample = np.zeros(P)
        for i, index in enumerate(index_columns):
            beta_sample[index] = beta_smoothing_reduced[i]

        old_beta_averaged = np.copy(beta_averaged)
        beta_averaged += np.array(beta_sample)
        delta_l2_diff_mean = np.linalg.norm(
            1.0 / max(1, k) * beta_averaged - 1.0 / max(1, k - 1) * old_beta_averaged
        )

    beta_averaged *= 1.0 / k

    max_n_cols = 200
    index_columns_smoothing = np.where(beta_averaged != 0)[0]
    LOGGER.info(f"Len support primal: {len(index_columns_smoothing)}")

    if len(index_columns_smoothing) > max_n_cols:
        argsort_columns = np.argsort(np.abs(beta_averaged))
        index_columns_smoothing = argsort_columns[::-1][:max_n_cols]
        LOGGER.info(f"Reduced primal support size to {max_n_cols}")

    gs = np.array([pair.gap * pair.sample_weight for pair in pairs])
    constraints = (
        gs - np.dot(X_tilde, beta_averaged)  # type: ignore
        if not is_sparse
        else gs - X_tilde.dot(beta_averaged)
    )
    index_samples_smoothing = np.arange(N)[constraints >= 0]
    LOGGER.info(f"Len dual smoothing: {len(index_samples_smoothing)}")

    return list(index_samples_smoothing), list(index_columns_smoothing)


def _init_constraint_sampling_smoothing(
    X_tilde: Matrix, pairs: List[Pair], C: float, is_restricted=True
) -> List[int]:
    """Initial sampling for constraint generation using smoothing.

    Args:
        X_tilde (Matrix): The X_tilde matrix, shape (n_pairs, n_features).
        pairs (List[Pair]): The list of pairs, shape (n_pairs,).
        C (float): The regularisation parameter.
        is_restricted (bool, optional): Not sure yet :(. Defaults to True.

    Returns:
        List[int]: the initial set of constraints
    """
    assert X_tilde.shape is not None
    is_sparse = issparse(X_tilde)

    pairs_np = np.array(pairs)
    N, P = X_tilde.shape
    if isinstance(X_tilde, csr_matrix):
        rho = (1 / C) / np.max(abs(X_tilde).sum(axis=0))
    else:
        assert isinstance(X_tilde, np.ndarray)
        rho = (1 / C) / np.max(
            np.sum(np.abs(X_tilde), axis=0)
        )  # lambda (1/C) relative to norms

    N0 = int(min(10 * P, 5 * math.sqrt(N)))
    if N0 < 3:
        LOGGER.warning("Not enough samples to perform initial sampling, returning none")
        return []

    LOGGER.info("Finding initial solution using first-order method")

    tau_max = 0.1
    n_loop = 20
    n_iter = 20

    # result
    old_beta_averaged = -np.ones(P)
    beta_averaged = np.zeros(P)
    delta_variance = 1e6

    k = 0
    while delta_variance > 5e-2 and k < int(N / N0):
        k += 1
        LOGGER.info(f"Sample number: {k}")
        LOGGER.info(f"Difference variance: {delta_variance}")

        subset = np.sort(random.sample(range(N), N0))
        X_tilde_reduced: Matrix = X_tilde[subset]  # type: ignore
        pairs_reduced = pairs_np[subset]

        if is_sparse:
            alpha_sample = rho * np.max(abs(X_tilde_reduced).sum(axis=0))
        else:
            assert isinstance(X_tilde_reduced, np.ndarray)
            alpha_sample = rho * np.max(np.sum(np.abs(X_tilde_reduced), axis=0))

        if is_restricted:
            n_loop = 10
            _, beta_sample = loop_smoothing_hinge_loss_samples_restricted(
                LossType.Hinge,
                PenalisationType.L1,
                pairs_reduced,
                X_tilde_reduced,
                alpha_sample,
                tau_max,
                n_loop,
                n_iter,
            )
        else:
            raise NotImplementedError("Unrestricted sampling not implemented yet.")
            # _, _, _, beta_sample, beta0_sample = loop_smoothing_hinge_loss(
            #     LossType.Hinge,
            #     PenalisationType.L1,
            #     X,
            #     pairs_reduced,
            #     alpha_sample,
            #     tau_max,
            #     n_loop,
            #     n_iter,
            # )
            # beta_sample = np.concatenate([beta_sample, np.array([beta0_sample])])

        old_beta_averaged = np.copy(beta_averaged)
        beta_averaged += np.array(beta_sample)
        delta_variance = np.linalg.norm(
            1.0 / max(1, k) * beta_averaged - 1.0 / max(1, k - 1) * old_beta_averaged
        )

    # ---Determine set of constraints
    beta_averaged *= N0 / float(N)

    gs = np.array([pair.gap * pair.sample_weight for pair in pairs])

    constraints = (
        gs - (np.dot(X_tilde, beta_averaged))  # type: ignore
        if not is_sparse
        else (gs - X_tilde.dot(beta_averaged))
    )
    idx_samples_smoothing = np.arange(N)[constraints >= 0]

    LOGGER.info("Finished initial sampling")
    LOGGER.info(f"Len dual smoothing: {len(idx_samples_smoothing)}")

    return list(idx_samples_smoothing)

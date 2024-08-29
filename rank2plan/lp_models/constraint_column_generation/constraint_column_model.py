from rank2plan import Model, Pair, LossType, PenalisationType
from rank2plan.lp_models.constraint_column_generation.smoothing_hinge_loss import (
    loop_smoothing_hinge_loss_columns_samples_restricted,
)
from rank2plan.lp_models.objective_values import (
    compute_main_objective,
    compute_regularisation_objective,
    compute_overall_objective,
)
from rank2plan.lp_models.constraint_column_generation.utils import compute_X_tilde
from pulp import LpSolver, LpProblem, LpMinimize, LpVariable, lpSum, lpDot, LpConstraint
from typing import List, Optional, Tuple
from numpy import ndarray
import numpy as np
from time import time
import logging
import random
import math

LOGGER = logging.getLogger(__name__)


class ConstraintColumnModel(Model):
    def __init__(
        self,
        solver: LpSolver,
        C: float,
        tol: float,
        dynamic_regularisation_target: Optional[float],
    ) -> None:
        """Don't use this directly, use LpModel instead."""
        if solver.mip == True:
            raise ValueError("solver must be configured for LP, use mip=False")
        self.solver = solver
        self.C = C
        self.tol = tol
        self.dynamically_regularise = dynamic_regularisation_target is not None
        self.omega = dynamic_regularisation_target
        self._weights = None

    def fit(self, X: ndarray, pairs: List[Pair]) -> None:

        start = time()
        X_tilde = compute_X_tilde(X, pairs)
        LOGGER.info(f"Computed X_tilde in {time() - start:.2f}s")

        start = time()
        constraint_indices, feature_indices = (
            _init_constraint_column_sampling_smoothing(X_tilde, pairs, self.C)
        )
        LOGGER.info(
            f"Initial constraint and column sampling done in {time() - start:.3f} seconds"
        )

        problem = self._build_subproblem(
            X_tilde, pairs, constraint_indices, feature_indices, None
        )

        N, P = X_tilde.shape
        features_to_check = list(set(range(P)) - set(feature_indices))
        constraint_to_check = list(set(range(N)) - set(constraint_indices))
        gs = np.array([pair.gap * pair.sample_weight for pair in pairs])

        cur_iter = 0
        while True:
            cur_iter += 1

            # LOGGER.info(problem)
            start = time()
            problem.solve(self.solver)
            LOGGER.info(f"Subproblem solved in {time() - start:.3f} seconds")
            LOGGER.info(f"Current objective: {problem.objective.value()}")  # type: ignore

            if len(feature_indices) == P and len(constraint_indices) == N:
                LOGGER.info("Finishing as all features and constraints are used")
                break

            constraints: List[LpConstraint] = [
                problem.constraints[f"constraint_{i}"] for i in constraint_indices
            ]
            dual_values = np.array([constraint.pi for constraint in constraints])

            variable_dict = problem.variablesDict()
            beta_plus = [
                variable_dict[f"beta_plus_{feature_id}"]
                for feature_id in feature_indices
            ]
            beta_minus = [
                variable_dict[f"beta_minus_{feature_id}"]
                for feature_id in feature_indices
            ]
            beta = np.array(
                [
                    beta_plus[i].varValue - beta_minus[i].varValue
                    for i in range(len(feature_indices))
                ]
            )

            # Reduced costs for features
            X_tilde_reduced = X_tilde[constraint_indices, :][:, features_to_check]
            reduced_costs_features = (1 / self.C) * np.ones(
                len(features_to_check)
            ) - np.abs(np.dot(X_tilde_reduced.T, dual_values))
            violated_features = np.array(features_to_check)[
                reduced_costs_features < -self.tol
            ]

            if violated_features.shape[0] > 400:
                indices = np.argsort(reduced_costs_features)[:400]
                violated_features = np.array(features_to_check)[indices]

            X_tilde_reduced = X_tilde[:, feature_indices][constraint_to_check, :]
            reduced_costs_constraints = gs[constraint_to_check] - np.dot(
                X_tilde_reduced, beta
            )
            violated_constraints = np.array(constraint_to_check)[
                reduced_costs_constraints > self.tol
            ]

            if len(violated_constraints) == 0 and len(violated_features) == 0:
                LOGGER.info("Finishing as no violated constraints or features")
                break
            # if len(violated_constraints) == 0:
            #     LOGGER.info("Finishing as no violated constraints")
            #     break

            if len(violated_features) > 0:
                LOGGER.info(f"Adding {len(violated_features)} features")
                LOGGER.info(
                    f"Most negative feature reduced cost: {np.min(reduced_costs_features)}"
                )
                problem = self._add_features_to_subproblem(
                    X_tilde,
                    pairs,
                    problem,
                    violated_features,
                    constraint_indices,
                )
                feature_indices += list(violated_features)
                features_to_check = list(set(range(P)) - set(feature_indices))

                variable_dict = problem.variablesDict()
                beta_plus += [
                    variable_dict[f"beta_plus_{feature_id}"]
                    for feature_id in violated_features
                ]
                beta_minus += [
                    variable_dict[f"beta_minus_{feature_id}"]
                    for feature_id in violated_features
                ]

            if len(violated_constraints) > 0:
                LOGGER.info(f"Adding {len(violated_constraints)} constraints")
                LOGGER.info(
                    f"Most violated constraint: {np.max(reduced_costs_constraints)}"
                )
                problem = self._add_constraints_to_subproblem(
                    X_tilde,
                    pairs,
                    problem,
                    violated_constraints,
                    feature_indices,
                    beta_plus,
                    beta_minus,
                )
                constraint_indices.extend(violated_constraints)
                constraint_to_check = list(
                    set(constraint_to_check) - set(violated_constraints)
                )

        variable_dict = problem.variablesDict()
        beta_plus = np.array(
            [
                (
                    variable_dict[f"beta_plus_{feature_id}"].varValue
                    if feature_id in feature_indices
                    else 0
                )
                for feature_id in range(P)
            ]
        )
        beta_minus = np.array(
            [
                (
                    variable_dict[f"beta_minus_{feature_id}"].varValue
                    if feature_id in feature_indices
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
            f"Final support size {len(feature_indices)}, of which {len(np.where(self._weights != 0)[0])} are non-zero"
        )
        LOGGER.info(f"Pulp objective: {problem.objective.value()}")  # type: ignore
        LOGGER.info(
            f"Overall objective: {compute_overall_objective(X, pairs, self._weights, self.C)}"
        )
        LOGGER.info(
            f"Main objective: {compute_main_objective(X, pairs, self._weights)}"
        )
        LOGGER.info(
            f"Regularisation objective: {compute_regularisation_objective(self._weights)}"
        )

    def predict(self, X: ndarray) -> ndarray:
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights

    def _build_subproblem(
        self,
        X_tilde: ndarray,
        pairs: List[Pair],
        constraint_indices: List[int],
        feature_indices: List[int],
        warm_start: Optional[ndarray],
    ) -> LpProblem:
        start = time()
        N, P = X_tilde.shape
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
                f"xi_{constraint_id}_{pairs[constraint_id].i}_{pairs[constraint_id].j}",
                lowBound=0,
            )
            for constraint_id in constraint_indices
        ]

        # Beta
        beta_plus = [
            LpVariable(f"beta_plus_{feature_id}", lowBound=0)
            for feature_id in feature_indices
        ]
        beta_minus = [
            LpVariable(f"beta_minus_{feature_id}", lowBound=0)
            for feature_id in feature_indices
        ]

        problem += (
            self.C * lpDot(sample_weights, xi) + lpSum(beta_plus) + lpSum(beta_minus)
        )

        # The selected constraints using the selected features
        X_tilde_reduced = X_tilde[:, feature_indices]
        for i, constraint_id in enumerate(constraint_indices):
            pair = pairs[constraint_id]
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
        X_tilde: ndarray,
        pairs: List[Pair],
        problem: LpProblem,
        violated_features: ndarray,
        current_constraints: List[int],
    ) -> LpProblem:
        # add new features
        added_beta_plus = []
        added_beta_minus = []

        for feature_id in violated_features:
            added_beta_plus.append(LpVariable(f"beta_plus_{feature_id}", lowBound=0))
            added_beta_minus.append(LpVariable(f"beta_minus_{feature_id}", lowBound=0))

        X_tilde_reduced = X_tilde[:, violated_features]
        problem.addVariables(added_beta_plus)
        problem.addVariables(added_beta_minus)
        for constraint_id in current_constraints:
            old_constraint: LpConstraint = problem.constraints[
                f"constraint_{constraint_id}"
            ]
            updated_constraint = (
                old_constraint
                + lpDot(X_tilde_reduced[constraint_id], added_beta_plus)
                - lpDot(X_tilde_reduced[constraint_id], added_beta_minus)
            )
            problem.constraints[f"constraint_{constraint_id}"] = updated_constraint
            problem.modifiedConstraints.append(updated_constraint)

        problem.setObjective(
            problem.objective + lpSum(added_beta_plus) + lpSum(added_beta_minus)
        )
        LOGGER.info("Features added and objective updated")

        return problem

    def _add_constraints_to_subproblem(
        self,
        X_tilde: ndarray,
        pairs: List[Pair],
        problem: LpProblem,
        violated_constraints: List[int],
        current_features: List[int],
        beta_plus: List[LpVariable],
        beta_minus: List[LpVariable],
    ):
        xis_to_add = []
        sample_weights_to_add = []
        X_reduced = X_tilde[:, current_features]

        for violated_constraint in violated_constraints:
            pair = pairs[violated_constraint]
            xi_violated = LpVariable(
                f"xi_{violated_constraint}_{pair.i}_{pair.j}", lowBound=0
            )
            problem.addConstraint(
                (
                    lpDot(X_reduced[violated_constraint], beta_plus)
                    - lpDot(X_reduced[violated_constraint], beta_minus)
                    >= pair.gap * pair.sample_weight - pair.sample_weight * xi_violated
                ),
                name=f"constraint_{violated_constraint}",
            )
            xis_to_add.append(xi_violated)
            sample_weights_to_add.append(pair.sample_weight)

        problem.setObjective(
            problem.objective + lpDot(xis_to_add, sample_weights_to_add) * self.C
        )
        LOGGER.info("Constraints added and objective updated")

        return problem


def _init_constraint_column_sampling_smoothing(
    X_tilde: ndarray, pairs: List[Pair], C: float
) -> Tuple[List[int], List[int]]:
    pairs_np = np.array(pairs)
    N, P = X_tilde.shape
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
        X_tilde_reduced = X_tilde[subset]
        pairs_reduced = pairs_np[subset]

        # Correlation screening
        argsort_columns = np.argsort(np.abs(np.sum(X_tilde_reduced, axis=0)))
        index_columns = argsort_columns[::-1][:P0]

        X_tilde_reduced = X_tilde_reduced[:, index_columns]

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
    constraints = 1 * gs - np.dot(X_tilde, beta_averaged)
    index_samples_smoothing = np.arange(N)[constraints >= 0]
    LOGGER.info(f"Len dual smoothing: {len(index_samples_smoothing)}")

    return list(index_samples_smoothing), list(index_columns_smoothing)

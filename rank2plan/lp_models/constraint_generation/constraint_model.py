from rank2plan import Model, Pair, LossType, PenalisationType
from rank2plan.lp_models.warm_starting.smoothing_hinge_loss import (
    loop_smoothing_hinge_loss_samples_restricted,
)
from pulp import LpSolver, LpProblem, LpMinimize, LpVariable, lpSum, lpDot
import numpy as np
from numpy import ndarray
from time import time
import random
from typing import List, Optional


class ConstraintModel(Model):
    """Constraint generation model when the number of pairs (constraints) is
    significantly higher than the number of features. Based on Section 2.3.2 of
    (Dedieu et al, 2022).
    """

    def __init__(self, solver: LpSolver, C=1.0, verbose=False, tol=1e-4) -> None:
        self.solver = solver
        self.C = C
        self.verbose = verbose
        self.tol = tol
        self._weights = None

    def fit(self, X: ndarray, pairs: List[Pair]) -> None:
        def custom_print(str):
            if self.verbose:
                print(f"[{self.fit.__name__}] {str}")

        start = time()
        X_tilde = compute_X_tilde(X, pairs)
        custom_print(f"X_tilde computed in {time() - start:.3f} seconds")

        start = time()
        constraint_indices = _init_sampling_smoothing(
            X_tilde, pairs, self.C, verbose=self.verbose
        )
        custom_print(f"Initial sampling done in {time() - start:.3f} seconds")

        problem = self._build_subproblem(
            X_tilde, pairs, constraint_indices, None, verbose=self.verbose
        )

        N, P = X_tilde.shape
        constraints_to_check = list(set(range(N)) - set(constraint_indices))
        gs = np.array([pair.gap * pair.sample_weight for pair in pairs])

        # infinite loop until we are done
        cg_start = time()
        continue_loop = True
        cur_iter = 0
        while continue_loop:
            continue_loop = False
            cur_iter += 1

            start = time()
            problem.solve(self.solver)
            custom_print(f"Subproblem solved in {time() - start:.3f} seconds")

            if len(constraint_indices) != N:
                # get parameters
                beta_plus = [
                    problem.variablesDict()[f"beta_plus_{p}"] for p in range(P)
                ]
                beta_minus = [
                    problem.variablesDict()[f"beta_minus_{p}"] for p in range(P)
                ]
                beta = np.array(
                    [beta_plus[p].varValue - beta_minus[p].varValue for p in range(P)]
                )

                # find constraints with negative reduced cost
                reduced_costs = gs[np.array(constraints_to_check)] - np.dot(
                    X_tilde[constraints_to_check], beta
                )
                violated_constraints = np.array(constraints_to_check)[
                    reduced_costs > self.tol
                ]

                # add violated constraints to the subproblem
                if len(violated_constraints) > 0:
                    custom_print(f"Adding {len(violated_constraints)} constraints")
                    custom_print(f"Most violated constraint: {np.max(reduced_costs)}")
                    problem = self._add_constraints_to_subproblem(
                        X_tilde,
                        pairs,
                        problem,
                        violated_constraints,
                        beta_plus,
                        beta_minus,
                        verbose=self.verbose,
                    )

                    continue_loop = True
                    constraint_indices.extend(violated_constraints)
                    constraints_to_check = list(
                        set(constraints_to_check) - set(violated_constraints)
                    )

        custom_print(
            f"Finished constraint generation in {cur_iter} iterations, using {time() - cg_start:.3f} seconds"
        )

        beta_plus = np.array(
            [problem.variablesDict()[f"beta_plus_{p}"].varValue for p in range(P)]
        )
        beta_minus = np.array(
            [problem.variablesDict()[f"beta_minus_{p}"].varValue for p in range(P)]
        )
        self._weights = beta_plus - beta_minus
        custom_print(f"Overall objective: {problem.objective.value()}")  # type: ignore

    def predict(self, X: ndarray) -> ndarray:
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights

    def _build_subproblem(
        self,
        X_tilde: ndarray,
        pairs: List[Pair],
        constraint_indices: List[int],
        warm_start: Optional[ndarray],
        verbose=False,
    ) -> LpProblem:
        def custom_print(str):
            if verbose:
                print(f"[{self._build_subproblem.__name__}] {str}")

        start = time()
        N, P = X_tilde.shape
        N_constraints = len(constraint_indices)
        sample_weights = [
            pairs[constraint_id].sample_weight for constraint_id in constraint_indices
        ]
        custom_print(f"Subproblem with {N_constraints} constraints and {P} features")

        problem = LpProblem("ConstraintGenerationSubproblem", LpMinimize)

        # Hinge loss
        xi = [
            LpVariable(
                f"xi_{constraint_id}_{pairs[constraint_id].i}_{pairs[constraint_id].j}",
                lowBound=0,
            )
            for constraint_id in constraint_indices
        ]

        # Beta
        beta_plus = [LpVariable(f"beta_plus_{p}", lowBound=0) for p in range(P)]
        beta_minus = [LpVariable(f"beta_minus_{p}", lowBound=0) for p in range(P)]

        problem += (
            self.C * lpDot(sample_weights, xi) + lpSum(beta_plus) + lpSum(beta_minus)
        )

        # Only the selected constraints
        for i, constraint_id in enumerate(constraint_indices):
            pair = pairs[constraint_id]
            # we've multiplied both sides by the sample weight
            problem += (
                lpDot(X_tilde[constraint_id], beta_plus)
                - lpDot(X_tilde[constraint_id], beta_minus)
                >= pair.gap * pair.sample_weight - pair.sample_weight * xi[i]
            )

        if warm_start is not None:
            for i in range(P):
                beta_plus[i].varValue = max(0, warm_start[i])
                beta_minus[i].varValue = max(0, -warm_start[i])

        custom_print(f"Subproblem built in {time() - start:.3f} seconds")
        return problem

    def _add_constraints_to_subproblem(
        self,
        X_tilde: ndarray,
        pairs: List[Pair],
        problem: LpProblem,
        violated_constraints: List[int],
        beta_plus: List[LpVariable],
        beta_minus: List[LpVariable],
        verbose=False,
    ):
        xis_to_add = []
        sample_weights_to_add = []

        for violated_constraint in violated_constraints:
            pair = pairs[violated_constraint]
            xi_violated = LpVariable(
                f"xi_{violated_constraint}_{pair.i}_{pair.j}", lowBound=0
            )
            problem += (
                lpDot(X_tilde[violated_constraint], beta_plus)
                - lpDot(X_tilde[violated_constraint], beta_minus)
                >= pair.gap * pair.sample_weight - pair.sample_weight * xi_violated
            )
            xis_to_add.append(xi_violated)
            sample_weights_to_add.append(pair.sample_weight)

        problem.setObjective(
            problem.objective + lpDot(xis_to_add, sample_weights_to_add)
        )

        return problem


def compute_X_tilde(X: ndarray, pairs: List[Pair]) -> ndarray:
    """Compute the X_tilde matrix.

    Args:
        X (ndarray): The original feature matrix, shape (n_samples, n_features).
        pairs (List[Pair]): The list of pairs, shape (n_pairs,).

    Returns:
        ndarray: The X_tilde matrix, shape (n_pairs, n_features).
    """
    X_tilde = []
    for pair in pairs:
        X_tilde.append(pair.sample_weight * (X[pair.j] - X[pair.i]))
    return np.array(X_tilde)


def _init_sampling_smoothing(
    X_tilde: ndarray, pairs: List[Pair], C: float, is_restricted=True, verbose=False
) -> List[int]:
    """Initial sampling for constraint generation using smoothing.

    Args:
        X_tilde (ndarray): The X_tilde matrix, shape (n_pairs, n_features).
        pairs (List[Pair]): The list of pairs, shape (n_pairs,).
        C (float): The regularisation parameter.
        is_restricted (bool, optional): Not sure yet :(. Defaults to True.
        verbose (bool, optional): Whether to print messages. Defaults to False.

    Returns:
        List[int]: the initial set of constraints
    """

    def custom_print(str):
        if verbose:
            print(f"[{_init_sampling_smoothing.__name__}] {str}")

    pairs_np = np.array(pairs)
    N, P = X_tilde.shape
    rho = (1 / C) / np.max(
        np.sum(np.abs(X_tilde), axis=0)
    )  # lambda (1/C) relative to norms

    N0 = int(min(10 * P, N / 4))
    if N0 < 3:
        custom_print("Not enough samples to perform initial sampling, returning none")
        return []

    start_time = time()
    custom_print("Finding initial solution using first-order method")

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
        custom_print(f"Sample number: {k}")
        custom_print(f"Difference variance: {delta_variance}")

        subset = np.sort(random.sample(range(N), N0))
        X_tilde_reduced = X_tilde[subset]
        pairs_reduced = pairs_np[subset]

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
                verbose=verbose,
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
    ones_N = np.ones(N)

    g = np.array([pair.gap for pair in pairs])
    s = np.array([pair.sample_weight for pair in pairs])
    gs = np.array([pair.gap * pair.sample_weight for pair in pairs])

    constraints = 1 * gs - (np.dot(X_tilde, beta_averaged))
    idx_samples_smoothing = np.arange(N)[constraints >= 0]

    custom_print("Finished initial sampling")
    custom_print(f"Len dual smoothing: {len(idx_samples_smoothing)}")

    time_smoothing = time() - start_time
    custom_print(f"Total time: {time_smoothing:.3f}")

    return list(idx_samples_smoothing)

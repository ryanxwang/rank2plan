# from rank2plan import Model, Pair, LossType, PenalisationType
# from rank2plan.lp_models.smoothing_hinge_loss import (
#     loop_smoothing_hinge_loss_samples_restricted,
# )
# from pulp import LpSolver
# import numpy as np
# from numpy import ndarray
# from time import time
# import random
# from typing import List


# class ConstraintModel(Model):
#     """Constraint generation model when the number of pairs (constraints) is
#     significantly higher than the number of features. Based on Section 2.3.2 of
#     (Dedieu et al, 2022).
#     """

#     def __init__(self, solver: LpSolver, C=1.0, verbose=False, tol=1e-2) -> None:
#         self.solver = solver
#         self.C = C
#         self.verbose = verbose
#         self.tol = tol
#         self._weights = None


# def _data_l1_norm(X: ndarray, pairs: List[Pair]) -> ndarray:
#     """Compute the L1 norm of all the implied feature vectors (one for each
#     pair.

#     Args:
#         X (ndarray): Training data, shape (n_samples, n_features)
#         pairs (List[Pair]): Training pairs, shape (n_pairs,)

#     Returns:
#         ndarray: L1 norms, shape (n_pairs,)
#     """
#     norms = []
#     for pair in pairs:
#         norms.append(np.sum(np.abs(X[pair.j] - X[pair.i])))
#     return np.array(norms)


# def _init_sampling_smoothing(
#     X: ndarray, pairs: List[Pair], C: float, is_restricted=True, verbose=False
# ):
#     N = len(pairs)  # number of constraints is given by the pairs, not X
#     P = X.shape[1]
#     pairs_np = np.array(pairs)
#     norms = _data_l1_norm(X, pairs)
#     rho = (1 / C) / np.max(norms)  # lambda (1/C) relative to norms

#     N0 = int(min(10 * P, N / 4))

#     if verbose:
#         start_time = time()
#         print("Finding initial solution using first-order method")

#     tau_max = 0.1
#     n_loop = 20
#     n_iter = 20

#     # result
#     old_beta_averaged = -np.ones(P + 1)
#     beta_averaged = np.zeros(P + 1)
#     delta_variance = 1e6

#     k = 0
#     while delta_variance > 5e-2 and k < int(N / N0):
#         k += 1
#         if verbose:
#             print(f"Sample number: {k}")
#             print(f"Difference variance: {delta_variance}")

#         subset = np.sort(random.sample(range(N), N0))
#         pairs_reduced = pairs_np[subset]

#         alpha_sample = rho * np.max(norms[subset])

#         if is_restricted:
#             n_loop = 10
#             _, beta_sample = loop_smoothing_hinge_loss_samples_restricted(
#                 LossType.Hinge,
#                 PenalisationType.L1,
#                 X,
#                 pairs_reduced,
#                 alpha_sample,
#                 tau_max,
#                 n_loop,
#                 n_iter,
#                 verbose=verbose,
#             )
#         else:
#             _, _, _, beta_sample, beta0_sample = loop_smoothing_hinge_loss(
#                 LossType.Hinge,
#                 PenalisationType.L1,
#                 X,
#                 pairs_reduced,
#                 alpha_sample,
#                 tau_max,
#                 n_loop,
#                 n_iter,
#             )
#             beta_sample = np.concatenate([beta_sample, np.array([beta0_sample])])

#         old_beta_averaged = np.copy(beta_averaged)
#         beta_averaged += np.array(beta_sample)
#         delta_variance = np.linalg.norm(
#             1.0 / max(1, k) * beta_averaged - 1.0 / max(1, k - 1) * old_beta_averaged
#         )

#     # ---Determine set of constraints
#     beta_averaged *= N0 / float(N)
#     b0_averaged = beta_averaged[-1]
#     beta_averaged = beta_averaged[:-1]
#     b0_averaged = 0
#     ones_N = np.ones(N)

#     constraints = 1 * ones_N - y_train * (
#         np.dot(X_train, beta_averaged) + b0_averaged * ones_N
#     )
#     idx_samples_smoothing = np.arange(N)[constraints >= 0]
#     write_and_print("\n\n\nFINISHED", f)
#     write_and_print("Len dual smoothing: " + str(idx_samples_smoothing.shape[0]), f)

#     time_smoothing = time.time() - start_time
#     write_and_print("Total time: " + str(round(time_smoothing, 3)), f)
#     return list(idx_samples_smoothing), time_smoothing

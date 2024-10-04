import numpy as np
from numpy import ndarray
import time
import math
from rank2plan import LossType, PenalisationType, Matrix
import logging
from scipy.sparse import issparse

LOGGER = logging.getLogger(__name__)


def smoothing_hinge_loss(
    loss_type: LossType,
    penalisation_type: PenalisationType,
    X_tilde: Matrix,
    pairs: ndarray,
    alpha: float,
    beta_start: ndarray,
    highest_eig: float,
    tau: float,
    n_iter: int,
):
    # TYPE_PENALIZATION = 1 : L1 -> soft thresholding
    # TYPE_PENALIZATION = 2 : L2
    is_sparse = issparse(X_tilde)

    # ---Initialization
    start_time = time.time()
    assert X_tilde.shape is not None
    N, P = X_tilde.shape

    old_beta = np.ones(P)
    beta_m = beta_start

    # ---MAIN LOOP
    test = 0
    t_AGD_old = 1
    t_AGD = 1
    eta_m_old = beta_start
    g = np.array([pair.gap for pair in pairs])
    s = np.array([pair.sample_weight for pair in pairs])
    gs = np.array([g[i] * s[i] for i in range(N)])

    if loss_type == LossType.Hinge:
        Lipchtiz_coeff = highest_eig / (4 * tau)
    elif loss_type == LossType.SquaredHinge:
        raise ValueError("Squared hinge not implemented")
        # the following line MAY be correct
        # Lipchtiz_coeff = 2 * highest_eig

    while np.linalg.norm(beta_m - old_beta) > 1e-3 and test < n_iter:
        test += 1
        aux = gs - (np.dot(X_tilde, beta_m) if not is_sparse else X_tilde.dot(beta_m))  # type: ignore

        # ---Hinge loss
        if loss_type == LossType.Hinge:
            w_tau = [
                min(1, abs(aux[i]) / (2 * tau)) * np.sign(aux[i]) for i in range(N)
            ]
            # gradient_loss   = -0.5*np.sum([y[i]*(1+w_tau[i])*X_add[i,:] for i in range(N)], axis=0)
            gradient_aux = np.array([1 + w_tau[i] for i in range(N)])

            gradient_loss = (
                -0.5 * np.dot(X_tilde.T, gradient_aux)
                if not is_sparse
                else -0.5 * X_tilde.T.dot(gradient_aux)
            )

        # ---Gradient descent
        old_beta = beta_m
        grad = beta_m - 1 / float(Lipchtiz_coeff) * gradient_loss

        # ---Thresholding of top 100 guys !
        dict_thresholding = {
            PenalisationType.L1: soft_thresholding_l1,
            PenalisationType.L2: soft_thresholding_l2,
        }
        eta_m = np.array(
            [
                dict_thresholding[penalisation_type](grad[i], alpha / Lipchtiz_coeff)
                for i in range(P)
            ]
        )

        # ---AGD
        t_AGD = (1 + math.sqrt(1 + 4 * t_AGD_old**2)) / 2.0
        aux_t_AGD = (t_AGD_old - 1) / t_AGD

        beta_m = eta_m + aux_t_AGD * (eta_m - eta_m_old)

        t_AGD_old = t_AGD
        eta_m_old = eta_m

    LOGGER.info(f"Number of iterations: {test}")
    LOGGER.info(f"X_tilde shape: {X_tilde.shape}")

    # ---Support
    idx_columns_smoothing = np.where(beta_m[:P] != 0)[0]
    LOGGER.info(f"Len support smoothing: {idx_columns_smoothing.shape[0]}")

    # ---Constraints
    ##### USE B0 !!!!!!!
    if not is_sparse:
        constraints = 1.05 * gs - (
            np.dot(X_tilde[:, idx_columns_smoothing], beta_m[idx_columns_smoothing])  # type: ignore
        )
    else:
        constraints = 1.05 * gs - (
            X_tilde[:, idx_columns_smoothing].dot(beta_m[idx_columns_smoothing])  # type: ignore
        )

    idx_samples_smoothing = np.arange(N)[constraints >= 0]

    LOGGER.info(f"Number violated constraints: {idx_samples_smoothing.shape[0]}")
    LOGGER.info(f"Convergence rate: {np.linalg.norm(beta_m - old_beta):.3f}")

    time_smoothing = time.time() - start_time
    LOGGER.info(f"Smoothing time: {time_smoothing:.3f}")

    return (
        idx_samples_smoothing.tolist(),
        idx_columns_smoothing.tolist(),
        time_smoothing,
        beta_m,
    )


def loop_smoothing_hinge_loss_samples_restricted(
    loss_type: LossType,
    penalisation_type: PenalisationType,
    pairs: ndarray,
    X_tilde: Matrix,
    alpha: float,
    tau_max: float,
    n_loop: int,
    n_iter: int,
):
    # n_loop: how many times should we run the loop ?
    # Apply the smoothing technique from the best subset selection

    start_time = time.time()

    assert X_tilde.shape is not None
    N, P = X_tilde.shape
    old_beta = -np.ones(P)

    # ---New matrix and SVD
    highest_eig = power_method(X_tilde)

    # ---Results
    beta_smoothing = np.zeros(P)
    tau = tau_max

    # ---Prepare for restrcition
    idx_samples = np.arange(N)
    X_tilde_reduced: Matrix = X_tilde
    pairs_reduced = pairs

    test = -1
    while np.linalg.norm(beta_smoothing - old_beta) > 1e-4 and test < n_loop:
        LOGGER.info(
            f"Test CV before tau: {np.linalg.norm(beta_smoothing - old_beta):.4f}"
        )

        test += 1
        old_beta = beta_smoothing

        idx_samples_restricted, _, _, beta_smoothing = smoothing_hinge_loss(
            loss_type,
            penalisation_type,
            X_tilde_reduced,
            pairs_reduced,
            alpha,
            beta_smoothing,
            highest_eig,
            tau,
            n_iter,
        )

        if test == 0:
            # ---Restrict to samples
            X_tilde_reduced = X_tilde[idx_samples_restricted, :]  # type: ignore
            pairs_reduced = pairs[idx_samples_restricted]

            highest_eig = power_method(X_tilde_reduced)
            idx_samples = idx_samples[idx_samples_restricted]

        # ---Update parameters
        tau = 0.7 * tau

    time_smoothing_tot = time.time() - start_time
    LOGGER.info(f"Num iters: {test}")
    LOGGER.info(f"Total smoothing time: {time_smoothing_tot:.3f}")

    return idx_samples.tolist(), beta_smoothing


def loop_smoothing_hinge_loss_columns_samples_restricted(
    loss_type: LossType,
    penalisation_type: PenalisationType,
    X_tilde: Matrix,
    pairs: ndarray,
    alpha,
    tau_max,
    n_loop,
    n_iter,
):

    start_time = time.time()
    assert X_tilde.shape is not None
    N, P = X_tilde.shape
    old_beta = -np.ones(P)

    highest_eig = power_method(X_tilde)

    # ---Results
    beta_smoothing = np.zeros(P)

    tau = tau_max

    # ---Prepare for restriction
    X_tilde_reduced: Matrix = X_tilde
    pairs_reduced = pairs
    idx_columns_restricted = np.arange(P)

    test = -1
    while np.linalg.norm(beta_smoothing - old_beta) > 1e-2 and test < n_loop:
        LOGGER.info(
            f"Test CV between 2 values of tau: {np.linalg.norm(beta_smoothing - old_beta):.4f}"
        )
        if test == 0:
            old_beta = beta_smoothing[idx_columns_restricted]
        else:
            old_beta = beta_smoothing

        test += 1
        (
            _,
            idx_columns_restricted,
            _,
            beta_smoothing,
        ) = smoothing_hinge_loss(
            loss_type,
            penalisation_type,
            X_tilde_reduced,
            pairs_reduced,
            alpha,
            old_beta,
            highest_eig,
            tau,
            n_iter,
        )

        if test == 0:
            # ---Dont change samples -> just restrict columns
            X_tilde_reduced = X_tilde_reduced[:, idx_columns_restricted]  # type: ignore
            assert X_tilde_reduced.shape is not None
            P_reduced = X_tilde_reduced.shape[1]

            highest_eig = power_method(X_tilde_reduced)
            idx_columns = idx_columns_restricted

        # ---Update parameters
        tau = 0.7 * tau
        n_iter = 50

    # ---Results
    beta_smoothing_sample = np.zeros(P)
    for i in range(len(idx_columns)):
        beta_smoothing_sample[idx_columns[i]] = beta_smoothing[i]

    time_smoothing_tot = time.time() - start_time
    LOGGER.info(f"Number of iterations: {test}")
    LOGGER.info("Total time smoothing: " + str(round(time_smoothing_tot, 3)))

    return beta_smoothing_sample


def power_method(X: Matrix) -> float:
    """Compute the highest eigenvalue of X^T X.

    Args:
        X (Matrix): The matrix to compute the highest eigenvalue of.

    Returns:
        float: The highest eigenvalue of X^T X.
    """
    is_sparse = issparse(X)
    assert X.shape is not None
    P = X.shape[1]

    highest_eigvctr = np.random.rand(P)
    old_highest_eigvctr = -1

    while np.linalg.norm(highest_eigvctr - old_highest_eigvctr) > 1e-2:
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr = (
            np.dot(X.T, np.dot(X, highest_eigvctr))  # type: ignore
            if not is_sparse
            else X.T.dot(X.dot(highest_eigvctr))
        )
        highest_eigvctr /= np.linalg.norm(highest_eigvctr)

    X_highest_eig = (
        np.dot(X, highest_eigvctr) if not is_sparse else X.dot(highest_eigvctr)  # type: ignore
    )

    highest_eig = np.dot(X_highest_eig.T, X_highest_eig) / np.linalg.norm(
        highest_eigvctr
    )
    return highest_eig


def soft_thresholding_l1(c: float, alpha: float) -> float:
    """The thresholding operator for L1 penalisation. See Section 4.2 of Dedieu
    et al. (2022).
    """
    if alpha >= abs(c):
        return 0
    else:
        if c >= 0:
            return c - alpha
        else:
            return c + alpha


def soft_thresholding_l2(c: float, alpha: float) -> float:
    """The thresholding operator for L2 penalisation. See Section 4.2 of Dedieu
    et al. (2022).
    """
    return c / float(1 + 2 * alpha)

import typing

from hermite_functions import hermite_functions
import scipy.special
import scipy.stats
import numpy as np


pl_params = {
    "irr_sublevel": [(0, 150)],
    "airflow_sublevel": [(-0.002, 0.002)],
    "airflow_rips": [(0, 0.002), (0, 0.003)],
}

n_hom_deg = {
    "irr_sublevel": 1,
    "airflow_sublevel": 1,
    "airflow_rips": 2,
}


def midlife_persistence(dgm: np.ndarray) -> np.ndarray:
    return (dgm[:, 1] + dgm[:, 0]) / 2


def lifespan_persistence(dgm: np.ndarray) -> np.ndarray:
    return dgm[:, 1] - dgm[:, 0]


def mean_pers(pers_arr: np.ndarray) -> float:
    return np.mean(pers_arr, axis=-1)


def std_pers(pers_arr: np.ndarray) -> float:
    return np.std(pers_arr, axis=-1)


def skew_pers(pers_arr: np.ndarray) -> float:
    return scipy.stats.skew(pers_arr, axis=-1)


def kurt_pers(pers_arr: np.ndarray) -> float:
    """Calculates kurtosis of persistence"""
    return scipy.stats.kurtosis(pers_arr, axis=-1)


def entr_pers_lifespan(dgms: np.ndarray) -> float:
    """Calculates entropy of persistence"""
    pers_arr = dgms[:, 1] - dgms[:, 0]
    #  L = np.sum(pers_arr)
    #  return -np.sum((pers_arr / L) * np.log(pers_arr / L))
    return scipy.stats.entropy(pers_arr)


def entr_pers_midlife(dgms: np.ndarray) -> float:
    """Calculates entropy of persistence"""
    #  pers_arr_p = dgms[:, 1] + dgms[:, 0]
    #  pers_arr_n = dgms[:, 1] - dgms[:, 0]

    # Correcting for small errors in calculation
    #  pers_arr_p[pers_arr_p < 0] = 0.0

    #  M = np.sum(pers_arr_p)
    #  return -np.sum((pers_arr_p / M) * np.log(pers_arr_n / M))
    M = (dgms[:, 1] - dgms[:, 0]) / 2
    return scipy.stats.entropy(np.abs(M))


def gaussian_persistence_curve(dgms: np.ndarray, sigma: float) -> float:
    """Calculates the 1-norm of the Gaussian persistence curve"""
    pers_arr_n = dgms[:, 1] - dgms[:, 0]
    rv_gen = scipy.stats.norm()

    val = pers_arr_n / (np.sqrt(2) * sigma)

    gp_arr = pers_arr_n * rv_gen.cdf(val) + (np.sqrt(2) * sigma) * rv_gen.pdf(val)
    norm_1_val = np.sum(gp_arr)
    return norm_1_val


def psi(dgms: np.ndarray) -> np.ndarray:
    pers_arr_n = dgms[:, 1] - dgms[:, 0]
    L = np.sum(pers_arr_n)
    val = pers_arr_n / L
    return -val * np.log(val)


def hepc(dgms: np.ndarray, n_coef=15) -> np.ndarray:
    """Calculates the Hermite function expansion of persistence curve (HEPC)"""
    b = dgms[:, 0]
    d = dgms[:, 1]
    psi_dgms = psi(dgms)

    rv_gen = scipy.stats.norm()
    rv_val_cdf_db = rv_gen.cdf(d) - rv_gen.cdf(b)
    rv_val_pdf_bd = rv_gen.pdf(b) - rv_gen.pdf(d)

    coeff = np.sqrt(2) * np.power(np.pi, 0.25)
    alphas = []
    alphas.append(np.sum(coeff * psi_dgms * rv_val_cdf_db))
    alphas.append(np.sum(2 * np.power(np.pi, 0.25) * psi_dgms * rv_val_pdf_bd))

    for n in range(2, n_coef):
        # Indexing starts from 1 before current n
        n_index = n - 1

        coeff_i = np.sqrt(2) / np.sqrt(n_index + 1)
        const_i = (n_index * alphas[n_index - 1]) / np.sqrt(n_index * (n_index + 1))

        hfunc_b = hermite_functions(n, b, all_n=False)
        hfunc_d = hermite_functions(n, d, all_n=False)
        hdiff = hfunc_b - hfunc_d
        val_i = psi_dgms * hdiff

        alphas.append((coeff_i * np.sum(val_i)) + const_i)

    alphas = np.asarray(alphas)
    return alphas


def fft_pc(dgms: np.ndarray) -> np.ndarray:
    psi_dgm = psi(dgms)
    x = np.linspace(dgms.min(), dgms.max(), 1000)
    y = np.zeros(x.shape)

    for idx, (b, d) in enumerate(dgms):
        arr_idx = (x >= b) & (x <= d)
        y[arr_idx] += psi_dgm[idx]
    alpha = np.fft.rfft(y)
    return alpha


def fft_closed_form(
    dgms: np.ndarray, L: typing.Union[float, None] = None
) -> np.ndarray:
    max_coef = 502
    if not L:
        L = dgms.max() - dgms.min()
    psi_dgms = psi(dgms)

    b = dgms[:, 0]
    d = dgms[:, 1]
    beta = []

    # Calculating 0 coefficient
    beta.append(2 * np.sum(psi_dgms * (d - b)) / L)

    for n in range(1, max_coef):
        bval = 2 * 1.0j * np.pi * b * n / L
        dval = 2 * 1.0j * np.pi * d * n / L
        diff = np.exp(bval) - np.exp(dval)
        val = diff * 1.0j * L / (2 * np.pi * n)
        beta.append(2 * np.sum(psi_dgms * val) / L)
    return np.asarray(beta)


def h(t):
    if t >= 0 and t <= 0.5:
        return 1
    elif t > 0.5 and t <= 1:
        return -1
    return 0


def h_n(t, n):
    npow = np.power(2, n)
    npow2 = np.power(2, n / 2)
    val = npow2 * h(npow * t)
    return val

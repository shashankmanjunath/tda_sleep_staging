import scipy.special
import scipy.stats
import numpy as np


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


def psi(dgms: np.ndarray) -> float:
    pers_arr_n = dgms[:, 1] - dgms[:, 0]
    L = np.sum(pers_arr_n)
    val = pers_arr_n / L
    return -val * np.log(val)


def hepc(dgms: np.ndarray) -> np.ndarray:
    """Calculates the Hermite function expancsion of persistence curve (HEPC)"""
    b = dgms[:, 0]
    d = dgms[:, 1]
    psi_dgms = psi(dgms)

    rv_gen = scipy.stats.norm()
    rv_val_cdf_db = rv_gen.cdf(d) - rv_gen.cdf(b)
    rv_val_pdf_bd = rv_gen.pdf(b) - rv_gen.pdf(d)

    coeff = np.sqrt(2) * np.power(np.pi, 0.25)
    alpha = []
    alpha.append(np.sum(coeff * psi_dgms * rv_val_cdf_db))
    alpha.append(np.sum(coeff * psi_dgms * rv_val_pdf_bd))

    for n in range(2, 15):
        # Indexing starts from 1 before current n
        n_index = n - 1

        coeff_i = np.sqrt(2) / np.sqrt(n_index + 1)
        const_i = (n_index * alpha[n_index - 1]) / np.sqrt(n_index * (n_index + 1))

        hfunc = scipy.special.hermite(n_index)
        #  hdiff = hfunc(b) - hfunc(d)

        # NOTE: This multiplication by the Gaussian PDF is found in the code,
        # but not in the paper. I have included it since it fixes some scaling
        # issues.
        hdiff = rv_gen.pdf(b) * hfunc(b) - rv_gen.pdf(d) * hfunc(d)
        val_i = psi_dgms * hdiff

        alpha.append((coeff_i * np.sum(val_i)) + const_i)
    alpha = np.asarray(alpha)
    return alpha

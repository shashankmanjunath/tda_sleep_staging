from collections import defaultdict
import typing
import math

from teaspoon.ML import feature_functions
from teaspoon.ML import Base

import gudhi.representations
import scipy.special
import pandas as pd
import gtda.diagrams
import scipy.stats
import numpy as np
import persim

import time


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


def apply_template_function(
    data_dict: typing.Dict,
    train_fnames: typing.List,
    test_fnames: typing.List,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    train_data = []
    for train_fname in train_fnames:
        train_data += data_dict[train_fname]

    test_data = []
    for test_fname in test_fnames:
        test_data += data_dict[test_fname]

    train_output_data = []
    test_output_data = []

    for k, total_hom_deg in n_hom_deg.items():
        for h in range(total_hom_deg):
            t1 = time.time()
            train_data_arr = [x[k][h] for x in train_data]
            test_data_arr = [x[k][h] for x in test_data]

            # Removing inf values
            train_clean_data = [x[~np.isinf(x).any(axis=1)] for x in train_data_arr]
            test_clean_data = [x[~np.isinf(x).any(axis=1)] for x in test_data_arr]

            df_key = f"h_{h}"
            train_df = pd.DataFrame({df_key: train_clean_data})
            test_df = pd.DataFrame({df_key: test_clean_data})

            # Creating object to transform data
            params = Base.ParameterBucket()
            params.feature_function = feature_functions.interp_polynomial
            params.d = 20
            params.jacobi_poly = "cheb1"

            params.makeAdaptivePartition(train_df, meshingScheme=None)

            train_feature = Base.build_G(train_df[df_key], params)
            test_feature = Base.build_G(test_df[df_key], params)

            train_output_data.append(train_feature)
            test_output_data.append(test_feature)
            t2 = time.time()
            print(f"{k}, h_{h} Time: {t2-t1}")
    train_output_data = np.concatenate(train_output_data, axis=-1)
    test_output_data = np.concatenate(test_output_data, axis=-1)
    return train_output_data, test_output_data


def convert_pd_pl_train_test(
    data_dict: typing.Dict,
    train_fnames: typing.List,
    test_fnames: typing.List,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    train_data = []
    for train_fname in train_fnames:
        train_data += data_dict[train_fname]

    test_data = []
    for test_fname in test_fnames:
        test_data += data_dict[test_fname]

    landscapers = defaultdict(list)
    train_output_data = []
    test_output_data = []

    t1 = time.time()
    for k, total_hom_deg in n_hom_deg.items():
        for h in range(total_hom_deg):
            train_data_arr = [x[k][h] for x in train_data]
            test_data_arr = [x[k][h] for x in test_data]

            # Removing inf values
            train_clean_data = [x[~np.isinf(x).any(axis=1)] for x in train_data_arr]
            test_clean_data = [x[~np.isinf(x).any(axis=1)] for x in test_data_arr]

            landscapers[k].append(gudhi.representations.vector_methods.Landscape())
            landscapers[k][h].fit(train_clean_data)

            train_output_data.append(landscapers[k][h].transform(train_clean_data))
            test_output_data.append(landscapers[k][h].transform(test_clean_data))
    t2 = time.time()
    #  print(f"Time: {t2-t1}")
    train_output_data = np.concatenate(train_output_data, axis=-1)
    test_output_data = np.concatenate(test_output_data, axis=-1)
    return train_output_data, test_output_data


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


def hepc(dgms: np.ndarray) -> np.ndarray:
    """Calculates the Hermite function expansion of persistence curve (HEPC)"""
    b = dgms[:, 0]
    d = dgms[:, 1]
    psi_dgms = psi(dgms)

    rv_gen = scipy.stats.norm()
    rv_val_cdf_db = rv_gen.cdf(d) - rv_gen.cdf(b)
    rv_val_pdf_bd = rv_gen.pdf(b) - rv_gen.pdf(d)

    coeff = np.sqrt(2) * np.power(np.pi, 0.25)
    alpha = []
    alpha.append(np.sum(coeff * psi_dgms * rv_val_cdf_db))
    alpha.append(np.sum(2 * np.power(np.pi, 0.25) * psi_dgms * rv_val_pdf_bd))

    for n in range(2, 15):
        # Indexing starts from 1 before current n
        n_index = n - 1

        coeff_i = np.sqrt(2) / np.sqrt(n_index + 1)
        const_i = (n_index * alpha[n_index - 1]) / np.sqrt(n_index * (n_index + 1))
        cn = np.sqrt(2 * np.pi) / np.sqrt(
            np.power(2.0, n_index) * math.factorial(n_index) * np.sqrt(np.pi)
        )

        hfunc = scipy.special.hermite(n_index)
        hdiff = cn * (rv_gen.pdf(b) * hfunc(b) - rv_gen.pdf(d) * hfunc(d))
        val_i = psi_dgms * hdiff

        alpha.append((coeff_i * np.sum(val_i)) + const_i)
    alpha = np.asarray(alpha)
    return alpha


def fft_pc(dgms: np.ndarray) -> np.ndarray:
    psi_dgm = psi(dgms)
    x = np.linspace(0, dgms.max(), 1000)
    y = np.zeros(x.shape)

    for idx, (b, d) in enumerate(dgms):
        arr_idx = (x >= b) & (x <= d)
        y[arr_idx] += psi_dgm[idx]
    alpha = np.fft.rfft(y)
    return alpha


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

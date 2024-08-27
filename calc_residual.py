import typing
import pickle
import math
import os

from hermite_functions import hermite_functions
from fire import Fire
from tqdm import tqdm
import numpy as np
import scipy
import h5py

import tda_utils
import dataset


class ResidualCalculator(dataset.AirflowSignalProcessor):
    def __init__(self, pt_id: str, data_dir: str, save_dir: str):
        super().__init__(pt_id, data_dir, save_dir)
        self.dgms_to_calc = {
            "sub_irr": [0],
            "sub_airflow": [0],
            "rips_airflow": [0, 1],
        }

        self.dgms_name_map = {
            "rips_airflow": "airflow_rips",
            "sub_airflow": "airflow_sublevel",
            "sub_irr": "irr_sublevel",
        }
        self.approx_types = get_approx_types()

    def calculate_residual(self):
        res_result_dict = {
            "rips_airflow": [
                {x: [] for x in self.approx_types},
                {x: [] for x in self.approx_types},
            ],
            "sub_airflow": [{x: [] for x in self.approx_types}],
            "sub_irr": [{x: [] for x in self.approx_types}],
        }
        with open(self.save_fname_pkl, "rb") as f_pkl:
            dgms_data = pickle.load(f_pkl)

        with h5py.File(self.save_fname_hdf5, "r") as f:
            for dgm_name, h_list in self.dgms_to_calc.items():
                dgm_list = dgms_data[self.dgms_name_map[dgm_name]]
                for idx, dgm_idx in enumerate(tqdm(dgm_list, desc=dgm_name)):
                    for h in h_list:
                        dgm = dgm_idx[h]
                        x, pc = self.calculate_persistence_curve(dgm)
                        for approx_type in self.approx_types:
                            if dgm_name == "sub_irr":
                                coefs = f[f"{approx_type}_irr"][idx]
                            else:
                                coefs = f[f"{approx_type}_{dgm_name}_{h}"][idx]
                            res = self.calculate_res(x, pc, coefs, approx_type)
                            res_result_dict[dgm_name][h][approx_type].append(res)
        return res_result_dict

    def get_dmax(self):
        dmax_result_dict = {
            "rips_airflow": [[], []],
            "sub_airflow": [[]],
            "sub_irr": [[]],
        }
        if not os.path.exists(self.save_fname_pkl):
            return

        with open(self.save_fname_pkl, "rb") as f_pkl:
            dgms_data = pickle.load(f_pkl)

        with h5py.File(self.save_fname_hdf5, "r") as f:
            for dgm_name, h_list in self.dgms_to_calc.items():
                dgm_list = dgms_data[self.dgms_name_map[dgm_name]]
                for dgm_idx in dgm_list:
                    for h in h_list:
                        dgm = dgm_idx[h]
                        dgm_clean = dgm[~np.isinf(dgm).any(1)]
                        dmax_result_dict[dgm_name][h].append(dgm_clean.max())
        return dmax_result_dict

    def calculate_persistence_curve(
        self,
        dgm: np.ndarray,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]
        psi_dgms = tda_utils.psi(dgm_clean)
        x = np.linspace(0, dgm_clean.max(), 1000)
        y = np.zeros(x.shape)

        for idx, (b, d) in enumerate(dgm_clean):
            arr_idx = (x >= b) & (x <= d)
            y[arr_idx] += psi_dgms[idx]
        return x, y

    def calculate_res(
        self,
        x: np.ndarray,
        pc: np.ndarray,
        coefs: np.ndarray,
        approx_type: str,
    ) -> float:
        if approx_type == "fft":
            res = self.calculate_fft_residual(pc, fft_coef=coefs)
        elif approx_type == "hepc":
            res = self.calculate_hepc_residual(x, pc, hepc_coef=coefs)
        else:
            raise RuntimeError("Approximation type not recognized!")
        return res

    def calculate_fft_residual(
        self,
        pc: np.ndarray,
        fft_coef: np.ndarray,
    ) -> float:
        n_coef = 15
        coef_app = fft_coef[:n_coef]
        approx = self.fft_pc_approx(pc, coef_app)
        return self.res(approx, pc)

    def fft_pc_approx(
        self,
        pc: np.ndarray,
        fft_coef: np.ndarray,
    ) -> float:
        coef_app = fft_coef
        approx = np.fft.irfft(coef_app, n=pc.shape[0])
        return approx

    def calculate_hepc_residual(
        self,
        x: np.ndarray,
        pc: np.ndarray,
        hepc_coef: np.ndarray,
    ) -> float:
        y_pred = self.hepc_pc_approx(x, pc, hepc_coef)
        return self.res(y_pred, pc)

    def hepc_pc_approx(
        self,
        x: np.ndarray,
        pc: np.ndarray,
        hepc_coef: np.ndarray,
    ) -> float:
        y_pred = np.zeros(pc.shape)
        for n, coef in enumerate(hepc_coef):
            hn = coef * hermite_functions(n, x, all_n=False).squeeze()
            y_pred += hn
        return y_pred

    @staticmethod
    def res(y1: np.ndarray, y2: np.ndarray) -> float:
        r = (y2 - y1) ** 2
        return r.sum()


def display() -> None:
    result_save_dir = "./residuals/"
    approx_types = get_approx_types()
    res_result_dict = {
        "rips_airflow": [
            {x: [] for x in approx_types},
            {x: [] for x in approx_types},
        ],
        "sub_airflow": [{x: [] for x in approx_types}],
        "sub_irr": [{x: [] for x in approx_types}],
    }
    for fname in tqdm(os.listdir(result_save_dir)):
        fpath = os.path.join(result_save_dir, fname)

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            for h_idx in range(len(v)):
                for atype in approx_types:
                    res_result_dict[k][h_idx][atype] += data[k][h_idx][atype]
    for k, v in res_result_dict.items():
        for h_idx in range(len(v)):
            for atype in approx_types:
                res_list = res_result_dict[k][h_idx][atype]
                res_list_avg = np.mean(res_list)
                print(f"{k} {h_idx} {atype}: {res_list_avg}")


def process_idx(idx):
    data_dir = "/work/thesathlab/nchsdb/"

    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    pt_id = pt_ids[idx]
    save_dir = "/work/thesathlab/manjunath.sh/tda_sleep_staging_ptaf/"

    result_save_dir = "./residuals/"
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    loader = ResidualCalculator(
        pt_id=pt_id,
        data_dir=data_dir,
        save_dir=save_dir,
    )
    subject_ahi = loader.get_ahi()
    subject_age = loader.get_age()

    if (subject_ahi < 1) and (subject_age >= 2) and (subject_age < 18):
        result_dict = loader.calculate_residual()

        save_fname_pkl = os.path.join(result_save_dir, f"{pt_id}_res.pkl")
        with open(os.path.join(save_fname_pkl), "wb") as f:
            pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if subject_ahi >= 1:
            print("Subject AHI too high!")

        if subject_age < 2:
            print("Subject age too low!")

        if subject_age >= 18:
            print("Subject age too high!")


def get_dmax():
    data_dir = "/work/thesathlab/nchsdb/"
    save_dir = "/work/thesathlab/manjunath.sh/tda_sleep_staging_ptaf/"

    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    all_results = {
        "rips_airflow": [[], []],
        "sub_airflow": [[]],
        "sub_irr": [[]],
    }

    for pt_idx, pt_id in enumerate(tqdm(pt_ids)):
        loader = ResidualCalculator(
            pt_id=pt_id,
            data_dir=data_dir,
            save_dir=save_dir,
        )
        subject_ahi = loader.get_ahi()
        subject_age = loader.get_age()

        if (subject_ahi < 1) and (subject_age >= 2) and (subject_age < 18):
            result_dict = loader.get_dmax()

            if result_dict is not None:
                for k, v in result_dict.items():
                    for h_idx, h_dmax_list in enumerate(v):
                        all_results[k][h_idx] += h_dmax_list

    for k, v in all_results.items():
        for h_idx, h_dmax_list in enumerate(v):
            mval = np.mean(h_dmax_list)
            medval = np.median(h_dmax_list)
            xval = np.max(h_dmax_list)
            print(f"{k} H_{h_idx}: {mval} mean, {medval} median, {xval} max")


def get_approx_types():
    return ["hepc", "fft"]


if __name__ == "__main__":
    Fire(
        {
            "process": process_idx,
            "dmax": get_dmax,
            "display": display,
        }
    )

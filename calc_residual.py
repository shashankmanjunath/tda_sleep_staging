from collections import defaultdict
import typing
import pickle
import os

from hermite_functions import hermite_functions
from fire import Fire
from tqdm import tqdm
import numpy as np
import h5py

import tda_utils
import dataset
import utils


def get_approx_types():
    return ["hepc", "sp_fapc", "ap_fapc"]


approx_types_name_map = {
    "hepc": "hepc",
    "sp_fapc": "fft_cf",
    "ap_fapc": "fft",
}


def get_unique_res_subjects(fnames: typing.List) -> typing.List:
    studies_dict = defaultdict(list)

    for fname in fnames:
        pt_id, study_id, _ = fname.split("_")
        studies_dict[pt_id].append(study_id)

    unique_fnames = []
    for k, v in studies_dict.items():
        # Always choose first study_id in list
        chosen_study_id = v[0]
        unique_fnames.append(f"{k}_{chosen_study_id}_res.pkl")
    return unique_fnames


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
        print(f"Subject ID: {self.pt_id}")
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
                        for approx_type in self.approx_types:
                            approx_name = approx_types_name_map[approx_type]
                            if dgm_name == "sub_irr":
                                coefs = f[f"{approx_name}_irr"][idx]
                            else:
                                coefs = f[f"{approx_name}_{dgm_name}_{h}"][idx]
                            dgm_key = f"{dgm_name}_h{h}".replace("sub", "sublevel")
                            res = self.calculate_res(
                                dgm,
                                coefs,
                                approx_type,
                                dgm_key,
                            )
                            res_result_dict[dgm_name][h][approx_type].append(res)

        return res_result_dict

    def get_domain_stats(self) -> typing.Tuple[dict, dict]:
        dmax_result_dict = {
            "rips_airflow": [[], []],
            "sub_airflow": [[]],
            "sub_irr": [[]],
        }
        dmin_result_dict = {
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
                        dmin_result_dict[dgm_name][h].append(dgm_clean.min())
        return dmin_result_dict, dmax_result_dict

    def calculate_persistence_curve(
        self,
        dgm: np.ndarray,
        scale: bool = False,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]

        if scale:
            dgm_clean = dgm_clean * (5 / np.abs(dgm_clean).max())

        psi_dgms = tda_utils.psi(dgm_clean)
        x = np.linspace(dgm_clean.min(), dgm_clean.max(), 1000)
        y = np.zeros(x.shape)

        for idx, (b, d) in enumerate(dgm_clean):
            arr_idx = (x >= b) & (x <= d)
            y[arr_idx] += psi_dgms[idx]
        return x, y

    def calculate_res(
        self,
        dgm: np.ndarray,
        coefs: np.ndarray,
        approx_type: str,
        dgm_key: str,
    ) -> float:
        if approx_type == "ap_fapc":
            x, pc = self.calculate_persistence_curve(dgm, scale=False)
            res = self.calculate_ap_fapc_residual(pc, ap_fapc_coef=coefs)
        elif approx_type == "sp_fapc":
            x, pc = self.calculate_persistence_curve(dgm, scale=False)
            set_domain = dataset.fapc_support[dgm_key]
            L = set_domain[1] - set_domain[0]
            res = self.calculate_sp_fapc_residual(x, L, pc, sp_fapc_coef=coefs)
        elif approx_type == "hepc":
            dgm_scale = dataset.hepc_scale[dgm_key]
            x, pc = self.calculate_persistence_curve(dgm * dgm_scale, scale=True)
            res = self.calculate_hepc_residual(x, pc, hepc_coef=coefs)
        else:
            raise RuntimeError("Approximation type not recognized!")
        #  print(f"{approx_type} {dgm_key}: {res}")
        return res

    def calculate_ap_fapc_residual(
        self,
        pc: np.ndarray,
        ap_fapc_coef: np.ndarray,
    ) -> float:
        n_coef = 15
        coef_app = ap_fapc_coef[:n_coef]
        approx = self.ap_fapc_approx(pc, coef_app)
        return self.res(approx, pc)

    def ap_fapc_approx(
        self,
        pc: np.ndarray,
        ap_fapc_coef: np.ndarray,
    ) -> float:
        approx = np.fft.irfft(ap_fapc_coef, n=pc.shape[0])
        return approx

    def calculate_sp_fapc_residual(
        self,
        x: np.ndarray,
        L: np.ndarray,
        pc: np.ndarray,
        sp_fapc_coef: np.ndarray,
    ) -> float:
        n_coef = 15
        coef_app = sp_fapc_coef[:n_coef]
        approx = self.sp_fapc_approx(x, L, pc, coef_app)
        return self.res(approx, pc)

    def sp_fapc_approx(
        self,
        x: np.ndarray,
        L: np.ndarray,
        pc: np.ndarray,
        sp_fapc_coef: np.ndarray,
    ) -> float:
        approx = np.zeros(pc.shape)
        for n, beta_n in enumerate(sp_fapc_coef):
            if n == 0:
                approx += np.real(beta_n) / 2
            else:
                approx += np.real(beta_n) * np.cos(2 * np.pi * n * x / L) + np.imag(
                    beta_n
                ) * np.sin(2 * np.pi * n * x / L)
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


def display(residual_save_dir: str) -> None:
    approx_types = get_approx_types()
    res_result_dict = {
        "rips_airflow": [
            {x: [] for x in approx_types},
            {x: [] for x in approx_types},
        ],
        "sub_airflow": [{x: [] for x in approx_types}],
        "sub_irr": [{x: [] for x in approx_types}],
    }

    subject_fnames = os.listdir(result_save_dir)
    subject_fnames = get_unique_res_subjects(subject_fnames)

    for fname in tqdm(subject_fnames):
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
                print(f"{k} {h_idx} {atype}: {res_list_avg:.3f}")


def process_idx(
    idx: int,
    preproc_dir: str,
    data_dir: str,
    residual_save_dir: str,
):
    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    pt_id = pt_ids[idx]

    if not os.path.exists(residual_save_dir):
        os.makedirs(residual_save_dir)

    loader = ResidualCalculator(
        pt_id=pt_id,
        data_dir=data_dir,
        save_dir=preproc_dir,
    )
    subject_ahi = loader.get_ahi()
    subject_age = loader.get_age()

    if (subject_ahi < 1) and (subject_age >= 2) and (subject_age < 18):
        result_dict = loader.calculate_residual()

        save_fname_pkl = os.path.join(residual_save_dir, f"{pt_id}_res.pkl")
        with open(os.path.join(save_fname_pkl), "wb") as f:
            pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if subject_ahi >= 1:
            print("Subject AHI too high!")

        if subject_age < 2:
            print("Subject age too low!")

        if subject_age >= 18:
            print("Subject age too high!")


def get_domain_stats(data_dir: str, preproc_dir: str):
    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    dmax_results = {
        "rips_airflow": [[], []],
        "sub_airflow": [[]],
        "sub_irr": [[]],
    }
    dmin_results = {
        "rips_airflow": [[], []],
        "sub_airflow": [[]],
        "sub_irr": [[]],
    }

    for pt_idx, pt_id in enumerate(tqdm(pt_ids)):
        loader = ResidualCalculator(
            pt_id=pt_id,
            data_dir=data_dir,
            save_dir=preproc_dir,
        )
        subject_ahi = loader.get_ahi()
        subject_age = loader.get_age()

        if (subject_ahi < 1) and (subject_age >= 2) and (subject_age < 18):
            retval = loader.get_domain_stats()
            if retval is not None:
                dmin_result_dict, dmax_result_dict = retval
            else:
                continue

            if dmax_result_dict is not None:
                for k, v in dmax_result_dict.items():
                    for h_idx, h_dmax_list in enumerate(v):
                        dmax_results[k][h_idx] += h_dmax_list
            if dmin_result_dict is not None:
                for k, v in dmin_result_dict.items():
                    for h_idx, h_dmin_list in enumerate(v):
                        dmin_results[k][h_idx] += h_dmin_list

    for k, v in dmax_results.items():
        for h_idx, h_dmax_list in enumerate(v):
            mval = np.mean(h_dmax_list)
            medval = np.median(h_dmax_list)
            xval = np.max(h_dmax_list)
            print(f"dmax {k} H_{h_idx}: {mval} mean, {medval} median, {xval} max")

    for k, v in dmin_results.items():
        for h_idx, h_dmin_list in enumerate(v):
            mval = np.mean(h_dmin_list)
            medval = np.median(h_dmin_list)
            xval = np.min(h_dmin_list)
            print(f"dmin {k} H_{h_idx}: {mval} mean, {medval} median, {xval} min")


if __name__ == "__main__":
    Fire(
        {
            "process": process_idx,
            "get_domain_stats": get_domain_stats,
            "display": display,
        }
    )

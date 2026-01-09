"""
Ultility functions
"""

import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from mrtool import MRBRT, MRBeRT, MRData


def fill_dict(des_dict: Dict, default_dict: Dict) -> Dict:
    """Fill given dictionary by the default dictionary.

    Parameters
    ----------
    des_dict : Dict
        Given dictionary that needs to be filled.
    default_dict : Dict
        Default dictionary

    Returns
    -------
    Dict
        Updated dictionary.
    """
    for key, value in default_dict.items():
        if key not in des_dict:
            des_dict[key] = value
        elif isinstance(value, dict):
            des_dict[key] = fill_dict(des_dict[key], value)
    return des_dict


def get_beta_info(
    model: MRBRT, cov_name: str | None = "signal"
) -> Tuple[float, float]:
    """Get the posterior information of beta.

    Parameters
    ----------
    model : MRBRT
        MRBRT model, preferably is a simple linear mixed effects model.
    cov_name : str, optional
        Name of the interested covariates, default to be "signal".

    Returns
    -------
    Tuple[float, float]
        Return the mean and standard deviation of the corresponding beta.
    """
    lt = model.lt
    if cov_name is None:
        beta = model.beta_soln.copy()
        beta_hessian = lt.hessian(lt.soln)[: lt.k_beta, : lt.k_beta]
        beta_sd = 1.0 / np.sqrt(np.diag(beta_hessian))
    else:
        index = model.cov_names.index(cov_name)
        beta = model.beta_soln[index]
        beta_hessian = lt.hessian(lt.soln)[: lt.k_beta, : lt.k_beta]
        beta_sd = 1.0 / np.sqrt(beta_hessian[index, index])
    return (beta, beta_sd)


def get_gamma_info(model: MRBRT) -> Tuple[float, float]:
    """Get the posterior information of gamma.

    Parameters
    ----------
    model : MRBRT
        MRBRT model, preferably is a simple linear mixed effects model. Requires
        only have one random effect.

    Returns
    -------
    Tuple[float, float]
        Return the mean and standard deviation of the corresponding gamma.
    """
    lt = model.lt
    gamma = model.gamma_soln[0]
    gamma_fisher = lt.get_gamma_fisher(lt.gamma)
    gamma_sd = 1.0 / np.sqrt(gamma_fisher[0, 0])
    return (gamma, gamma_sd)


def get_signal(
    signal_model: MRBeRT,
    risk: np.ndarray,
    risk_l_linear: float | None = None,
    risk_r_linear: float | None = None,
    tmrel: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk grid and signal. Will always anchor the curve at lnRR = 0
    at the data minimum and supports linear extrapolation and truncation

    Parameters
    ----------
    signal_model
        Fitted signal model object.
    risk
        Risk exposure vector, taken to be user-defined (may be data-supported or
        include extrapolation or truncation), that we want to create signal on
    risk_l_linear
        Lower risk exposure bound of data; None if no left extrapolation
    risk_r_linear
        Upper risk exposure bound of data; None if no right extrapolation
    tmrel
        Lower risk exposure bound of data when risk.min() > min risk in data;
        None if no left truncation


    Returns
    -------
    tuple[NDArray, NDArray]
        Risk values, derived from data or risk_lower/risk_upper settings if provided,
        and signal over the defined risk values, accounting for any extrapolation
        or truncation to guarantee the signal is identical to original signal
        predicted over the data-supported range.
    """
    # Get minimum risk from data to get anchor point for lnRR=0
    data_min = (
        risk_l_linear
        if risk_l_linear is not None
        else (tmrel if tmrel is not None else risk.min())
    )
    # Get max risk from data and nearby points for slope calculation
    data_max = risk_r_linear if risk_r_linear is not None else risk.max()
    eps = (data_max - data_min) * 1e-5
    anchor_and_slope_points = [
        data_min,
        data_min + eps,
        data_max - eps,
        data_max,
    ]
    risk_full = np.unique(
        np.sort(np.concatenate([risk, anchor_and_slope_points]))
    )
    idx = np.searchsorted(risk_full, anchor_and_slope_points)
    # signal_full = get_signal(signal_model, risk_full)
    signal_full = signal_model.predict(
        MRData(
            covs={
                "ref_risk_lower": np.repeat(risk_full.min(), len(risk_full)),
                "ref_risk_upper": np.repeat(risk_full.min(), len(risk_full)),
                "alt_risk_lower": risk_full,
                "alt_risk_upper": risk_full,
            }
        )
    )

    # Handle extrapolation
    if risk_l_linear is not None:
        slope_left = (signal_full[idx[1]] - signal_full[idx[0]]) / eps
        mask_left = risk_full < data_min
        signal_full[mask_left] = signal_full[idx[0]] + slope_left * (
            risk_full[mask_left] - data_min
        )
    if risk_r_linear is not None:
        slope_right = (signal_full[idx[3]] - signal_full[idx[2]]) / eps
        mask_right = risk_full > data_max
        signal_full[mask_right] = signal_full[idx[3]] + slope_right * (
            risk_full[mask_right] - data_max
        )

    # Offset signal to re-anchor to lnRR = 0 at min risk from data
    signal_anchor = signal_full[idx[0]]
    signal_full -= signal_anchor

    # Return signal for original risk – handles truncation, removes data anchor and slope points
    final_idx = np.searchsorted(risk_full, risk)
    final_signal = signal_full[final_idx]

    return risk, final_signal


def get_risk_bounds(
    settings: dict,
    summary: dict,
) -> tuple[np.ndarray, list[float | None]]:
    """
    Parameters
    ----------
    settings
        Settings file with user-defined risk bounds.
    summary
        Summary file with data-supported risk bounds.

    Returns
    -------
    tuple[np.ndarray, list[float | None]]
        Returns an array defining the output risk grid for quantiles and draws, and
        a list [risk_l_linear, risk_r_linear, tmrel] derived from the data and used
        to anchor/extrapolate/truncate the signal. Values are None if settings are
        equal to data-supported risk bounds.

    """
    data_risk_lower, data_risk_upper = summary["risk_bounds"]
    if (risk_lower := settings["draws"]["risk_lower"]) is None:
        risk_lower = data_risk_lower
    if (risk_upper := settings["draws"]["risk_upper"]) is None:
        risk_upper = data_risk_upper

    num_points = settings["draws"]["num_points"]
    modeling_range = np.linspace(risk_lower, risk_upper, num_points)

    risk_l_linear = data_risk_lower if risk_lower < data_risk_lower else None
    risk_r_linear = data_risk_upper if risk_upper > data_risk_upper else None
    tmrel = data_risk_lower if risk_lower > data_risk_lower else None
    modeling_bounds = [risk_l_linear, risk_r_linear, tmrel]

    return modeling_range, modeling_bounds


def score_to_star_rating(score: float) -> int:
    """Takes in risk outcome score(s) and returns associated star rating(s)

    Parameters
    ----------
    score
        risk outcome scores

    Returns
    -------
    int
        Associated star rating
    """
    if np.isnan(score):
        return 0
    elif score > np.log(1 + 0.85):
        return 5
    elif score > np.log(1 + 0.5):
        return 4
    elif score > np.log(1 + 0.15):
        return 3
    elif score > 0:
        return 2
    else:
        return 1


def get_point_estimate_and_UIs(
    inner_quantiles: pd.DataFrame, outer_quantiles: pd.DataFrame
) -> pd.DataFrame:
    """Get the point estimate, inner and outer UIs, and BPRF in log and linear
    space for a single combined summary file

    Parameters
    ----------
    inner_quantiles : DataFrame
        Inner quantiles csv file containing point estimate and 95% traditional
        UIs (all log space)
    outer_quantiles : DataFrame
        Outer quantiles csv file containing point estimate, 95% UIs with between-
        study heterogeneity, and the BPRF (all log space)

    Returns
    -------
    DataFrame
        Combined dataframe with point estimate, inner and outer 95% UIs, and
        BPRF, all in both log and linear space.
    """
    risk_col_options = ["risk", "ref_risk_cat", "alt_risk_cat", "risk_cat_pair"]
    risk_cols = inner_quantiles.columns.intersection(risk_col_options).to_list()

    df_summary = inner_quantiles[risk_cols].copy()
    df_summary["log_point_estimate"] = inner_quantiles["0.5"]
    df_summary["outer_log_UI_lower"] = outer_quantiles["0.025"]
    df_summary["inner_log_UI_lower"] = inner_quantiles["0.025"]
    df_summary["inner_log_UI_upper"] = inner_quantiles["0.975"]
    df_summary["outer_log_UI_upper"] = outer_quantiles["0.975"]
    pred = df_summary["log_point_estimate"]
    log_bprf = np.where(
        pred > 0, outer_quantiles["0.05"], outer_quantiles["0.95"]
    )
    df_summary["log_bprf"] = log_bprf

    log_col_names = [col for col in df_summary.columns if "log_" in col]
    for log_col in log_col_names:
        linear_col_name = log_col.replace("log_", "linear_")
        df_summary[linear_col_name] = np.exp(df_summary[log_col])

    return df_summary


def _validate_required_quantiles(
    user_quantiles: np.ndarray,
) -> np.ndarray:
    """Ensure that the user-specified quantiles from settings.yaml include the
    required quantiles to generate the point estimate (mean), 95% UIs, and BPRF.
    If any of these required quantiles are missing, they will be added to the
    list of generated quantiles.

    Parameters
    ----------
    user_quantiles: array
        Quantiles specified by the user in the settings.yaml file

    Returns
    -------
    Sorted array of quantiles, including all required quantiles.
    """
    user_list = user_quantiles.tolist()
    required_quantiles = (0.025, 0.05, 0.5, 0.95, 0.975)
    all_quantiles = set(user_list) | set(required_quantiles)
    return np.array(sorted(all_quantiles), dtype=float)


class ParseKwargs(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str],
        option_string: Optional[str] = None,
    ):
        """Parse keyword arguments into a dictionary. Be sure to set `nargs='*'`
        in the ArgumentParser to parse the input as a list of strings, otherwise
        this function will break. If provided string does not the form of
        '{key}={value}', an error will be raised.

        Example
        -------

        .. code-block:: python

            parser = ArgumentParser()
            parser.add_argument(
                "-m",
                "--metadata",
                nargs="*",
                required=False,
                default={},
                action=ParseKwargs,
            )

        """
        data = dict()
        for value in values:
            data_key, _, data_value = value.partition("=")
            if (not data_key) or (not data_value):
                raise ValueError(
                    "please provide kwargs in the form of {key}={value}, "
                    f"current input is '{value}'"
                )
            data[data_key] = data_value
        setattr(namespace, self.dest, data)

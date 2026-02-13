"""
Ultility functions
"""

import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from mrtool import MRBRT, MRBeRT, MRData
from pandas import DataFrame


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


def _get_signal_value(
    signal_model: MRBeRT, risk: np.ndarray, ref: float
) -> np.ndarray:
    """Get signal from signal_model over input risk vector, anchored at given
    reference exposure. Ref will almost always be minimum risk in data to ensure
    risk curve is anchored at the data minimum; if curve will be normalized
    to tmrel that will happen after signal is predicted.

    Parameters
    ----------
    signal_model
        Fitted signal model object.
    risk
        Risk exposure vector, taken to be user-defined (may be data-supported or
        include extrapolation or truncation), that we want to create signal on
    ref
        Reference risk value that signal is predicted relative to

    Returns
    -------
    ndarray
        Predicted signal, anchored at lnRR = 0 at ref exposure value.
    """
    risk = np.asarray(risk)
    signal = signal_model.predict(
        MRData(
            covs={
                "ref_risk_lower": np.repeat(ref, len(risk)),
                "ref_risk_upper": np.repeat(ref, len(risk)),
                "alt_risk_lower": risk,
                "alt_risk_upper": risk,
            }
        )
    )
    return signal


def _get_signal_slope(
    signal_model: MRBeRT,
    risk: np.ndarray,
    ref: float,
    dx: float | None = 1e-2,
) -> np.ndarray:
    """Get slope(s) of signal at specified risk points for linear extrapolation.

    Parameters
    ----------
    signal_model
        Fitted signal model object.
    risk
        Array of risk value(s) at which to compute the slope
    ref
        Lower bound of risk from data, used to obtain reference risk value to anchor signal
    dx
        Optional, step size for computing slope

    Returns
    -------
    ndarray
        Slope(s) at risk values for use in linear extrapolation
    """
    risk = np.asarray(risk)
    risk_offset = risk + dx

    signal = _get_signal_value(signal_model, risk, ref)
    signal_offset = _get_signal_value(signal_model, risk_offset, ref)
    slopes = (signal_offset - signal) / dx
    return slopes


def get_signal(
    signal_model: MRBeRT,
    risk: np.ndarray,
    risk_bounds: tuple[float, float],
    risk_l_linear: float | None = None,
    risk_r_linear: float | None = None,
    tmrel: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk grid and final signal. Will always anchor the curve at lnRR = 0
    at the data minimum and supports linear extrapolation and truncation

    Parameters
    ----------
    signal_model
        Fitted signal model object.
    risk
        Risk exposure vector, taken to be user-defined (may be data-supported or
        include extrapolation or truncation), that we want to create signal on
    risk_bounds
        Risk bounds from data
    risk_l_linear
        Lower risk exposure bound of data; None if no left extrapolation
    risk_r_linear
        Upper risk exposure bound of data; None if no right extrapolation
    tmrel
        Risk exposure value to shift curve to re-anchor (J-shaped curve), None
        unless normalize_to_tmrel = True in settings

    Returns
    -------
    tuple[NDArray, NDArray]
        Risk values, derived from data or risk_lower/risk_upper settings if provided,
        and signal over the defined risk values, accounting for any extrapolation
        or truncation to guarantee the signal is identical to original signal
        predicted over the data-supported range.
    """
    # Get minimum risk from data to get reference risk value for lnRR=0
    data_min, data_max = risk_bounds
    risk = np.asarray(risk)
    # Get signal over full risk vector, anchored to lnRR = 0 at data_min
    pred = _get_signal_value(signal_model, risk, data_min)

    # Extrapolation
    if risk_l_linear is not None and risk_l_linear > risk.min():
        val = _get_signal_value(signal_model, [risk_l_linear], data_min)
        slope_l_linear = _get_signal_slope(signal_model, [data_min], data_min)
        index = risk < risk_l_linear
        pred[index] = val + slope_l_linear * (risk[index] - risk_l_linear)
    else:
        risk_l_linear = None
    if risk_r_linear is not None and risk_r_linear < risk.max():
        val = _get_signal_value(signal_model, [risk_r_linear], data_min)
        slope_r_linear = _get_signal_slope(signal_model, [data_max], data_min)
        index = risk > risk_r_linear
        pred[index] = val + slope_r_linear * (risk[index] - data_max)
    else:
        risk_r_linear = None

    # Shift curve to TMREL
    offset = 0.0
    if tmrel is not None:
        offset = _get_signal_value(signal_model, [tmrel], data_min)
    pred -= offset

    return risk, pred


def get_risk_bounds(
    settings: dict,
    summary: dict,
    signal_model: MRBeRT,
) -> tuple[np.ndarray, list[float | None]]:
    """
    Parameters
    ----------
    settings
        Settings file with user-defined risk bounds.
    summary
        Summary file with data-supported risk bounds.
    signal_model
        Fitted signal model for calculation of tmrel

    Returns
    -------
    tuple[np.ndarray, list[float | None]]
        Returns an array defining the output risk grid for quantiles and draws, and
        a list [risk_l_linear, risk_r_linear, tmrel] derived from the data/ and used
        to anchor/extrapolate/truncate the signal. Values are None if settings are
        equal to data-supported risk bounds.

    """
    data_risk_min, data_risk_max = summary["risk_bounds"]
    if (risk_lower := settings["draws"]["risk_lower"]) is None:
        risk_lower = data_risk_min
    if (risk_upper := settings["draws"]["risk_upper"]) is None:
        risk_upper = data_risk_max
    num_points = settings["draws"]["num_points"]
    modeling_range = np.linspace(risk_lower, risk_upper, num_points)

    risk_l_linear = data_risk_min if risk_lower < data_risk_min else None
    risk_r_linear = data_risk_max if risk_upper > data_risk_max else None
    if summary["normalize_to_tmrel"]:
        risk_from_data = np.linspace(data_risk_min, data_risk_max, num_points)
        signal = _get_signal_value(signal_model, risk_from_data, data_risk_min)
        tmrel = risk_from_data[np.argmin(signal)]
    else:
        tmrel = None
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

"""
Ultility functions
"""

import argparse
from typing import Any, Dict, Optional, Tuple

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


def get_signal(signal_model: MRBeRT, risk: np.ndarray) -> np.ndarray:
    """Get signal from signal_model

    Parameters
    ----------
    signal_model : MRBeRT
        Signal model object.
    risk : np.ndarray
        Risk exposures that we want to create signal on.
    """
    return signal_model.predict(
        MRData(
            covs={
                "ref_risk_lower": np.repeat(risk.min(), len(risk)),
                "ref_risk_upper": np.repeat(risk.min(), len(risk)),
                "alt_risk_lower": risk,
                "alt_risk_upper": risk,
            }
        )
    )


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
    risk_cols = [
        col for col in risk_col_options if col in inner_quantiles.columns
    ]

    df_summary = inner_quantiles[risk_cols].copy()
    df_summary["log_point_estimate"] = inner_quantiles[0.5]
    df_summary["outer_log_UI_lower"] = outer_quantiles[0.025]
    df_summary["inner_log_UI_lower"] = inner_quantiles[0.025]
    df_summary["inner_log_UI_upper"] = inner_quantiles[0.975]
    df_summary["outer_log_UI_upper"] = outer_quantiles[0.975]
    pred = df_summary["log_point_estimate"]
    log_bprf = np.where(pred > 0, outer_quantiles[0.05], outer_quantiles[0.95])
    df_summary["log_bprf"] = log_bprf

    log_col_names = [col for col in df_summary.columns if "log_" in col]
    for log_col in log_col_names:
        linear_col_name = log_col.replace("log_", "linear_")
        df_summary[linear_col_name] = np.exp(df_summary[log_col])

    return df_summary


def _validate_required_quantiles(
    user_quantiles: list[float],
    required_quantiles: list[float] = [0.025, 0.05, 0.5, 0.95, 0.975],
) -> list[float]:
    """Ensure that the user-specified quantiles from settings.yaml include the
    required quantiles to generate the point estimate (mean), 95% UIs, and BPRF.
    If any of these required quantiles are missing, they will be added to the
    list of generated quantiles.

    Parameters
    ----------
    user_quantiles: list of floats
        Quantiles specified by the user in the settings.yaml file
    required_quantiles: list of floats
        Quantiles that must (also) be included for final summary outputs. Default
        is [0.025, 0.05, 0.5, 0.95, 0.975] corresponding to lower UI bound, BPRF
        (if harmful), point estimate, BPRF (if protective), upper UI bound.

    Returns
    -------
    Sorted list of quantiles, including all required quantiles, as a list of floats
    """
    all_quantiles = set(user_quantiles) | set(required_quantiles)
    return sorted(all_quantiles)


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

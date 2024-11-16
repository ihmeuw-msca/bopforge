from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes, Figure
from mrtool import MRBRT, CovFinder, LinearCatCovModel, LinearCovModel, MRData
from pandas import DataFrame
from scipy.stats import norm

from bopforge.utils import get_beta_info, get_gamma_info


def get_signal_model(settings: dict, df: DataFrame) -> MRBRT:
    """Create signal model for outliers identification and covariate selection
    step.

    Parameters
    ----------
    settings
        Dictionary contains all the settings.
    df
        Data frame contains the training data.

    Returns
    -------
    MRBRT
        Signal model to access the strength of the prior on the bias-covariate.

    """
    col_covs = [col for col in df.columns if col.startswith("cov_")]
    data = MRData()
    data.load_df(
        df,
        col_obs="ln_rr",
        col_obs_se="ln_rr_se",
        col_covs=col_covs + ["ref_risk_cat", "alt_risk_cat"],
        col_study_id="study_id",
        col_data_id="seq",
    )
    cov_models = [
        LinearCatCovModel(
            alt_cov="alt_risk_cat",
            ref_cov="ref_risk_cat",
            ref_cat=settings["cat_cov_model"]["ref_cat"],
            prior_order=settings["cat_cov_model"]["prior_order"],
            use_re=False,
        )
    ]

    signal_model = MRBRT(data, cov_models, **settings["signal_model"])

    return signal_model


def add_cols(df: DataFrame, signal_model: MRBRT) -> DataFrame:
    """Add columns of outlier indicator.

    Parameters
    ----------
    df
        Data frame that contains the training data.
    signal_model
        Fitted signal model.

    Returns
    -------
    DataFrame
        DataFrame with additional columns of oulier indicator.

    """

    data = signal_model.data
    is_outlier = (signal_model.w_soln < 0.1).astype(int)
    df = df.merge(
        pd.DataFrame({"seq": data.data_id, "is_outlier": is_outlier}),
        how="outer",
        on="seq",
    )

    return df


# def convert_bc_to_em(df: DataFrame, signal_model: MRBRT) -> DataFrame:
#     """Convert bias covariate to effect modifier and add one column to indicate
#     if data points are inliers or outliers.

#     Parameters
#     ----------
#     df
#         Data frame contains the training data.
#     signal_model
#         Fitted signal model.

#     Returns
#     -------
#     DataFrame
#         DataFrame with additional columns for effect modifiers and oulier
#         indicator.

#     """
#     data = signal_model.data
#     signal = signal_model.predict(data)
#     is_outlier = (signal_model.w_soln < 0.1).astype(int)
#     df = df.merge(
#         pd.DataFrame(
#             {
#                 "seq": data.data_id,
#                 "signal": signal,
#                 "is_outlier": is_outlier,
#             }
#         ),
#         how="outer",
#         on="seq",
#     )
#     for col in df.columns:
#         if col.startswith("cov_"):
#             df["em" + col[3:]] = df[col] * df["signal"]

#     return df


def get_signal_model_summary(
    name: str, all_settings: dict, df: DataFrame, df_coef: DataFrame
) -> dict:
    """Create signal model summary.

    Parameters
    ----------
    name
        Name of the pair.
    all_settings
        All the settings for the pipeline.
    df
        Data frame that contains the training dataset.
    df_coef
        Data frame containing beta coefficients

    Returns
    -------
    dict
        Summary dictionary from the signal model.

    """
    ref_cat = all_settings["fit_signal_model"]["cat_cov_model"]["ref_cat"]
    if ref_cat:
        ref_cat = ref_cat
    else:
        unique_cats, counts = np.unique(
            np.hstack([df.ref_risk_cat, df.alt_risk_cat]), return_counts=True
        )
        ref_cat = unique_cats[counts.argmax()]

    summary = {
        "name": name,
        "risk_type": str(df.risk_type.values[0]),
        "beta_coef_signal": dict(zip(df_coef.cat, df_coef.coef)),
        "ref_cat": ref_cat,
    }
    summary["normalize_to_tmrel"] = all_settings["complete_summary"]["score"][
        "normalize_to_tmrel"
    ]
    return summary


# def get_cov_finder_linear_model(df: DataFrame) -> MRBRT:
#     """Create the linear model for the CovFinder to determine the strength of
#     the prior on the bias covariates.

#     Parameters
#     ----------
#     df
#         Data frame that contains the training data, but without the outlier.

#     Returns
#     -------
#     MRBRT
#         The linear model.

#     """
#     col_covs = ["signal"] + [col for col in df.columns if col.startswith("em_")]
#     data = MRData()
#     data.load_df(
#         df,
#         col_obs="ln_rr",
#         col_obs_se="ln_rr_se",
#         col_covs=col_covs,
#         col_study_id="study_id",
#         col_data_id="seq",
#     )
#     cov_models = [
#         LinearCovModel("signal", use_re=True),
#     ]
#     cov_finder_linear_model = MRBRT(data, cov_models)

#     return cov_finder_linear_model


def get_cov_finder(settings: dict, cov_finder_linear_model: MRBRT) -> CovFinder:
    """Create the instance of CovFinder class.

    Parameters
    ----------
    settings
        Settings for bias covariate selection.
    cov_finder_linear_model
        Fitted cov finder linear model.

    Returns
    -------
    CovFinder
        The instance of the CovFinder class.

    """
    ###################
    data = cov_finder_linear_model.data
    cats = cov_finder_linear_model.cov_models[0].cats
    ref_cat = cov_finder_linear_model.cov_models[0].ref_cat
    alt_cats = [cat for cat in cats if cat != ref_cat]

    alt_mat, ref_mat = cov_finder_linear_model.cov_models[0].create_design_mat(
        data
    )
    design_mat = alt_mat - ref_mat
    df = data.to_df()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                design_mat,
                columns=[col for col in cats],
            ),
        ],
        axis=1,
    )
    df["seq"] = range(df.shape[0])
    col_covs = [col for col in df.columns if col.startswith("cov_")]
    col_alt_cats = [col for col in df.columns if col in alt_cats]
    data = MRData()
    data.load_df(
        df,
        col_obs="obs",
        col_obs_se="obs_se",
        col_covs=col_covs + col_alt_cats + ["ref_risk_cat", "alt_risk_cat"],
        col_study_id="study_id",
        col_data_id="seq",
    )

    beta_info = get_beta_info(cov_finder_linear_model, cov_name=None)
    index = list(cats).index(ref_cat)
    beta_info = tuple(np.delete(arr, index) for arr in beta_info)

    ###################

    # TODO: adjust pre-selected covariates, add alt cats.
    bias_covs = [name for name in data.covs.keys() if name.startswith("cov_")]

    # covariate selection
    pre_selected_covs = settings["cov_finder"]["pre_selected_covs"]
    if isinstance(pre_selected_covs, str):
        pre_selected_covs = [pre_selected_covs]
    if "intercept" not in pre_selected_covs:
        pre_selected_covs.append("intercept")
    settings["cov_finder"]["pre_selected_covs"] = pre_selected_covs
    candidate_covs = [
        cov_name
        for cov_name in bias_covs + alt_cats
        # for cov_name in data.covs.keys()
        if cov_name not in pre_selected_covs
    ]
    settings["cov_finder"] = {
        **dict(
            num_samples=1000,
            power_range=[-4, 4],
            power_step_size=0.05,
            laplace_threshold=0.00001,
            inlier_pct=1.0,
            bias_zero=True,
        ),
        **settings["cov_finder"],
    }
    cov_finder = CovFinder(
        data,
        covs=candidate_covs,
        # beta_gprior_std=0.1 * beta_info[1],
        **settings["cov_finder"],
    )

    return cov_finder


def get_cov_finder_result(
    cov_finder_linear_model: MRBRT, cov_finder: CovFinder
) -> dict:
    """Summarize result from bias covariate selection.

    Parameters
    ----------
    cov_finder_linear_model
        Fitted cov finder linear model.
    cov_finder
        Fitted cov finder model.

    Returns
    -------
    dict
        Result summary for bias covariate selection.

    """
    beta_info = get_beta_info(cov_finder_linear_model, cov_name=None)
    selected_covs = [
        cov_name
        for cov_name in cov_finder.selected_covs
        if cov_name != "intercept"
    ]

    # save results
    cov_finder_result = {
        "beta_sd": (beta_info[1] * 0.1).tolist(),
        # "beta_sd": float(beta_info[1] * 0.1),
        "selected_covs": selected_covs,
    }

    return cov_finder_result


def get_coefs(
    df: DataFrame,
    settings: dict,
    signal_model: MRBRT,
) -> tuple[DataFrame]:
    """Get beta coefficients for the categories.

    Parameters
    ----------
    df
        Data frame contains the training data.
    settings
        The settings for category order
    signal_model
        Fitted signal model for risk curve.
    linear_model
        Fitted linear model for risk curve. Default is `None`. When it is `None`
        the coefficients are extracted from the signal model. When a linear
        model is provided, the coefficients are extracted from the linear model.

    Returns
    -------
    tuple[DataFrame]
        Dataframe of beta coefficients and ranges to plot each category

    """
    # extract categories, betas
    df_coef = pd.DataFrame(
        dict(cat=signal_model.cov_models[0].cats, coef=signal_model.beta_soln)
    )
    cat_order = settings["figure"]["cat_order"]
    ref_cat = settings["fit_signal_model"]["cat_cov_model"]["ref_cat"]
    # Assign reference category from settings or by most common category
    if ref_cat:
        ref_cat = ref_cat
    else:
        unique_cats, counts = np.unique(
            np.hstack([df.ref_risk_cat, df.alt_risk_cat]), return_counts=True
        )
        ref_cat = unique_cats[counts.argmax()]
    # Order the categories
    if cat_order:
        df_coef["cat"] = pd.Categorical(
            df_coef["cat"], categories=cat_order, ordered=True
        )
        df_coef = df_coef.sort_values("cat").reset_index(drop=True)
    else:
        # Default behavior: ref_cat first, then order by proximity to ref_cat's coef
        ref_coef = df_coef.loc[df_coef["cat"] == ref_cat, "coef"].iloc[0]
        df_coef["abs_diff"] = abs(df_coef["coef"] - ref_coef)
        df_coef = df_coef.sort_values(by="abs_diff")
        df_coef = pd.concat(
            [
                df_coef[df_coef["cat"] == ref_cat],
                df_coef[df_coef["cat"] != ref_cat],
            ]
        ).reset_index(drop=True)
    # Add x ranges
    num_cats = df_coef["cat"].nunique()
    df_coef["x_start"] = range(num_cats)
    df_coef["x_end"] = range(1, num_cats + 1)
    df_coef["x_mid"] = df_coef.eval("0.5 * (x_start + x_end)")

    return df_coef


def plot_signal_model(
    name: str,
    summary: dict,
    df: DataFrame,
    df_coef: DataFrame,
    signal_model: MRBRT,
    show_ref: bool = True,
) -> Figure:
    """Plot the signal model

    Parameters
    ----------
    name
        Name of the pair.
    summary
        Summary from the signal model.
    df
        Data frame contains training data.
    df_coef
        Data frame containing the fitted beta coefficients
    signal_model
        Fitted signal model for risk curve.
    show_ref
        Whether to show the reference line. Default is `True`.

    Returns
    -------
    Figure
        The figure object for signal model.

    """
    offset = 0.05
    # create fig obj
    fig, ax = plt.subplots(figsize=(8, 5))

    # plot data
    _plot_data(
        name,
        summary,
        df,
        df_coef,
        ax,
        signal_model=signal_model,
        show_ref=show_ref,
    )

    # plot beta coefficients
    if summary["normalize_to_tmrel"]:
        coef_min = df_coef.coef.min()
        for i, row in df_coef.iterrows():
            ax.plot(
                [row["x_start"] + offset, row["x_end"] - offset],
                [row["coef"] - coef_min] * 2,
                color="#008080",
            )
    else:
        for i, row in df_coef.iterrows():
            ax.plot(
                [row["x_start"] + offset, row["x_end"] - offset],
                [row["coef"]] * 2,
                color="#008080",
            )

    return fig


def get_linear_model(
    df: DataFrame, settings: dict, cov_finder_result: dict
) -> MRBRT:
    """Create linear model for effect.

    Parameters
    ----------
    df
        Data frame contains training data without outliers.
    settings
        Settings for the categories
    cov_finder_result
        Summary result for bias covariate selection.

    Returns
    -------
    MRBRT
        The linear model for effect.

    """
    col_covs = cov_finder_result["selected_covs"]
    beta_sd = cov_finder_result["beta_sd"]
    cov_dict = {cov: beta for cov, beta in zip(col_covs, beta_sd)}
    bias_covs = [cov for cov in col_covs if cov.startswith("cov")]
    beta_sd_bias_covs = [cov_dict[cov] for cov in bias_covs]

    data = MRData()
    data.load_df(
        df,
        col_obs="ln_rr",
        col_obs_se="ln_rr_se",
        col_covs=bias_covs,
        col_study_id="study_id",
        col_data_id="seq",
    )
    cov_models = [
        LinearCatCovModel(
            alt_cov="alt_risk_cat",
            ref_cov="ref_risk_cat",
            ref_cat=settings["fit_signal_model"]["cat_cov_model"]["ref_cat"],
            prior_order=settings["fit_signal_model"]["cat_cov_model"][
                "prior_order"
            ],
            use_re=True,
        )
    ]
    # cov_models = [
    #     LinearCovModel("signal", use_re=True),
    #     LinearCovModel("intercept", use_re=True, prior_beta_uniform=[0.0, 0.0]),
    # ]
    # for cov_name in cov_finder_result["selected_covs"]:
    for cov_name in bias_covs:
        cov_models.append(
            LinearCovModel(
                cov_name,
                prior_beta_gaussian=[0.0, beta_sd_bias_covs],
            )
        )
    model = MRBRT(data, cov_models)
    # model = MRBRT(data, cov_models, **settings["signal_model"])
    return model


def get_linear_model_summary(
    settings: dict,
    summary: dict,
    df: DataFrame,
    df_coef: DataFrame,
    linear_model: MRBRT,
) -> dict:
    """Complete the summary from the signal model.

    Parameters
    ----------
     settings
        Settings for the complete summary section.
    summary
        Summary from the signal model.
    df
        Data frame contains the all dataset.
    df_coef
        Data frame containing beta coefficients
    signal_model
        Fitted signal model for risk curve.
    linear_model
        Fitted linear model for risk curve.

    Returns
    -------
    dict
        Summary file contains all necessary information.

    """
    # load summary
    summary["normalize_to_tmrel"] = settings["score"]["normalize_to_tmrel"]
    ref_cat = summary["ref_cat"]

    # solution of the final model
    beta_info = get_beta_info(linear_model)
    gamma_info = get_gamma_info(linear_model)
    summary["beta"] = [float(beta_info[0]), float(beta_info[1])]
    summary["gamma"] = [float(gamma_info[0]), float(gamma_info[1])]

    # compute the score and add star rating
    beta_sd = np.sqrt(beta_info[1] ** 2 + gamma_info[0] + 2 * gamma_info[1])
    pred = df_coef.coef * beta_info[0]
    inner_ui = np.vstack(
        [
            df_coef.coef * (beta_info[0] - 1.96 * beta_info[1]),
            df_coef.coef * (beta_info[0] + 1.96 * beta_info[1]),
        ]
    )
    burden_of_proof = df_coef.coef * (beta_info[0] - 1.645 * beta_sd)

    if settings["score"]["normalize_to_tmrel"]:
        index = np.argmin(pred)
        pred -= pred[index]
        burden_of_proof -= burden_of_proof[None, index]
        inner_ui -= inner_ui[:, None, index]

    sign = np.sign(pred)
    signed_bprf = sign * burden_of_proof
    # Number of alternative categories
    n = pred.size - 1
    # Index with largest signed coefficient
    max_idx = np.argmax(signed_bprf)
    if np.any(np.prod(inner_ui[:,], axis=0) < 0):
        summary["score"] = float("nan")
        summary["star_rating"] = 0
    else:
        score = float(
            (1 / n) * (np.sum(signed_bprf) - 0.5 * signed_bprf[max_idx])
        )
        summary["score"] = score
        df_alt_coefs = df_coef[df_coef["cat"] != ref_cat]
        summary["score_by_category"] = dict(
            zip(df_alt_coefs.cat, (0.5 * signed_bprf[df_alt_coefs.index]))
        )
        # Assign star rating based on ROS
        if np.isnan(score):
            summary["star_rating"] = 0
        elif score > np.log(1 + 0.85):
            summary["star_rating"] = 5
        elif score > np.log(1 + 0.50):
            summary["star_rating"] = 4
        elif score > np.log(1 + 0.15):
            summary["star_rating"] = 3
        elif score > 0:
            summary["star_rating"] = 2
        else:
            summary["star_rating"] = 1

    # compute the publication bias
    index = df.is_outlier == 0
    residual = df.ln_rr.values[index] - df.signal.values[index] * beta_info[0]
    residual_sd = np.sqrt(
        df.ln_rr_se.values[index] ** 2
        + df.signal.values[index] ** 2 * gamma_info[0]
    )
    weighted_residual = residual / residual_sd
    r_mean = weighted_residual.mean()
    r_sd = 1 / np.sqrt(weighted_residual.size)
    pval = 1 - norm.cdf(np.abs(r_mean / r_sd))
    summary["pub_bias"] = int(pval < 0.05)
    summary["pub_bias_pval"] = float(pval)

    return summary


def get_draws(
    settings: dict,
    summary: dict,
    df_coef: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Create effect draws for the pipeline.

    Parameters
    ----------
    settings
        Settings for complete the summary.
    summary
        Summary of the models.
    df_coef
        Data frame containing fitted beta coefficients

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer draw files.

    """

    beta_info = summary["beta"]
    gamma_info = summary["gamma"]
    inner_beta_sd = beta_info[1]
    outer_beta_sd = np.sqrt(
        beta_info[1] ** 2 + gamma_info[0] + 2 * gamma_info[1]
    )
    inner_beta_samples = np.random.normal(
        loc=beta_info[0],
        scale=inner_beta_sd,
        size=settings["draws"]["num_draws"],
    )
    outer_beta_samples = np.random.normal(
        loc=beta_info[0],
        scale=outer_beta_sd,
        size=settings["draws"]["num_draws"],
    )
    inner_draws = np.outer(df_coef.coef, inner_beta_samples)
    outer_draws = np.outer(df_coef.coef, outer_beta_samples)
    df_inner_draws = pd.DataFrame(
        np.hstack([df_coef["cat"].to_numpy().reshape(-1, 1), inner_draws]),
        columns=["risk_cat"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )
    df_outer_draws = pd.DataFrame(
        np.hstack([df_coef["cat"].to_numpy().reshape(-1, 1), outer_draws]),
        columns=["risk_cat"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )

    return df_inner_draws, df_outer_draws


def get_quantiles(
    settings: dict,
    summary: dict,
    df_coef: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Create effect quantiles for the pipeline.

    Parameters
    ----------
    settings
        The settings for complete the summary.
    summary
        The completed summary file.
    df_coef
        Data frame containing fitted beta coefficients

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer quantile files.

    """

    beta_info = summary["beta"]
    gamma_info = summary["gamma"]
    inner_beta_sd = beta_info[1]
    outer_beta_sd = np.sqrt(
        beta_info[1] ** 2 + gamma_info[0] + 2 * gamma_info[1]
    )
    # get quantiles
    cats = df_coef["cat"].to_numpy().reshape(-1, 1)
    coefs = df_coef["coef"].to_numpy()
    quantiles = np.asarray(settings["draws"]["quantiles"])
    signal_sign_index = np.zeros(coefs.size, dtype=int)
    signal_sign_index[coefs < 0] = 1
    inner_beta_quantiles = [
        norm.ppf(quantiles, loc=summary["beta"][0], scale=inner_beta_sd),
        norm.ppf(1 - quantiles, loc=summary["beta"][0], scale=inner_beta_sd),
    ]
    inner_beta_quantiles = np.vstack(inner_beta_quantiles).T
    outer_beta_quantiles = [
        norm.ppf(quantiles, loc=summary["beta"][0], scale=outer_beta_sd),
        norm.ppf(1 - quantiles, loc=summary["beta"][0], scale=outer_beta_sd),
    ]
    outer_beta_quantiles = np.vstack(outer_beta_quantiles).T
    inner_quantiles = [
        inner_beta_quantiles[i][signal_sign_index] * coefs
        for i in range(len(quantiles))
    ]
    inner_quantiles = np.vstack(inner_quantiles).T
    outer_quantiles = [
        outer_beta_quantiles[i][signal_sign_index] * coefs
        for i in range(len(quantiles))
    ]
    outer_quantiles = np.vstack(outer_quantiles).T

    df_inner_quantiles = pd.DataFrame(
        np.hstack([cats, inner_quantiles]),
        columns=["risk_cat"] + list(map(str, quantiles)),
    )
    df_outer_quantiles = pd.DataFrame(
        np.hstack([cats, outer_quantiles]),
        columns=["risk_cat"] + list(map(str, quantiles)),
    )

    return df_inner_quantiles, df_outer_quantiles


def plot_linear_model(
    name: str,
    summary: dict,
    df: DataFrame,
    df_coef: DataFrame,
    signal_model: MRBRT,
    linear_model: MRBRT,
    show_ref: bool = True,
) -> Figure:
    """Plot the linear model

    Parameters
    ----------
    name
        Name of the pair
    summary
        Completed summary file.
    df
        Data frame contains the training data.
    df_coef
        Data frame containing the fitted beta coefficients
    signal_model
        Fitted signal model for risk curve.
    linear_model
        Fitted linear model for risk curve.
    show_ref
        Whether to show the reference line. Default is `True`.

    Returns
    -------
    Figure
        The figure object for linear model.

    """
    offset = 0.05
    # create fig obj
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # plot data
    _plot_data(
        name,
        summary,
        df,
        df_coef,
        ax[0],
        signal_model,
        linear_model,
        show_ref=show_ref,
    )
    # plot beta coefficients and uncertainty
    beta = summary["beta"]
    gamma = summary["gamma"]
    inner_beta_sd = beta[1]
    outer_beta_sd = np.sqrt(beta[1] ** 2 + gamma[0] + 2 * gamma[1])

    pred = df_coef.coef * beta[0]
    df_coef["outer_ui_low"] = (beta[0] - 1.96 * outer_beta_sd) * df_coef.coef
    df_coef["inner_ui_low"] = (beta[0] - 1.96 * inner_beta_sd) * df_coef.coef
    df_coef["inner_ui_hi"] = (beta[0] + 1.96 * inner_beta_sd) * df_coef.coef
    df_coef["outer_ui_hi"] = (beta[0] + 1.96 * outer_beta_sd) * df_coef.coef

    if summary["normalize_to_tmrel"]:
        index = np.argmin(pred)
        pred -= pred[index]
        df_coef["outer_ui_low"] -= df_coef.outer_ui_low[None, index]
        df_coef["inner_ui_low"] -= df_coef.inner_ui_low[None, index]
        df_coef["inner_ui_hi"] -= df_coef.inner_ui_hi[None, index]
        df_coef["outer_ui_hi"] -= df_coef.outer_ui_hi[None, index]

    log_bprf = pred * (1.0 - 1.645 * outer_beta_sd / beta[0])

    x_start = df_coef["x_start"] + offset
    x_end = df_coef["x_end"] - offset

    # Plot coefficients
    ax[0].plot([x_start, x_end], [pred] * 2, color="#008080")
    # Plot BPRF
    ax[0].plot([x_start, x_end], [log_bprf] * 2, color="red")
    # Fill between UIs
    for i, row in df_coef.iterrows():
        ax[0].fill_between(
            [row["x_start"] + offset, row["x_end"] - offset],
            row["inner_ui_low"],
            row["inner_ui_hi"],
            color="gray",
            alpha=0.2,
        )
    for i, row in df_coef.iterrows():
        ax[0].fill_between(
            [row["x_start"] + offset, row["x_end"] - offset],
            row["outer_ui_low"],
            row["outer_ui_hi"],
            color="gray",
            alpha=0.2,
        )

    # plot funnel
    _plot_funnel(summary, df, ax[1])

    return fig


def _plot_data(
    name: str,
    summary: dict,
    df: DataFrame,
    df_coef: DataFrame,
    ax: Axes,
    signal_model: MRBRT = None,
    linear_model: Optional[MRBRT] = None,
    show_ref: bool = True,
) -> Axes:
    """Plot data points

    Parameters
    ----------
    name
        Name of the pair.
    summary
        The summary of the signal model.
    df
        Data frame contains training data.
    df_coef
        Data frame containing the fitted beta coefficients
    ax
        Axes of the figure. Usually corresponding to one panel of a figure.
    signal_model
        Fitted signal model for risk curve.
    linear_model
        Fitted linear model for risk curve. Default is `None`. When it is `None`
        the points are plotted reference to original signal model. When linear
        model is provided, the points are plotted reference to the linear model
        risk curve.
    show_ref
        Whether to show the reference line. Default is `True`.

    Returns
    -------
    Axes
        Return the axes back for further plotting.

    """
    np.random.seed(0)

    # Merge coefficient for each category into dataframe: beta coefficient for reference category
    # Then add x midpoint corresponding to reference and alternative categories
    df = (
        df.merge(
            df_coef[["cat", "coef"]].rename(columns={"cat": "ref_risk_cat"}),
            on="ref_risk_cat",
            how="left",
        )
        .merge(
            df_coef[["cat", "x_mid"]].rename(
                columns={"cat": "alt_risk_cat", "x_mid": "alt_cat_mid"}
            ),
            on="alt_risk_cat",
            how="left",
        )
        .merge(
            df_coef[["cat", "x_mid"]].rename(
                columns={"cat": "ref_risk_cat", "x_mid": "ref_cat_mid"}
            ),
            on="ref_risk_cat",
            how="left",
        )
    )

    ref_obs = df.coef
    if linear_model is not None:
        ref_obs *= linear_model.beta_soln[0]
    alt_obs = df.ln_rr + ref_obs

    # shift data position normalize to tmrel
    if summary["normalize_to_tmrel"]:
        beta_min = df_coef.coef.min()
        if linear_model is not None:
            beta_min *= linear_model.beta_soln[0]
        ref_obs -= beta_min
        alt_obs -= beta_min

    # Add a little jitter
    alt_cat_mid_jitter = df.alt_cat_mid + np.random.uniform(
        -0.2, 0.2, df.shape[0]
    )

    # plot data points
    index = df.is_outlier == 1
    ax.scatter(
        alt_cat_mid_jitter,
        alt_obs,
        s=5 / df["ln_rr_se"].values,
        color="#008080",
        alpha=0.5,
        edgecolor="none",
    )
    ax.scatter(
        alt_cat_mid_jitter[index],
        alt_obs[index],
        s=5 / df.ln_rr_se.values[index],
        color="red",
        alpha=0.5,
        marker="x",
    )
    if show_ref:
        for x_0, y_0, x_1, y_1 in zip(
            alt_cat_mid_jitter, alt_obs, df["ref_cat_mid"], ref_obs
        ):
            ax.plot(
                [x_0, x_1],
                [y_0, y_1],
                color="#008080",
                linewidth=0.5,
                alpha=0.5,
            )

    # plot support lines
    ax.axhline(0.0, linewidth=1, linestyle="-", color="gray")

    # add title and label
    rei, _ = tuple(name.split("-"))
    ax.set_title(name.replace("-", " / "), loc="left")
    ax.set_xticks(df_coef["x_mid"])
    ax.set_xticklabels(df_coef["cat"])
    ax.set_ylabel("ln relative risk")

    return ax


def _plot_funnel(summary: dict, df: DataFrame, ax: Axes) -> Axes:
    """Plot the funnel plot

    Parameters
    ----------
    summary
        Complete summary file.
    df
        Data frame that contains training data.
    ax
        Axes of the figure. Usually corresponding to one panel of a figure.

    Returns
    -------
    Axes
        Return the axes back for further plotting.

    """

    # add residual information
    beta, gamma = summary["beta"], summary["gamma"]
    residual = df.ln_rr.values - df.signal.values * beta[0]
    residual_sd = np.sqrt(
        df.ln_rr_se.values**2 + df.signal.values**2 * gamma[0]
    )

    index = df.is_outlier == 1
    sd_max = residual_sd.max() * 1.1
    ax.set_ylim(sd_max, 0.0)
    # plot data
    ax.scatter(
        residual, residual_sd, color="#008080", alpha=0.5, edgecolor="none"
    )
    ax.scatter(
        residual[index], residual_sd[index], color="red", alpha=0.5, marker="x"
    )
    # plot funnel
    ax.fill_betweenx(
        [0.0, sd_max],
        [0.0, -1.96 * sd_max],
        [0.0, 1.96 * sd_max],
        color="gray",
        alpha=0.2,
    )
    ax.plot([0.0, -1.96 * sd_max], [0.0, sd_max], linewidth=1, color="gray")
    ax.plot([0.0, 1.96 * sd_max], [0.0, sd_max], linewidth=1, color="gray")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    # set title and labels
    ax.set_xlabel("residual")
    ax.set_ylabel("residual sd")

    return ax

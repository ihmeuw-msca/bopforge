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


def get_cov_finder(
    settings: dict, cov_finder_linear_model: MRBRT, df: DataFrame
) -> CovFinder:
    """Create the instance of CovFinder class.

    Parameters
    ----------
    settings
        Settings for bias covariate selection.
    cov_finder_linear_model
        Fitted cov finder linear model (signal model for categorical risks)
    df
        Dataframe containing training data with column indicating outliers


    Returns
    -------
    CovFinder
        The instance of the CovFinder class.

    """
    data_signal = cov_finder_linear_model.data
    cats = cov_finder_linear_model.cov_models[0].cats
    ref_cat = cov_finder_linear_model.cov_models[0].ref_cat
    alt_cats = [cat for cat in cats if cat != ref_cat]

    # Create design matrix
    for cat in cats:
        df[cat] = 0
    for i, row in df.iterrows():
        df.at[i, row["ref_risk_cat"]] = -1  # Assign -1 for ref_risk_cat
        df.at[i, row["alt_risk_cat"]] = 1  # Assign 1 for alt_risk_cat

    df = pd.concat(
        [
            df,
            pd.DataFrame(data_signal.covs["intercept"], columns=["intercept"]),
        ],
        axis=1,
    )
    df = df[df.is_outlier == 0].copy()
    col_covs = [col for col in df.columns if col.startswith("cov_")]
    data = MRData()
    data.load_df(
        df,
        col_obs="ln_rr",
        col_obs_se="ln_rr_se",
        col_covs=col_covs + alt_cats + ["ref_risk_cat", "alt_risk_cat"],
        col_study_id="study_id",
        col_data_id="seq",
    )

    beta_info = get_beta_info(cov_finder_linear_model, cov_name=None)
    index = list(cats).index(ref_cat)
    beta_info = tuple(np.delete(arr, index) for arr in beta_info)

    # covariate selection
    pre_selected_covs = settings["cov_finder"]["pre_selected_covs"]
    if isinstance(pre_selected_covs, str):
        pre_selected_covs = [pre_selected_covs]
    if "intercept" not in pre_selected_covs:
        pre_selected_covs.append("intercept")
    pre_selected_covs += alt_cats
    settings["cov_finder"]["pre_selected_covs"] = pre_selected_covs
    candidate_covs = [
        cov_name
        for cov_name in data.covs.keys()
        if cov_name not in pre_selected_covs + ["ref_risk_cat", "alt_risk_cat"]
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
        beta_gprior_std=0.1 * np.mean(beta_info[1]),
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
    cats = cov_finder_linear_model.cov_models[0].cats
    ref_cat = cov_finder_linear_model.cov_models[0].ref_cat
    alt_cats = [cat for cat in cats if cat != ref_cat]
    index = list(cats).index(ref_cat)
    beta_info = tuple(np.delete(arr, index) for arr in beta_info)

    excluded_covs = {"intercept", "alt_risk_cat", "ref_risk_cat"} | set(
        alt_cats
    )
    selected_covs = [
        cov_name
        for cov_name in cov_finder.selected_covs
        if cov_name not in excluded_covs
    ]

    # save results
    cov_finder_result = {
        "beta_sd": float(np.mean(beta_info[1]) * 0.1),
        "selected_covs": selected_covs,
    }

    return cov_finder_result


def get_coefs(
    df: DataFrame,
    settings: dict,
    model: MRBRT,
    ref_cat_input: Optional[str] = None,
) -> tuple[DataFrame]:
    """Get beta coefficients for the categories.

    Parameters
    ----------
    df
        Data frame contains the training data.
    settings
        The settings for category order
    model
        Fitted model for the categories, either signal or linear model.
    ref_cat_input
        Reference category. Optional, will be inferred from settings or data
        if not provided.

    Returns
    -------
    tuple[DataFrame]
        Dataframe of beta coefficients and ranges to plot each category

    """
    # extract categories, betas
    df_coef = pd.DataFrame(
        dict(cat=model.cov_models[0].cats, coef=model.beta_soln)
    )
    if ref_cat_input is not None:
        ref_cat = ref_cat_input
    else:
        # Assign reference category from settings or by most common category
        ref_cat = settings["fit_signal_model"]["cat_cov_model"]["ref_cat"]
        if ref_cat:
            ref_cat = ref_cat
        else:
            unique_cats, counts = np.unique(
                np.hstack([df.ref_risk_cat, df.alt_risk_cat]),
                return_counts=True,
            )
            ref_cat = unique_cats[counts.argmax()]
    # Order the categories
    cat_order = settings["figure"]["cat_order"]
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
        Data frame containing the fitted beta coefficients for the signal model
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
            ref_cat=settings["fit_signal_model"]["cat_cov_model"]["ref_cat"],
            prior_order=settings["fit_signal_model"]["cat_cov_model"][
                "prior_order"
            ],
            use_re=True,
        )
    ]
    for cov_name in cov_finder_result["selected_covs"]:
        cov_models.append(
            LinearCovModel(
                cov_name,
                prior_beta_gaussian=[0.0, cov_finder_result["beta_sd"]],
            )
        )
    model = MRBRT(data, cov_models)
    return model


def get_linear_model_summary(
    settings: dict,
    summary: dict,
    df: DataFrame,
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
    cats = linear_model.cov_models[0].cats
    index = list(cats).index(ref_cat)
    alt_cats = [cat for cat in cats if cat != ref_cat]

    # solution of the final model
    beta_info = get_beta_info(linear_model, cov_name=None)
    gamma_info = get_gamma_info(linear_model)
    summary["beta"] = dict(zip(cats, beta_info[0].tolist()))
    summary["beta_sd"] = dict(zip(cats, beta_info[1].tolist()))
    summary["gamma"] = [float(gamma_info[0]), float(gamma_info[1])]

    # compute the score and add star rating
    # Subset to only alternative categories
    beta_alt_cats = tuple(np.delete(arr, index) for arr in beta_info)
    beta_sd = np.sqrt(beta_alt_cats[1] ** 2 + gamma_info[0] + 2 * gamma_info[1])
    pred = beta_alt_cats[0]
    inner_ui = np.vstack(
        [
            beta_alt_cats[0] - 1.96 * beta_alt_cats[1],
            beta_alt_cats[0] + 1.96 * beta_alt_cats[1],
        ]
    )
    sign = np.sign(pred)
    burden_of_proof = beta_alt_cats[0] - sign * 1.645 * beta_sd

    if settings["score"]["normalize_to_tmrel"]:
        index = np.argmin(pred)
        pred -= pred[index]
        burden_of_proof -= burden_of_proof[None, index]
        inner_ui -= inner_ui[:, None, index]

    signed_bprf = sign * burden_of_proof
    # Number of alternative categories
    n = len(alt_cats)
    # Assign dichotomous score for each alternative category
    score_by_category = np.zeros(n)
    product = np.prod(inner_ui, axis=0)
    score_by_category[product < 0] = float("nan")
    score_by_category[product >= 0] = 0.5 * signed_bprf[product >= 0]
    summary["score_by_category"] = dict(
        zip(alt_cats, score_by_category.tolist())
    )
    # Index with largest signed coefficient
    max_idx = np.argmax(signed_bprf)
    if np.any(product < 0):
        summary["score"] = float("nan")
        summary["star_rating"] = 0
    else:
        score = float(
            (1 / n) * (np.sum(signed_bprf) - 0.5 * signed_bprf[max_idx])
        )
        summary["score"] = score
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
    beta_dict = dict(zip(cats, beta_info[0]))
    residual = df["ln_rr"].values[index] - (
        df["alt_risk_cat"].map(beta_dict).values[index]
        - df["ref_risk_cat"].map(beta_dict).values[index]
    )
    residual_sd = np.sqrt(df.ln_rr_se.values[index] ** 2 + gamma_info[0])
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
) -> tuple[DataFrame, DataFrame]:
    """Create effect draws for the pipeline.

    Parameters
    ----------
    settings
        Settings for complete the summary.
    summary
        Summary of the models.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer draw files.

    """

    beta_info = summary["beta"]
    beta_sd_info = summary["beta_sd"]
    gamma_info = summary["gamma"]
    inner_beta_sd = np.array(list(beta_sd_info.values()))
    outer_beta_sd = np.sqrt(
        np.array(list(beta_sd_info.values())) ** 2
        + gamma_info[0]
        + 2 * gamma_info[1]
    )
    inner_beta_samples = np.random.normal(
        loc=np.array(list(beta_info.values()))[:, None],
        scale=inner_beta_sd[:, None],
        size=(len(beta_info), settings["draws"]["num_draws"]),
    )
    outer_beta_samples = np.random.normal(
        loc=np.array(list(beta_info.values()))[:, None],
        scale=outer_beta_sd[:, None],
        size=(len(beta_info), settings["draws"]["num_draws"]),
    )
    df_inner_draws = pd.DataFrame(
        np.hstack(
            [np.array(list(beta_info.keys()))[:, None], inner_beta_samples]
        ),
        columns=["risk_cat"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )
    df_outer_draws = pd.DataFrame(
        np.hstack(
            [np.array(list(beta_info.keys()))[:, None], outer_beta_samples]
        ),
        columns=["risk_cat"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )

    return df_inner_draws, df_outer_draws


def get_quantiles(
    settings: dict,
    summary: dict,
) -> tuple[DataFrame, DataFrame]:
    """Create effect quantiles for the pipeline.

    Parameters
    ----------
    settings
        The settings for complete the summary.
    summary
        The completed summary file.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer quantile files.

    """

    beta_info = summary["beta"]
    beta_sd_info = summary["beta_sd"]
    gamma_info = summary["gamma"]
    inner_beta_sd = np.array(list(beta_sd_info.values()))
    outer_beta_sd = np.sqrt(
        np.array(list(beta_sd_info.values())) ** 2
        + gamma_info[0]
        + 2 * gamma_info[1]
    )
    # get quantiles
    cats = np.array(list(beta_info.keys()))[:, None]
    coefs = np.array(list(beta_info.values()))
    quantiles = np.asarray(settings["draws"]["quantiles"])
    inner_beta_quantiles = [
        norm.ppf(quantiles, loc=coefs[:, None], scale=inner_beta_sd[:, None]),
    ]
    inner_beta_quantiles = np.vstack(inner_beta_quantiles)
    outer_beta_quantiles = [
        norm.ppf(quantiles, loc=coefs[:, None], scale=outer_beta_sd[:, None]),
    ]
    outer_beta_quantiles = np.vstack(outer_beta_quantiles)
    df_inner_quantiles = pd.DataFrame(
        np.hstack([cats, inner_beta_quantiles]),
        columns=["risk_cat"] + list(map(str, quantiles)),
    )
    df_outer_quantiles = pd.DataFrame(
        np.hstack([cats, outer_beta_quantiles]),
        columns=["risk_cat"] + list(map(str, quantiles)),
    )

    # signal_sign_index = np.zeros(coefs.size, dtype=int)
    # signal_sign_index[coefs < 0] = 1
    # inner_beta_quantiles = [
    #     norm.ppf(quantiles, loc=summary["beta"][0], scale=inner_beta_sd),
    #     norm.ppf(1 - quantiles, loc=summary["beta"][0], scale=inner_beta_sd),
    # ]
    # inner_beta_quantiles = np.vstack(inner_beta_quantiles).T
    # outer_beta_quantiles = [
    #     norm.ppf(quantiles, loc=summary["beta"][0], scale=outer_beta_sd),
    #     norm.ppf(1 - quantiles, loc=summary["beta"][0], scale=outer_beta_sd),
    # ]
    # outer_beta_quantiles = np.vstack(outer_beta_quantiles).T
    # inner_quantiles = [
    #     inner_beta_quantiles[i][signal_sign_index] * coefs
    #     for i in range(len(quantiles))
    # ]
    # inner_quantiles = np.vstack(inner_quantiles).T
    # outer_quantiles = [
    #     outer_beta_quantiles[i][signal_sign_index] * coefs
    #     for i in range(len(quantiles))
    # ]
    # outer_quantiles = np.vstack(outer_quantiles).T

    return df_inner_quantiles, df_outer_quantiles


def plot_linear_model(
    name: str,
    summary: dict,
    df: DataFrame,
    df_coef: DataFrame,
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
        show_ref=show_ref,
    )
    # plot beta coefficients and uncertainty
    gamma = summary["gamma"]
    inner_beta_sd = summary["beta_sd"]
    df_coef["inner_beta_sd"] = df_coef["cat"].map(inner_beta_sd).values
    df_coef["outer_beta_sd"] = np.sqrt(
        df_coef["cat"].map(inner_beta_sd).values ** 2 + gamma[0] + 2 * gamma[1]
    )

    pred = df_coef.coef
    df_coef["outer_ui_low"] = df_coef.coef - 1.96 * df_coef.outer_beta_sd
    df_coef["inner_ui_low"] = df_coef.coef - 1.96 * df_coef.inner_beta_sd
    df_coef["inner_ui_hi"] = df_coef.coef + 1.96 * df_coef.inner_beta_sd
    df_coef["outer_ui_hi"] = df_coef.coef + 1.96 * df_coef.outer_beta_sd
    # Reset reference UIs to be ~ 0
    ref_idx = df_coef.loc[df_coef["cat"] == summary["ref_cat"]].index[0]
    ui_cols = [col for col in df_coef.columns if "_ui_" in col]
    df_coef.loc[ref_idx, ui_cols] = df_coef.loc[ref_idx, "coef"]

    if summary["normalize_to_tmrel"]:
        index = np.argmin(pred)
        pred -= pred[index]
        df_coef["outer_ui_low"] -= df_coef.outer_ui_low[None, index]
        df_coef["inner_ui_low"] -= df_coef.inner_ui_low[None, index]
        df_coef["inner_ui_hi"] -= df_coef.inner_ui_hi[None, index]
        df_coef["outer_ui_hi"] -= df_coef.outer_ui_hi[None, index]

    sign = np.sign(pred)
    log_bprf = pred * (
        1.0 - sign * 1.645 * df_coef["outer_beta_sd"] / df_coef["coef"]
    )
    # reset reference index
    log_bprf[ref_idx] = df_coef.loc[ref_idx, "coef"]

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
        Data frame containing the fitted beta coefficients,
        either for signal or linear model
    ax
        Axes of the figure. Usually corresponding to one panel of a figure.
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
    alt_obs = df.ln_rr + ref_obs

    # shift data position normalize to tmrel
    if summary["normalize_to_tmrel"]:
        beta_min = df_coef.coef.min()
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
    residual = df.ln_rr.values - (
        df.alt_risk_cat.map(beta).values - df.ref_risk_cat.map(beta).values
    )
    residual_sd = np.sqrt(df.ln_rr_se.values**2 + gamma[0])

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

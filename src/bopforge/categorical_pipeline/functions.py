from itertools import combinations
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes, Figure
from mrtool import MRBRT, CovFinder, LinearCatCovModel, LinearCovModel, MRData
from pandas import DataFrame
from scipy.stats import norm

from bopforge.utils import get_beta_info


def covariate_preprocessing(
    df: DataFrame,
    settings: dict,
) -> tuple[DataFrame, dict]:
    """Parse and pre-process the covariates,
    including dropping covariates with all or all-but-one of the same value
    and performing validation checks

    Parameters
    ----------
    df
        Dataframe with original dataset
    settings
        Settings for the bias covariate section

    Returns
    -------
    tuple[DataFrame, dict]
        Updated dataframe with problematic covariate columns dropped
        and updated settings file with problematic covariates removed
    """
    pre_selected_cov_settings = settings["cov_finder"]
    cov_settings = settings["cov_type"]

    # Parse types of covariates
    bias_covs = set(cov_settings["bias_covs"])
    interacted_covs = set(cov_settings["interacted_covs"])
    non_interacted_covs = set(cov_settings["non_interacted_covs"])
    pre_selected_covs = set(pre_selected_cov_settings["pre_selected_covs"])

    # Validate: each set should be distinct
    cov_sets = {
        "bias": bias_covs,
        "interacted": interacted_covs,
        "non_interacted": non_interacted_covs,
    }
    for (name1, covs1), (name2, covs2) in combinations(cov_sets.items(), 2):
        overlap = covs1 & covs2
        if overlap:
            raise ValueError(
                f"Covariates defined in both '{name1}' and '{name2}': {overlap}"
            )

    # Validate: all covs in settings are present in data
    all_covs = bias_covs | interacted_covs | non_interacted_covs
    covs_missing_from_df = all_covs - set(df.columns)
    if covs_missing_from_df:
        raise ValueError(
            f"The following covariates are specified in settings but not found in the dataframe: {covs_missing_from_df}"
        )

    # Validate: all bias covariates should be binary
    for cov in bias_covs:
        unique_vals = df[cov].unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Bias covariate '{cov}' is not binary")

    # Validate: any pre-selected covs are present in the list of bias covariates
    assert pre_selected_covs <= bias_covs, (
        f"pre_selected_covs must be a subset of bias_covs, but had additional non-bias covariates: "
        f"{pre_selected_covs - bias_covs}"
    )

    # Identify covariates to be removed: all or all-but-one of the same value
    covs_to_remove = set()
    for col in all_covs:
        counts = df[col].value_counts()
        if counts.iloc[0] >= len(df[col]) - 1:
            covs_to_remove.add(col)
    # Drop from dataframe
    df.drop(columns=covs_to_remove, inplace=True)
    # Drop from covariate lists
    bias_covs -= covs_to_remove
    interacted_covs -= covs_to_remove
    non_interacted_covs -= covs_to_remove
    pre_selected_covs -= covs_to_remove
    # Update the cov_settings
    cov_settings["bias_covs"] = list(bias_covs)
    cov_settings["interacted_covs"] = list(interacted_covs)
    cov_settings["non_interacted_covs"] = list(non_interacted_covs)
    pre_selected_cov_settings["pre_selected_covs"] = list(pre_selected_covs)
    # Save the updated settings back to the YAML file
    settings["cov_finder"] = pre_selected_cov_settings
    settings["cov_type"] = cov_settings

    return df, settings


def covariate_design_mat(
    df: DataFrame,
    settings: dict,
) -> DataFrame:
    """Create design matrices for interacted model covariates

    Parameters
    ----------
    df
        Dataframe with original dataset
    settings
        Settings for the bias covariate section

    Returns
    -------
    DataFrame
        Dataframe with interacted model covariates design matrices appended
    """
    interacted_covs = settings["cov_type"]["interacted_covs"]
    cats = np.unique(df[["ref_risk_cat", "alt_risk_cat"]].to_numpy().ravel())
    alt_cats_mat = pd.get_dummies(df["alt_risk_cat"], drop_first=False).astype(
        float
    )
    ref_cats_mat = pd.get_dummies(df["ref_risk_cat"], drop_first=False).astype(
        float
    )
    for cat in cats:
        if cat not in alt_cats_mat:
            alt_cats_mat[cat] = 0.0
        if cat not in ref_cats_mat:
            ref_cats_mat[cat] = 0.0
    alt_cats_mat = alt_cats_mat[cats]
    ref_cats_mat = ref_cats_mat[cats]
    design = alt_cats_mat - ref_cats_mat
    model_covs = list(interacted_covs)
    design_matrices = {}
    for cov_name in model_covs:
        cov_name_key = f"{cov_name}_design"
        cov_design = design.copy()
        cat_name_temp = [
            f"interacted_{cov_name}_{col}" for col in cov_design.columns
        ]
        cov_design.columns = cat_name_temp
        cov_design[:] = cov_design.to_numpy() * df[cov_name].to_numpy()[:, None]
        cov_design[cov_design == -0.0] = 0.0
        design_matrices[cov_name_key] = cov_design

    # Append model covariate design matrices to dataframe
    for cov_name, cov_design in design_matrices.items():
        df = pd.concat([df, cov_design], axis=1)

    return df


def get_signal_model(settings: dict, df: DataFrame, summary: dict) -> MRBRT:
    """Create signal model for outliers identification and covariate selection
    step.

    Parameters
    ----------
    settings
        Dictionary containing settings for covariates and fitting the signal model
    df
        Data frame contains the training data.
    summary
        Dictionary containing initial summary outputs

    Returns
    -------
    MRBRT
        Signal model to access the strength of the prior on the bias-covariate.

    """
    ref_cat = summary["ref_cat"]
    signal_model_settings = settings["fit_signal_model"]
    cov_settings = settings["select_bias_covs"]["cov_type"]

    # Load in model covariates and candidate bias covariates
    # interacted needs the design matrix columns
    interacted_covs = [
        col for col in df.columns if col.startswith("interacted_")
    ]
    non_interacted_covs = cov_settings["non_interacted_covs"]
    bias_covs = cov_settings["bias_covs"]

    data = MRData()
    data.load_df(
        df,
        col_obs="ln_rr",
        col_obs_se="ln_rr_se",
        col_covs=bias_covs
        + ["ref_risk_cat", "alt_risk_cat"]
        + interacted_covs
        + non_interacted_covs,
        col_study_id="study_id",
        col_data_id="seq",
    )
    cov_models = [
        LinearCatCovModel(
            alt_cov="alt_risk_cat",
            ref_cov="ref_risk_cat",
            ref_cat=signal_model_settings["cat_cov_model"]["ref_cat"],
            prior_order=signal_model_settings["cat_cov_model"]["prior_order"],
            use_re=False,
        )
    ]
    # Add interacted covariates
    for cov_name in interacted_covs:
        cov_models.append(
            LinearCovModel(
                alt_cov=cov_name,
                use_re=False,
            )
            if ref_cat not in cov_name
            else LinearCovModel(
                alt_cov=cov_name, use_re=False, prior_beta_uniform=[0.0, 0.0]
            )
        )
    # Add non-interacted covariates
    for cov_name in non_interacted_covs:
        cov_models.append(
            LinearCovModel(
                cov_name,
                use_re=False,
            )
        )

    signal_model = MRBRT(
        data, cov_models, **signal_model_settings["signal_model"]
    )

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


def get_cov_finder(
    settings: dict, cov_finder_linear_model: MRBRT, df: DataFrame, summary: dict
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
    summary
        Summary of signal model, including list of candidate bias covariates


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
        df[cat] = 0.0
    for i, row in df.iterrows():
        df.at[i, row["ref_risk_cat"]] = -1.0  # Assign -1 for ref_risk_cat
        df.at[i, row["alt_risk_cat"]] = 1.0  # Assign 1 for alt_risk_cat

    df = pd.concat(
        [
            df,
            pd.DataFrame(data_signal.covs["intercept"], columns=["intercept"]),
        ],
        axis=1,
    )
    df = df[df.is_outlier == 0].copy()
    col_covs = settings["cov_type"]["bias_covs"]
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


def get_cat_coefs(
    settings: dict,
    model: MRBRT,
    type: str,
    ref_cat_input: str,
) -> tuple[DataFrame]:
    """Get beta and gamma coefficients for the categories.

    Parameters
    ----------
    settings
        The settings for category order
    model
        Fitted model for the categories, linear or signal
    type
        String specifying whether model is signal or linear model
    ref_cat_input
        Reference category. Optional, will be inferred from settings or data
        if not provided.

    Returns
    -------
    tuple[DataFrame]
        Dataframe of beta, beta_sd, gamma, gamma_sd for each category and ranges to plot each category

    """

    lt = model.lt

    # Extract betas for categories and covariates, ensuring correct matching
    cov_names = []
    for cov_model in model.cov_models:
        if isinstance(cov_model, LinearCovModel):
            cov_name = cov_model.alt_cov[0]
            if cov_model.num_x_vars == 1:
                cov_names.append(cov_name)
            else:
                cov_names.extend(
                    [f"{cov_name}_{i}" for i in range(cov_model.num_x_vars)]
                )
        elif isinstance(cov_model, LinearCatCovModel):
            cov_names.extend("cat_" + cov_model.cats.astype(str))
        else:
            raise TypeError("Unknown cov_model type")

    beta = model.beta_soln

    beta_info = pd.DataFrame(
        {
            "cov_name": cov_names,
            "beta": beta,
            # "beta_sd": beta_sd,
        }
    )
    # Subset to betas for categories only
    beta_cats = beta_info[beta_info["cov_name"].str.startswith("cat_")].copy()
    beta_cats["cat"] = beta_cats["cov_name"].str.removeprefix("cat_")

    # Extract gamma, calculate gamma_sd, and merge with beta dataframe for linear model only
    if type == "linear":
        lt = model.lt
        gamma = model.gamma_soln[0]
        gamma_fisher = lt.get_gamma_fisher(gamma)
        gamma_sd = 1.0 / np.sqrt(gamma_fisher[0, 0])
        gamma_cats = pd.DataFrame(
            {
                "cat": model.cov_models[
                    model.cov_model_names == "alt_risk_cat"
                ].cats,
                "gamma": np.repeat(
                    gamma,
                    len(
                        model.cov_models[
                            model.cov_model_names == "alt_risk_cat"
                        ].cats
                    ),
                ),
                "gamma_sd": np.repeat(
                    gamma_sd,
                    len(
                        model.cov_models[
                            model.cov_model_names == "alt_risk_cat"
                        ].cats
                    ),
                ),
            }
        )
        cat_coefs = beta_cats.merge(gamma_cats, on="cat", how="left")
    else:
        cat_coefs = beta_cats

    return cat_coefs


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
    ref_cat = settings["fit_signal_model"]["cat_cov_model"]["ref_cat"]
    col_covs = cov_finder_result["selected_covs"]
    interacted_covs = [
        col for col in df.columns if col.startswith("interacted_")
    ]
    non_interacted_covs = settings["select_bias_covs"]["cov_type"][
        "non_interacted_covs"
    ]
    data = MRData()
    data.load_df(
        df,
        col_obs="ln_rr",
        col_obs_se="ln_rr_se",
        col_covs=col_covs
        + ["ref_risk_cat", "alt_risk_cat"]
        + interacted_covs
        + non_interacted_covs,
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
            use_re_intercept=True,
            # use_re_intercept = false => each category has own random effect
            # In settings, cat_specific_gamma = True => use_re_intercept = False
            # so invert the boolean here to appropriately link things
            # use_re_intercept=not settings["complete_summary"]["cat_gamma"][
            #     "cat_specific_gamma"
            # ],
            prior_gamma_uniform=[0.0, np.inf],
        )
    ]
    # Add interacted covariates
    for cov_name in interacted_covs:
        cov_models.append(
            LinearCovModel(
                alt_cov=cov_name,
                use_re=False,
            )
            if ref_cat not in cov_name
            else LinearCovModel(
                alt_cov=cov_name, use_re=False, prior_beta_uniform=[0.0, 0.0]
            )
        )
    # Add non-interacted covariates
    for cov_name in non_interacted_covs:
        cov_models.append(
            LinearCovModel(
                cov_name,
                use_re=False,
            )
        )
    # Add selected bias covariates
    for cov_name in cov_finder_result["selected_covs"]:
        cov_models.append(
            LinearCovModel(
                cov_name,
                prior_beta_gaussian=[0.0, cov_finder_result["beta_sd"]],
            )
        )

    model = MRBRT(data, cov_models)
    return model


def get_pair_info(
    settings: dict,
    summary: dict,
    cat_coefs: DataFrame,
    linear_model: MRBRT,
) -> tuple[DataFrame]:
    """Returns pairwise comparisons.

    Parameters
    ----------
     settings
        All settings.
    summary
        Summary from the signal model and bias covariate selection.
    cat_coefs
        Data frame with beta and gamma for each category for fitted linear model
    linear_model
        Fitted linear model for risk curve.

    Returns
    -------
    dict
        Dataframe containing pairwise beta, gamma, their standard deviations,
        and summary outputs.

    """
    # load summary
    ref_cat = summary["ref_cat"]
    cats = cat_coefs["cat"]
    n_cats = len(cats)
    cat_pairs_list = list(combinations(cats, 2))
    cat_pairs = pd.DataFrame(
        cat_pairs_list, columns=["ref_risk_cat", "alt_risk_cat"]
    )
    cat_pairs = (
        cat_pairs.merge(
            cat_coefs[["cat", "beta"]],
            left_on="ref_risk_cat",
            right_on="cat",
            how="left",
        )
        .rename(columns={"beta": "beta_ref"})
        .drop(columns=["cat"])
    )
    cat_pairs = (
        cat_pairs.merge(
            cat_coefs[["cat", "beta"]],
            left_on="alt_risk_cat",
            right_on="cat",
            how="left",
        )
        .rename(columns={"beta": "beta_alt"})
        .drop(columns=["cat"])
    )
    cat_pairs = (
        cat_pairs.merge(
            cat_coefs[["cat", "gamma"]],
            left_on="ref_risk_cat",
            right_on="cat",
            how="left",
        )
        # .rename(columns={"gamma": "gamma_ref"})
        .drop(columns=["cat"])
    )

    # Compute beta_adjusted: difference between alt and ref
    cat_pairs["beta_adjusted"] = cat_pairs["beta_alt"] - cat_pairs["beta_ref"]
    # Adjust so that all pairwise betas are positive
    mask = cat_pairs["beta_adjusted"] < 0
    cat_pairs.loc[
        mask,
        [
            "ref_risk_cat",
            "alt_risk_cat",
            "beta_ref",
            "beta_alt",
        ],
    ] = cat_pairs.loc[
        mask,
        [
            "alt_risk_cat",
            "ref_risk_cat",
            "beta_alt",
            "beta_ref",
        ],
    ].values
    cat_pairs.loc[mask, "beta_adjusted"] = -cat_pairs.loc[mask, "beta_adjusted"]
    cat_pairs["pair"] = (
        cat_pairs["alt_risk_cat"] + "-" + cat_pairs["ref_risk_cat"]
    )
    cat_pairs["pair_standardized"] = cat_pairs["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )

    # Compute variance for the pairwise betas
    indices = np.fromiter(combinations(range(n_cats), 2), dtype=(int, (2,)))
    pair_names = ["-".join(sorted([cats[i], cats[j]])) for i, j in indices]

    lt = linear_model.lt
    hessian = lt.hessian(lt.soln)[:n_cats, :n_cats]
    vmat = np.zeros((len(indices), n_cats))
    for i, index in enumerate(indices):
        vmat[i, index[0]] = 1
        vmat[i, index[1]] = -1
    variance = (vmat * np.linalg.solve(hessian, vmat.T).T).sum(axis=1)
    df_variance = pd.DataFrame(
        {"pair": pair_names, "variance": variance, "beta_sd": np.sqrt(variance)}
    )
    df_variance["pair_standardized"] = df_variance["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )
    cat_pairs = cat_pairs.merge(
        df_variance[["pair_standardized", "beta_sd"]],
        on="pair_standardized",
        how="left",
    ).rename(columns={"beta_sd": "inner_beta_sd"})
    # Now gamma – multiply by 2 since shifting to the pairwise comparison?
    cat_pairs["gamma_adjusted"] = 2 * cat_pairs["gamma"]
    gamma_sd = set(cat_coefs["gamma_sd"])
    if len(gamma_sd) == 1:
        # Sum variance of gamma from ref and alt to account for pairwise comparison
        cat_pairs["gamma_sd"] = 2 * gamma_sd.pop()
    else:
        print("Warning: there is more than one unique gamma_sd value")
    cat_pairs["outer_beta_sd"] = np.sqrt(
        cat_pairs["inner_beta_sd"] ** 2
        + cat_pairs["gamma_adjusted"]
        + 2 * cat_pairs["gamma_sd"]
    )

    cat_pairs_subset = cat_pairs[
        [
            "ref_risk_cat",
            "alt_risk_cat",
            "pair",
            "beta_adjusted",
            "inner_beta_sd",
            "outer_beta_sd",
            "gamma_adjusted",
            "gamma_sd",
        ]
    ].rename(columns={"beta_adjusted": "beta", "gamma_adjusted": "gamma"})
    sign = np.sign(cat_pairs_subset["beta"])
    cat_pairs_subset = cat_pairs_subset.assign(
        inner_ui_lower=cat_pairs_subset["beta"]
        - 1.96 * cat_pairs_subset["inner_beta_sd"],
        inner_ui_upper=cat_pairs_subset["beta"]
        + 1.96 * cat_pairs_subset["inner_beta_sd"],
        outer_ui_lower=cat_pairs_subset["beta"]
        - 1.96 * cat_pairs_subset["outer_beta_sd"],
        outer_ui_upper=cat_pairs_subset["beta"]
        + 1.96 * cat_pairs_subset["outer_beta_sd"],
        log_bprf=cat_pairs_subset["beta"]
        - sign * 1.645 * cat_pairs_subset["outer_beta_sd"],
    )
    signed_bprf = sign * cat_pairs_subset["log_bprf"]
    product = np.prod(
        cat_pairs_subset[["inner_ui_lower", "inner_ui_upper"]], axis=1
    )
    cat_pairs_subset["score"] = np.where(
        product < 0, float("nan"), 0.5 * signed_bprf
    )
    score_bounds = [
        np.isnan(cat_pairs_subset["score"]),
        cat_pairs_subset["score"] > np.log(1 + 0.85),
        cat_pairs_subset["score"] > np.log(1 + 0.5),
        cat_pairs_subset["score"] > np.log(1 + 0.15),
        cat_pairs_subset["score"] > 0,
    ]
    ratings = [0, 5, 4, 3, 2]
    cat_pairs_subset["star_rating"] = np.select(
        score_bounds, ratings, default=1
    )
    # Subset dataset to only original ref-alt comparisons if ordinal categories
    cat_order = settings["cat_order"]
    if not cat_order:
        cat_pairs_subset = cat_pairs_subset
    else:
        if sorted(cat_order) != sorted(cats.tolist()):
            raise ValueError(
                f"Error: cat_order does not match the expected categories. Expected: {cats}, but got: {cat_order}"
            )
        else:
            cat_pairs_subset = cat_pairs_subset[
                cat_pairs_subset["pair"].str.contains(ref_cat)
            ]
    cat_pairs_subset.sort_values(by="beta", ascending=False, inplace=True)
    # # Add plotting ranges
    # num_cats = cat_pairs_subset["pair"].nunique()
    # cat_pairs_subset["y_start"] = range(num_cats)
    # cat_pairs_subset["y_end"] = range(1, num_cats + 1)
    # cat_pairs_subset["y_mid"] = cat_pairs_subset.eval("0.5 * (y_start + y_end)")

    return cat_pairs_subset


def get_linear_model_summary(
    all_settings: dict,
    settings: dict,
    summary: dict,
    df: DataFrame,
    cat_coefs: DataFrame,
    pair_coefs: DataFrame,
    linear_model: MRBRT,
) -> dict:
    """Complete the summary from the signal model.

    Parameters
    ----------
    all_settings
        Complete list of settings
     settings
        Settings for the complete summary section.
    summary
        Summary from the signal model.
    df
        Data frame contains the all dataset.
    cat_coefs
        Data frame with beta and gamma for each category for fitted linear model
    pair_coefs
        Data frame with beta and gamma for the pairwise category comparisons
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
    cats = cat_coefs["cat"]
    # cats = linear_model.cov_models[0].cats
    alt_cats = [cat for cat in cats if cat != ref_cat]

    # solution of the final model with pairwise comparisons
    summary["beta"] = dict(zip(pair_coefs["pair"], pair_coefs["beta"]))
    summary["beta_sd"] = dict(
        zip(pair_coefs["pair"], pair_coefs["inner_beta_sd"])
    )
    summary["gamma"] = set(cat_coefs["gamma"]).pop()
    summary["gamma_sd"] = set(cat_coefs["gamma_sd"]).pop()
    summary["score"] = dict(zip(pair_coefs["pair"], pair_coefs["score"]))
    summary["star_rating"] = dict(
        zip(pair_coefs["pair"], pair_coefs["star_rating"])
    )
    # Output combined score and star rating if ordinal categories
    cat_order = all_settings["cat_order"]
    if cat_order:
        if sorted(cat_order) != sorted(cats.tolist()):
            raise ValueError(
                f"Error: cat_order does not match the expected categories. Expected: {cats}, but got: {cat_order}"
            )
        else:
            if np.any(np.isnan(pair_coefs["score"])):
                summary["combined_score"] = float("nan")
                summary["combined_star_rating"] = 0
            else:
                sign = np.sign(pair_coefs["beta"])
                signed_bprf = sign * pair_coefs["log_bprf"]
                max_idx = signed_bprf.idxmax()
                score = float(
                    (1 / len(alt_cats))
                    * (np.sum(signed_bprf) - 0.5 * signed_bprf[max_idx])
                )
                summary["combined_score"] = score
                # Assign star rating based on ROS
                if np.isnan(score):
                    summary["combined_star_rating"] = 0
                elif score > np.log(1 + 0.85):
                    summary["combined_star_rating"] = 5
                elif score > np.log(1 + 0.50):
                    summary["combined_star_rating"] = 4
                elif score > np.log(1 + 0.15):
                    summary["combined_star_rating"] = 3
                elif score > 0:
                    summary["combined_star_rating"] = 2
                else:
                    summary["combined_star_rating"] = 1

    # compute the publication bias
    index = df.is_outlier == 0
    beta_dict = dict(zip(cat_coefs["cat"], cat_coefs["beta"]))
    gamma_dict = dict(zip(cat_coefs["cat"], cat_coefs["gamma"]))
    residual = df["ln_rr"].values[index] - (
        df["alt_risk_cat"].map(beta_dict).values[index]
        - df["ref_risk_cat"].map(beta_dict).values[index]
    )
    residual_sd = np.sqrt(
        df.ln_rr_se.values[index] ** 2
        + df["ref_risk_cat"].map(gamma_dict).values[index]
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
    pair_coefs: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Create effect draws for the pipeline.

    Parameters
    ----------
    settings
        Settings for complete the summary.
    pair_coefs
        Dataframe containing parameter information for the pairwise results

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer draw files.

    """

    # beta_info = summary["beta"]
    # beta_sd_info = summary["beta_sd"]
    # gamma_info = summary["gamma"]
    # gamma_sd_info = summary["gamma_sd"]
    # inner_beta_sd = np.array(list(beta_sd_info.values()))
    # outer_beta_sd = np.sqrt(
    #     np.array(list(beta_sd_info.values())) ** 2
    #     + np.array(list(gamma_info.values()))
    #     + 2 * np.array(list(gamma_sd_info.values()))
    # )
    beta_info = pair_coefs["beta"]
    inner_beta_sd = pair_coefs["inner_beta_sd"]
    outer_beta_sd = pair_coefs["outer_beta_sd"]
    inner_beta_samples = np.random.normal(
        loc=np.array(beta_info)[:, None],
        scale=np.array(inner_beta_sd)[:, None],
        size=(len(beta_info), settings["draws"]["num_draws"]),
    )
    outer_beta_samples = np.random.normal(
        loc=np.array(beta_info)[:, None],
        scale=np.array(outer_beta_sd)[:, None],
        size=(len(beta_info), settings["draws"]["num_draws"]),
    )
    df_inner_draws = pd.DataFrame(
        np.hstack(
            [
                np.array(pair_coefs["ref_risk_cat"])[:, None],
                np.array(pair_coefs["alt_risk_cat"])[:, None],
                np.array(pair_coefs["pair"])[:, None],
                inner_beta_samples,
            ]
        ),
        columns=["ref_risk_cat"]
        + ["alt_risk_cat"]
        + ["risk_cat_pair"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )
    df_outer_draws = pd.DataFrame(
        np.hstack(
            [
                np.array(pair_coefs["ref_risk_cat"])[:, None],
                np.array(pair_coefs["alt_risk_cat"])[:, None],
                np.array(pair_coefs["pair"])[:, None],
                outer_beta_samples,
            ]
        ),
        columns=["ref_risk_cat"]
        + ["alt_risk_cat"]
        + ["risk_cat_pair"]
        + [f"draw_{i}" for i in range(settings["draws"]["num_draws"])],
    )

    return df_inner_draws, df_outer_draws


def get_quantiles(
    settings: dict,
    pair_coefs: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Create effect quantiles for the pipeline.

    Parameters
    ----------
    settings
        The settings for complete the summary.
    pair_coefs
        Dataframe containing parameter information for the pairwise results

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Inner and outer quantile files.

    """

    beta_info = pair_coefs["beta"]
    inner_beta_sd = pair_coefs["inner_beta_sd"]
    outer_beta_sd = pair_coefs["outer_beta_sd"]
    # get quantiles
    cat_pairs = np.array(pair_coefs["pair"])[:, None]
    coefs = np.array(beta_info)
    quantiles = np.asarray(settings["draws"]["quantiles"])
    inner_beta_quantiles = [
        norm.ppf(
            quantiles,
            loc=coefs[:, None],
            scale=np.array(inner_beta_sd)[:, None],
        ),
    ]
    inner_beta_quantiles = np.vstack(inner_beta_quantiles)
    outer_beta_quantiles = [
        norm.ppf(
            quantiles,
            loc=coefs[:, None],
            scale=np.array(outer_beta_sd)[:, None],
        ),
    ]
    outer_beta_quantiles = np.vstack(outer_beta_quantiles)
    df_inner_quantiles = pd.DataFrame(
        np.hstack(
            [
                np.array(pair_coefs["ref_risk_cat"])[:, None],
                np.array(pair_coefs["alt_risk_cat"])[:, None],
                cat_pairs,
                inner_beta_quantiles,
            ]
        ),
        columns=["ref_risk_cat"]
        + ["alt_risk_cat"]
        + ["risk_cat_pair"]
        + list(map(str, quantiles)),
    )
    df_outer_quantiles = pd.DataFrame(
        np.hstack(
            [
                np.array(pair_coefs["ref_risk_cat"])[:, None],
                np.array(pair_coefs["alt_risk_cat"])[:, None],
                cat_pairs,
                outer_beta_quantiles,
            ]
        ),
        columns=["ref_risk_cat"]
        + ["alt_risk_cat"]
        + ["risk_cat_pair"]
        + list(map(str, quantiles)),
    )

    return df_inner_quantiles, df_outer_quantiles


def plot_linear_model(
    name: str,
    summary: dict,
    df: DataFrame,
    cat_coefs: DataFrame,
    pair_coefs: DataFrame,
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
    cat_coefs
        Data frame containing the fitted beta and gamma coefficients
    pair_coefs
        Data frame containing fitted pairwise model coefficients
    show_ref
        Whether to show the reference line. Default is `True`.

    Returns
    -------
    Figure
        The figure object for linear model.

    """
    offset = 0.05
    pair_coefs = pair_coefs.copy()
    num_cats = pair_coefs["pair"].nunique()
    pair_coefs["y_start"] = range(num_cats)
    pair_coefs["y_end"] = range(1, num_cats + 1)
    pair_coefs["y_mid"] = pair_coefs.eval("0.5 * (y_start + y_end)")
    # create fig obj
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # plot data
    _plot_data(
        name,
        summary,
        df,
        pair_coefs,
        ax[0],
        show_ref=show_ref,
    )
    # plot beta coefficients and uncertainty
    pred = pair_coefs.beta
    log_bprf = pair_coefs.log_bprf
    y_start = pair_coefs["y_start"] + offset
    y_end = pair_coefs["y_end"] - offset
    # Plot coefficients
    ax[0].plot([pred] * 2, [y_start, y_end], color="#008080")
    # Plot BPRF
    ax[0].plot([log_bprf] * 2, [y_start, y_end], color="red")
    # Fill between UIs
    for i, row in pair_coefs.iterrows():
        ax[0].fill_betweenx(
            [row["y_start"] + offset, row["y_end"] - offset],
            row["inner_ui_lower"],
            row["inner_ui_upper"],
            color="gray",
            alpha=0.2,
        )
    for i, row in pair_coefs.iterrows():
        ax[0].fill_betweenx(
            [row["y_start"] + offset, row["y_end"] - offset],
            row["outer_ui_lower"],
            row["outer_ui_upper"],
            color="gray",
            alpha=0.2,
        )

    xlim = ax[0].get_xlim()
    max_star_count = pair_coefs["star_rating"].fillna(0).astype(int).max()
    max_label_length = max(max_star_count, 1)  # account for "0" or NaNs
    char_spacing = 0.02 * (xlim[1] - xlim[0])
    x_text_pos = xlim[1] + max_label_length * char_spacing
    # Add star ratings as text labels
    for _, row in pair_coefs.iterrows():
        stars = row["star_rating"]
        label = "★" * int(stars) if stars > 0 else "0"
        ax[0].text(
            x_text_pos,
            row["y_mid"],
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )
    ax[0].set_xlim(xlim[0], x_text_pos + 0.12 * (xlim[1] - xlim[0]))

    # plot funnel
    _plot_funnel(summary, cat_coefs, df, ax[1])

    return fig


def plot_linear_panel_model(
    df: DataFrame,
    cat_coefs: DataFrame,
    pair_coefs: DataFrame,
) -> Figure:
    """Plot the linear model

    Parameters
    ----------
    df
        Data frame contains the training data.
    cat_coefs
        Data frame containing the fitted beta and gamma coefficients
    pair_coefs
        Data frame containing fitted pairwise model coefficients

    Returns
    -------
    Figure
        The figure object for linear model.

    """
    offset = 0.05

    cat_pairs_flipped = pair_coefs.copy()
    cat_pairs_flipped["ref_risk_cat"], cat_pairs_flipped["alt_risk_cat"] = (
        cat_pairs_flipped["alt_risk_cat"],
        cat_pairs_flipped["ref_risk_cat"],
    )
    cat_pairs_flipped["pair"] = (
        cat_pairs_flipped["alt_risk_cat"]
        + "-"
        + cat_pairs_flipped["ref_risk_cat"]
    )
    cat_pairs_flipped[["beta", "log_bprf"]] *= -1
    cat_pairs_flipped["inner_ui_lower"], cat_pairs_flipped["inner_ui_upper"] = (
        -pair_coefs["inner_ui_upper"],
        -pair_coefs["inner_ui_lower"],
    )
    cat_pairs_flipped["outer_ui_lower"], cat_pairs_flipped["outer_ui_upper"] = (
        -pair_coefs["outer_ui_upper"],
        -pair_coefs["outer_ui_lower"],
    )

    all_pair_coefs = pd.concat([pair_coefs, cat_pairs_flipped])
    ref_cat_order = (
        all_pair_coefs.groupby("ref_risk_cat")["beta"]
        .apply(lambda x: (x > 0).sum())
        .sort_values(ascending=False)
        .index.tolist()
    )
    all_pair_coefs["pair_standardized"] = all_pair_coefs["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )
    x_min = all_pair_coefs["outer_ui_lower"].min() - 0.1
    x_max = all_pair_coefs["outer_ui_upper"].max() + 0.1
    n_cats = len(cat_coefs["cat"])
    n_cols = int(np.ceil(np.sqrt(n_cats)))  # Columns based on square root
    n_rows = int(np.ceil(n_cats / n_cols))  # Rows based on number of columns
    sub_width = 6
    sub_height = 4

    # create fig obj
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * sub_width, n_rows * sub_height)
    )
    axes = axes.flatten()

    for i, ref_cat in enumerate(ref_cat_order):
        ax = axes[i] if len(ref_cat_order) > 1 else axes

        # Subset
        panel_data = all_pair_coefs[all_pair_coefs["ref_risk_cat"] == ref_cat]
        num_pairs = len(panel_data)
        panel_data = panel_data.sort_values(by="beta", ascending=False)
        panel_data["y_start"] = range(num_pairs)
        panel_data["y_end"] = range(1, num_pairs + 1)
        panel_data["y_mid"] = 0.5 * (
            panel_data["y_start"] + panel_data["y_end"]
        )
        y_start = panel_data["y_start"] + offset
        y_end = panel_data["y_end"] - offset

        pred = panel_data.beta
        log_bprf = panel_data.log_bprf

        # plot coefficients
        ax.plot([pred] * 2, [y_start, y_end], color="#008080")
        # plot BPRF
        ax.plot([log_bprf] * 2, [y_start, y_end], color="red")

        # Fill between UIs
        for i, row in panel_data.iterrows():
            ax.fill_betweenx(
                [row["y_start"] + offset, row["y_end"] - offset],
                row["inner_ui_lower"],
                row["inner_ui_upper"],
                color="gray",
                alpha=0.2,
            )
        for i, row in panel_data.iterrows():
            ax.fill_betweenx(
                [row["y_start"] + offset, row["y_end"] - offset],
                row["outer_ui_lower"],
                row["outer_ui_upper"],
                color="gray",
                alpha=0.2,
            )

        # plot data
        df_subset = df[
            (df["ref_risk_cat"] == ref_cat) | (df["alt_risk_cat"] == ref_cat)
        ].copy()
        flipped_mask = df_subset["ref_risk_cat"] != ref_cat
        df_subset["ln_rr_adjusted"] = df_subset["ln_rr"]
        df_subset.loc[flipped_mask, "ln_rr_adjusted"] *= -1
        df_subset["pair_standardized"] = df_subset.apply(
            lambda row: "-".join(
                sorted([row["ref_risk_cat"], row["alt_risk_cat"]])
            ),
            axis=1,
        )
        df_subset = df_subset.merge(
            panel_data[["pair_standardized", "beta", "y_mid"]],
            on="pair_standardized",
            how="left",
        )
        adj_obs = df_subset.ln_rr_adjusted
        alt_cat_mid_jitter = df_subset.y_mid + np.random.uniform(
            -0.2, 0.2, df_subset.shape[0]
        )

        index = df_subset.is_outlier == 1
        ax.scatter(
            adj_obs,
            alt_cat_mid_jitter,
            s=5 / df_subset["ln_rr_se"].values,
            color="#008080",
            alpha=0.5,
            edgecolor="none",
        )
        ax.scatter(
            adj_obs[index],
            alt_cat_mid_jitter[index],
            s=5 / df_subset.ln_rr_se.values[index],
            color="red",
            alpha=0.5,
            marker="x",
        )

        # plot support lines
        ax.axvline(0.0, linewidth=1, linestyle="-", color="gray")

        # add labels
        ax.set_yticks(panel_data["y_mid"])
        ax.set_yticklabels(panel_data["alt_risk_cat"], rotation=0, ha="right")
        ax.set_xlabel("ln relative risk")
        ax.set_ylabel("alternative risk category")
        ax.set_title(f"reference risk category: {ref_cat}")
        ax.set_xlim(x_min, x_max)
        fig.align_ylabels(axes)
        plt.tight_layout()

    # Hide any unused subplots
    for i in range(n_cats, len(axes)):
        axes[i].axis("off")

    return fig


def _plot_data(
    name: str,
    summary: dict,
    df: DataFrame,
    pair_coefs: DataFrame,
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
    pair_coefs
        Data frame containing the fitted pairwise model coefficients
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

    # Check that pair ordering between pairwise comparisons
    # and data alt-ref pairs is consistent
    df["pair"] = df["alt_risk_cat"] + "-" + df["ref_risk_cat"]
    df["pair_standardized"] = df["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )
    # pair_coefs = pair_coefs.copy()
    pair_coefs["pair_standardized"] = pair_coefs["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )
    pair_coefs["pair_model"] = pair_coefs["pair"]
    df = df.merge(
        pair_coefs[["pair_standardized", "y_mid", "pair_model"]],
        on="pair_standardized",
        how="left",
    )
    # If order of alt-ref pair is swapped between df and pair_coefs,
    # multiply ln_rr by -1
    df["ln_rr_adjusted"] = np.where(
        df["pair"] == df["pair_model"],
        df["ln_rr"],
        -df["ln_rr"],
    )

    adj_obs = df.ln_rr_adjusted
    alt_cat_mid_jitter = df.y_mid + np.random.uniform(-0.2, 0.2, df.shape[0])

    # plot data points
    index = df.is_outlier == 1
    ax.scatter(
        adj_obs,
        alt_cat_mid_jitter,
        s=5 / df["ln_rr_se"].values,
        color="#008080",
        alpha=0.5,
        edgecolor="none",
    )
    ax.scatter(
        adj_obs[index],
        alt_cat_mid_jitter[index],
        s=5 / df.ln_rr_se.values[index],
        color="red",
        alpha=0.5,
        marker="x",
    )

    # # plot support lines
    ax.axvline(0.0, linewidth=1, linestyle="-", color="gray")

    # add title and label
    rei, _ = tuple(name.split("-"))
    ax.set_title(name.replace("-", " / "), loc="left")
    ax.set_yticks(pair_coefs["y_mid"])
    ax.set_yticklabels(pair_coefs["pair"], rotation=0, ha="right")
    ax.set_xlabel("ln relative risk")

    return ax


def _plot_funnel(
    summary: dict, cat_coefs: DataFrame, df: DataFrame, ax: Axes
) -> Axes:
    """Plot the funnel plot

    Parameters
    ----------
    summary
        Complete summary file.
    cat_coefs
        Data frame with category-specific fitted model coefficients
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
    beta = dict(zip(cat_coefs["cat"], cat_coefs["beta"]))
    gamma = summary["gamma"]
    residual = df.ln_rr.values - (
        df.alt_risk_cat.map(beta).values - df.ref_risk_cat.map(beta).values
    )
    residual_sd = np.sqrt(df["ln_rr_se"].values ** 2 + (gamma))
    # if summary["cat_specific_gamma"]:
    #     residual_sd = np.sqrt(
    #         df["ln_rr_se"].values ** 2
    #         + (
    #             df["alt_risk_cat"].map(gamma).values
    #             + df["ref_risk_cat"].map(gamma).values
    #         )
    #     )
    # else:
    #     residual_sd = np.sqrt(
    #         df.ln_rr_se.values**2 + df["ref_risk_cat"].map(gamma).values
    #     )

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


# def plot_signal_model(
#     name: str,
#     summary: dict,
#     df: DataFrame,
#     cat_coefs: DataFrame,
#     show_ref: bool = True,
# ) -> Figure:
#     """Plot the signal model

#     Parameters
#     ----------
#     name
#         Name of the pair.
#     summary
#         Summary from the signal model.
#     df
#         Data frame contains training data.
#     df_coef
#         Data frame containing the fitted beta coefficients for the signal model
#     show_ref
#         Whether to show the reference line. Default is `True`.

#     Returns
#     -------
#     Figure
#         The figure object for signal model.

#     """
#     offset = 0.05
#     # create fig obj
#     fig, ax = plt.subplots(figsize=(8, 5))

#     # plot data
#     _plot_data(
#         name,
#         summary,
#         df,
#         cat_coefs,
#         ax,
#         show_ref=show_ref,
#     )

#     # plot beta coefficients
#     if summary["normalize_to_tmrel"]:
#         coef_min = cat_coefs.beta.min()
#         for i, row in cat_coefs.iterrows():
#             ax.plot(
#                 [row["x_start"] + offset, row["x_end"] - offset],
#                 [row["beta"] - coef_min] * 2,
#                 color="#008080",
#             )
#     else:
#         for i, row in cat_coefs.iterrows():
#             ax.plot(
#                 [row["x_start"] + offset, row["x_end"] - offset],
#                 [row["beta"]] * 2,
#                 color="#008080",
#             )

#     return fig


# def get_cat_coefs_old(
#     settings: dict,
#     model: MRBRT,
#     type: str,
#     ref_cat_input: str,
# ) -> tuple[DataFrame]:
#     """Get beta and gamma coefficients for the categories.

#     Parameters
#     ----------
#     df
#         Data frame contains the training data.
#     settings
#         The settings for category order
#     model
#         Fitted model for the categories, linear or signal
#     type
#         String specifying whether model is signal or linear model
#     ref_cat_input
#         Reference category. Optional, will be inferred from settings or data
#         if not provided.

#     Returns
#     -------
#     tuple[DataFrame]
#         Dataframe of beta, beta_sd, gamma, gamma_sd for each category and ranges to plot each category

#     """

#     lt = model.lt

#     # Extract betas for categories and covariates, ensuring correct matching
#     cov_names = []
#     for cov_model in model.cov_models:
#         if isinstance(cov_model, LinearCovModel):
#             cov_name = cov_model.alt_cov[0]
#             if cov_model.num_x_vars == 1:
#                 cov_names.append(cov_name)
#             else:
#                 cov_names.extend(
#                     [f"{cov_name}_{i}" for i in range(cov_model.num_x_vars)]
#                 )
#         elif isinstance(cov_model, LinearCatCovModel):
#             cov_names.extend("cat_" + cov_model.cats.astype(str))
#         else:
#             raise TypeError("Unknown cov_model type")

#     beta = model.beta_soln
#     hessian = lt.hessian(lt.soln)[: lt.k_beta, : lt.k_beta]
#     beta_sd = 1.0 / np.sqrt(np.diag(hessian))

#     beta_info = pd.DataFrame(
#         {
#             "cov_name": cov_names,
#             "beta": beta,
#             # "beta_sd": beta_sd,
#         }
#     )
#     # Subset to betas for categories only
#     beta_cats = beta_info[beta_info["cov_name"].str.startswith("cat_")].copy()
#     beta_cats["cat"] = beta_cats["cov_name"].str.removeprefix("cat_")

#     # Extract gamma, calculate gamma_sd, and merge with beta dataframe for linear model only
#     # If category-specific gamma is not included:
#     # gamma/gamma_sd values will be identical for each category
#     if type == "linear":
#         lt = model.lt
#         if settings["complete_summary"]["cat_gamma"]["cat_specific_gamma"]:
#             gamma = model.gamma_soln
#             gamma_fisher = lt.get_gamma_fisher(gamma)
#             gamma_cov = np.linalg.inv(gamma_fisher)
#             gamma_sd = np.sqrt(np.diag(gamma_cov))
#             gamma_cats = pd.DataFrame(
#                 {
#                     "cat": model.cov_models[
#                         model.cov_model_names == "alt_risk_cat"
#                     ].cats,
#                     "gamma": gamma,
#                     "gamma_sd": gamma_sd,
#                 }
#             )
#         else:
#             gamma = model.gamma_soln[0]
#             gamma_fisher = lt.get_gamma_fisher(gamma)
#             gamma_sd = 1.0 / np.sqrt(gamma_fisher[0, 0])
#             gamma_cats = pd.DataFrame(
#                 {
#                     "cat": model.cov_models[
#                         model.cov_model_names == "alt_risk_cat"
#                     ].cats,
#                     "gamma": np.repeat(
#                         gamma,
#                         len(
#                             model.cov_models[
#                                 model.cov_model_names == "alt_risk_cat"
#                             ].cats
#                         ),
#                     ),
#                     "gamma_sd": np.repeat(
#                         gamma_sd,
#                         len(
#                             model.cov_models[
#                                 model.cov_model_names == "alt_risk_cat"
#                             ].cats
#                         ),
#                     ),
#                 }
#             )
#         cat_coefs = beta_cats.merge(gamma_cats, on="cat", how="left")
#     else:
#         cat_coefs = beta_cats

#     # Order the categories
#     cat_order = settings["figure"]["cat_order"]
#     if cat_order:
#         cat_coefs["cat"] = pd.Categorical(
#             cat_coefs["cat"], categories=cat_order, ordered=True
#         )
#         cat_coefs = cat_coefs.sort_values("cat").reset_index(drop=True)
#         cat_coefs["cat"] = cat_coefs["cat"].astype(str)
#     else:
#         # ref_cat first, then order by proximity to ref_cat's beta coefficient
#         ref_beta = cat_coefs.loc[cat_coefs["cat"] == ref_cat_input, "beta"].iloc[0]
#         cat_coefs["abs_diff"] = (cat_coefs["beta"] - ref_beta).abs()
#         cat_coefs = cat_coefs.sort_values(by="abs_diff")
#         cat_coefs = pd.concat(
#             [
#                 cat_coefs[cat_coefs["cat"] == ref_cat_input],
#                 cat_coefs[cat_coefs["cat"] != ref_cat_input],
#             ]
#         ).reset_index(drop=True)
#     cat_coefs = cat_coefs.sort_values(by=["cat" == ref_cat_input, "abs_diff"], ascending=[False, True])
#     cat_coefs = cat_coefs.drop(columns=["abs_diff"])
#     # Add x ranges for plotting
#     num_cats = cat_coefs["cat"].nunique()
#     cat_coefs["x_start"] = range(num_cats)
#     cat_coefs["x_end"] = range(1, num_cats + 1)
#     cat_coefs["x_mid"] = cat_coefs.eval("0.5 * (x_start + x_end)")

#     return cat_coefs


# def get_linear_model_summary(
#     settings: dict,
#     summary: dict,
#     df: DataFrame,
#     cat_coefs: DataFrame,
#     linear_model: MRBRT,
# ) -> dict:
#     """Complete the summary from the signal model.

#     Parameters
#     ----------
#      settings
#         Settings for the complete summary section.
#     summary
#         Summary from the signal model.
#     df
#         Data frame contains the all dataset.
#     cat_coefs
#         Data frame with beta and gamma for each category for fitted linear model
#     linear_model
#         Fitted linear model for risk curve.

#     Returns
#     -------
#     dict
#         Summary file contains all necessary information.

#     """
#     # load summary
#     summary["normalize_to_tmrel"] = settings["score"]["normalize_to_tmrel"]
#     ref_cat = summary["ref_cat"]
#     cats = cat_coefs["cat"]
#     # cats = linear_model.cov_models[0].cats
#     alt_cats = [cat for cat in cats if cat != ref_cat]

#     # solution of the final model
#     summary["beta"] = dict(zip(cat_coefs["cat"], cat_coefs["beta"]))
#     summary["beta_sd"] = dict(zip(cat_coefs["cat"], cat_coefs["beta_sd"]))
#     summary["gamma"] = dict(zip(cat_coefs["cat"], cat_coefs["gamma"]))
#     summary["gamma_sd"] = dict(zip(cat_coefs["cat"], cat_coefs["gamma_sd"]))
#     # beta_info = get_beta_cats(linear_model)
#     # gamma_info = get_gamma_info(linear_model)
#     # summary["beta"] = dict(
#     #     zip(beta_info["cov_name_standard"], beta_info["beta"])
#     # )
#     # summary["beta_sd"] = dict(
#     #     zip(beta_info["cov_name_standard"], beta_info["beta_sd"])
#     # )
#     # summary["gamma"] = [float(gamma_info[0]), float(gamma_info[1])]

#     # compute the score and add star rating
#     # Subset to only alternative categories
#     alt_cat_coefs = cat_coefs[cat_coefs["cat"] != ref_cat]
#     beta_sd = np.sqrt(
#         alt_cat_coefs["beta_sd"] ** 2
#         + alt_cat_coefs["gamma"]
#         + 2 * alt_cat_coefs["gamma_sd"]
#     )
#     pred = np.array(alt_cat_coefs["beta"])
#     inner_ui = np.vstack(
#         [
#             alt_cat_coefs["beta"] - 1.96 * alt_cat_coefs["beta_sd"],
#             alt_cat_coefs["beta"] + 1.96 * alt_cat_coefs["beta_sd"],
#         ]
#     )
#     sign = np.sign(pred)
#     burden_of_proof = alt_cat_coefs["beta"] - sign * 1.645 * beta_sd

#     if settings["score"]["normalize_to_tmrel"]:
#         index = np.argmin(pred)
#         pred -= pred[index]
#         burden_of_proof -= burden_of_proof[None, index]
#         inner_ui -= inner_ui[:, None, index]

#     signed_bprf = sign * burden_of_proof
#     # Number of alternative categories
#     n = len(alt_cats)
#     # Assign dichotomous score for each alternative category
#     score_by_category = np.zeros(n)
#     product = np.prod(inner_ui, axis=0)
#     score_by_category[product < 0] = float("nan")
#     score_by_category[product >= 0] = 0.5 * signed_bprf[product >= 0]
#     summary["score_by_category"] = dict(
#         zip(alt_cats, score_by_category.tolist())
#     )
#     # Index with largest signed coefficient
#     max_idx = signed_bprf.idxmax()
#     if np.any(product < 0):
#         summary["score"] = float("nan")
#         summary["star_rating"] = 0
#     else:
#         score = float(
#             (1 / n) * (np.sum(signed_bprf) - 0.5 * signed_bprf[max_idx])
#         )
#         summary["score"] = score
#         # Assign star rating based on ROS
#         if np.isnan(score):
#             summary["star_rating"] = 0
#         elif score > np.log(1 + 0.85):
#             summary["star_rating"] = 5
#         elif score > np.log(1 + 0.50):
#             summary["star_rating"] = 4
#         elif score > np.log(1 + 0.15):
#             summary["star_rating"] = 3
#         elif score > 0:
#             summary["star_rating"] = 2
#         else:
#             summary["star_rating"] = 1

#     # compute the publication bias
#     index = df.is_outlier == 0
#     beta_dict = dict(zip(cat_coefs["cat"], cat_coefs["beta"]))
#     gamma_dict = dict(zip(cat_coefs["cat"], cat_coefs["gamma"]))
#     # beta_dict = dict(zip(beta_info["cov_name_standard"], beta_info["beta"]))
#     residual = df["ln_rr"].values[index] - (
#         df["alt_risk_cat"].map(beta_dict).values[index]
#         - df["ref_risk_cat"].map(beta_dict).values[index]
#     )
#     if summary["cat_specific_gamma"]:
#         residual_sd = np.sqrt(
#             df["ln_rr_se"].values[index] ** 2
#             + (
#                 df["alt_risk_cat"].map(gamma_dict).values[index]
#                 + df["ref_risk_cat"].map(gamma_dict).values[index]
#             )
#         )
#     else:
#         # Avoid doubling gamma contribution in case of shared gamma across categories
#         # Could equally use "alt_risk_cat" here as gamma_dict will return the
#         # same value for both, since gamma is the same across all categories
#         residual_sd = np.sqrt(
#             df.ln_rr_se.values[index] ** 2
#             + df["ref_risk_cat"].map(gamma_dict).values[index]
#         )
#     weighted_residual = residual / residual_sd
#     r_mean = weighted_residual.mean()
#     r_sd = 1 / np.sqrt(weighted_residual.size)
#     pval = 1 - norm.cdf(np.abs(r_mean / r_sd))
#     summary["pub_bias"] = int(pval < 0.05)
#     summary["pub_bias_pval"] = float(pval)

#     return summary

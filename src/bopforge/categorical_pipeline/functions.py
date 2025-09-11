from itertools import chain, combinations
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes, Figure
from mrtool import MRBRT, CovFinder, LinearCatCovModel, LinearCovModel, MRData
from pandas import DataFrame
from scipy.stats import norm

from bopforge.utils import (
    _validate_required_quantiles,
    get_beta_info,
    score_to_star_rating,
)


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
        Settings for the model fitting

    Returns
    -------
    tuple[DataFrame, dict]
        Updated dataframe with problematic covariate columns dropped
        and updated settings file with problematic covariates removed
    """
    pre_selected_cov_settings = settings["select_bias_covs"]["cov_finder"]
    cov_settings = settings["cov_type"]

    # Parse types of covariates
    bias_covs = set(cov_settings["bias_covs"])
    interacted_covs = set(cov_settings["interacted_covs"])
    non_interacted_covs = set(cov_settings["non_interacted_covs"])
    pre_selected_covs = set(pre_selected_cov_settings["pre_selected_covs"])

    # Validation checks
    _validate_distinct_cov_sets(bias_covs, interacted_covs, non_interacted_covs)
    all_covs = bias_covs | interacted_covs | non_interacted_covs
    _validate_covs_in_data(df, all_covs)
    _validate_binary_bias_covs(df, bias_covs)
    _validate_preselected_subset_bias(pre_selected_covs, bias_covs)

    # Drop from dataframe
    covs_to_remove = _find_covs_to_remove(df, all_covs)
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
    settings["select_bias_covs"]["cov_finder"] = pre_selected_cov_settings
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
        Settings for the entire model

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


def get_signal_model(df: DataFrame, settings: dict, summary: dict) -> MRBRT:
    """Create signal model for outliers identification and covariate selection
    step.

    Parameters
    ----------
    df
        Data frame contains the training data.
    settings
        Dictionary containing settings for covariates and fitting the signal model
    summary
        Dictionary containing initial summary outputs

    Returns
    -------
    MRBRT
        Signal model to access the strength of the prior on the bias-covariate.
    """
    ref_cat = summary["ref_cat"]
    signal_model_settings = settings["fit_signal_model"]
    cov_settings = settings["cov_type"]

    # Load in model covariates and candidate bias covariates
    # interacted needs the design matrix columns from input dataframe
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
    df: DataFrame,
    all_settings: dict,
    settings: dict,
    cov_finder_linear_model: MRBRT,
) -> CovFinder:
    """Create the instance of CovFinder class.

    Parameters
    ----------
    df
        Dataframe containing training data with column indicating outliers
    all_settings
        All model settings
    settings
        Settings for pre-selected bias covariates.
    cov_finder_linear_model
        Fitted cov finder linear model (signal model for categorical risks)

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
    col_covs = all_settings["cov_type"]["bias_covs"]
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
    model: MRBRT,
) -> tuple[DataFrame]:
    """Get beta and gamma coefficients for the categories.

    Parameters
    ----------
    model
        Fitted model for the categories, linear or signal

    Returns
    -------
    tuple[DataFrame]
        Dataframe of beta, gamma, gamma_sd for each category
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
        }
    )
    # Subset to betas for exposure categories only (no covariates)
    beta_cats = beta_info[beta_info["cov_name"].str.startswith("cat_")].copy()
    beta_cats["cat"] = beta_cats["cov_name"].str.removeprefix("cat_")

    # Extract gamma, calculate gamma_sd, and merge with beta dataframe
    gamma = model.gamma_soln[0]
    gamma_fisher = lt.get_gamma_fisher(gamma)
    gamma_sd = 1.0 / np.sqrt(gamma_fisher[0, 0])
    cat_coefs = beta_cats
    n_cats = len(cat_coefs["cat"])
    cat_coefs["gamma"] = np.repeat(gamma, n_cats)
    cat_coefs["gamma_sd"] = np.repeat(gamma_sd, n_cats)

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
    non_interacted_covs = settings["cov_type"]["non_interacted_covs"]
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


def hess_subset(matrix: np.typing.NDArray, i: int) -> np.ndarray:
    """Remove the ith row and ith column from the Hessian to create the
    sub-matrix of the Hessian for use in calculating pairwise beta variance

    Parameters
    ----------
    matrix
        Hessian matrix for subsetting
    i
        index of the row and column to remove

    Returns
    -------
    np.ndarray
        The (n-1) x (n-1) sub-Hessian.
    """

    return np.delete(np.delete(matrix, i, axis=0), i, axis=1)


def get_pairwise_beta_var(
    hessian: np.ndarray, cat_names: list[str]
) -> DataFrame:
    """Calculate beta_sd for pairwise comparisons using submatrices of the Hessian.
    Removing the i'th row and column of the Hessian allows us to directly invert
    the resulting sub-Hessian (no longer singular) to get the approximated
    covariance matrix sigma_i, where the diagonal elements sigma_i_{jj} are the
    variance of category j given category i as reference. Repeating for each
    row/column gives us the full set of variances for each category with respect
    to all other categories; variance of category j with respect to category i
    is the same as variance of category i with respect to category j; i.e., the
    variance is category order-invariant. This function extracts the diagonals
    of each sub-Hessian and stores only the unique pair-variance combinations to
    obtain the full set of pairwise variances/standard deviations of beta.

    Parameters
    ----------
     hessian
        Hessian submatrix obtained by removing ith row and column from Hessian.
    cat_names
        Category names corresponding to coefficients

    Returns
    -------
    DataFrame
        Dataframe containing pairwise beta standard deviations.
    """
    n_cats = len(cat_names)
    pair_variances = []
    for i in range(n_cats):
        ref_cat = cat_names[i]
        vcov_sub = np.linalg.inv(hess_subset(hessian, i))
        alt_idx = [j for j in range(n_cats) if j != i]
        for j, idx in enumerate(alt_idx):
            alt_cat = cat_names[idx]
            pair = "-".join(sorted([ref_cat, alt_cat]))
            if pair not in {p[0] for p in pair_variances}:
                variance = vcov_sub[j, j]
                pair_variances.append((pair, variance))
    # Convert to DataFrame and join
    var_df = pd.DataFrame(
        pair_variances, columns=["pair_standardized", "variance"]
    )
    var_df["inner_beta_sd"] = np.sqrt(var_df["variance"])

    return var_df[["pair_standardized", "inner_beta_sd"]]


def get_scores(pair_info: DataFrame) -> DataFrame:
    """Calculate UIs, logBPRF, and scores for pairwise comparisons

    Parameters
    ----------
    pair_info
        Dataframe containing pairwise betas and inner/outer beta SDs

    Returns
    -------
    DataFrame
        Updated pair_info dataframe with UIs, logBPRF, and scores appended
    """
    sign = np.sign(pair_info["beta"])
    pair_info = pair_info.assign(
        inner_ui_lower=pair_info["beta"] - 1.96 * pair_info["inner_beta_sd"],
        inner_ui_upper=pair_info["beta"] + 1.96 * pair_info["inner_beta_sd"],
        outer_ui_lower=pair_info["beta"] - 1.96 * pair_info["outer_beta_sd"],
        outer_ui_upper=pair_info["beta"] + 1.96 * pair_info["outer_beta_sd"],
        log_bprf=pair_info["beta"] - sign * 1.645 * pair_info["outer_beta_sd"],
    )
    signed_bprf = sign * pair_info["log_bprf"]
    product = np.prod(pair_info[["inner_ui_lower", "inner_ui_upper"]], axis=1)
    pair_info["score"] = np.where(product < 0, float("nan"), 0.5 * signed_bprf)

    return pair_info


def get_pair_info(
    settings: dict,
    summary: dict,
    cat_coefs: DataFrame,
    linear_model: MRBRT,
) -> tuple[DataFrame]:
    """Returns pairwise comparisons and parameters.

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
    DataFrame
        Dataframe containing pairwise beta, gamma, their standard deviations,
        and summary outputs.
    """
    # load summary
    ref_cat = summary["ref_cat"]
    cats = cat_coefs["cat"]
    n_cats = len(cats)

    cat_pairs = pd.DataFrame(
        list(combinations(cats, 2)), columns=["ref_risk_cat", "alt_risk_cat"]
    )
    # Calculate pairwise beta as beta_alt - beta_ref
    cat_to_beta = cat_coefs.set_index("cat")["beta"]
    cat_pairs["beta_adjusted"] = cat_pairs["alt_risk_cat"].map(
        cat_to_beta
    ) - cat_pairs["ref_risk_cat"].map(cat_to_beta)
    # Adjust so all pairwise betas are positive
    mask = cat_pairs["beta_adjusted"] < 0
    cat_pairs.loc[
        mask,
        [
            "ref_risk_cat",
            "alt_risk_cat",
        ],
    ] = cat_pairs.loc[
        mask,
        [
            "alt_risk_cat",
            "ref_risk_cat",
        ],
    ].values
    cat_pairs["beta_adjusted"] = cat_pairs["beta_adjusted"].abs()
    # Add pair name
    cat_pairs["pair"] = (
        cat_pairs["alt_risk_cat"] + "-" + cat_pairs["ref_risk_cat"]
    )
    cat_pairs["pair_standardized"] = cat_pairs["pair"].apply(
        lambda x: "-".join(sorted(x.split("-")))
    )

    # Compute variance for the pairwise betas
    cat_names = []
    for cov_model in linear_model.cov_models:
        if isinstance(cov_model, LinearCatCovModel):
            cat_names.extend(cov_model.cats.astype(str))
    lt = linear_model.lt
    hessian = lt.hessian(lt.soln)[:n_cats, :n_cats]
    var_df = get_pairwise_beta_var(hessian, cat_names)
    cat_pairs = cat_pairs.merge(
        var_df,
        on="pair_standardized",
        how="left",
    )

    # Add gamma and gamma_sd
    cat_to_gamma = cat_coefs.set_index("cat")["gamma"]
    cat_pairs["gamma"] = cat_pairs["ref_risk_cat"].map(cat_to_gamma)
    unique_gamma_sd = cat_coefs["gamma_sd"].unique()
    if len(unique_gamma_sd) != 1:
        print("Warning: more than one unique gamma_sd value found")
    gamma_sd = unique_gamma_sd[0]
    cat_pairs["gamma_sd"] = gamma_sd
    cat_pairs["outer_beta_sd"] = np.sqrt(
        cat_pairs["inner_beta_sd"] ** 2
        + cat_pairs["gamma"]
        + 2 * cat_pairs["gamma_sd"]
    )
    # Subset to just the relevant columns
    cat_pairs_subset = cat_pairs.drop(columns="pair_standardized").rename(
        columns={"beta_adjusted": "beta"}
    )

    # Calculate UIs, logBPRF, and scores
    cat_pairs_subset = get_scores(cat_pairs_subset)
    # Add star ratings
    cat_pairs_subset["star_rating"] = cat_pairs_subset["score"].apply(
        score_to_star_rating
    )
    # Subset dataset to only original ref-alt comparisons if ordinal categories
    cat_order = settings["cat_order"]
    if not cat_order:
        cat_pairs_subset = cat_pairs_subset
    else:
        cat_pairs_subset = cat_pairs_subset[
            cat_pairs_subset["pair"].str.contains(ref_cat)
        ]
    cat_pairs_subset.sort_values(by="beta", ascending=False, inplace=True)

    return cat_pairs_subset


def get_linear_model_summary(
    df: DataFrame,
    all_settings: dict,
    settings: dict,
    summary: dict,
    cat_coefs: DataFrame,
    pair_coefs: DataFrame,
) -> dict:
    """Complete the summary from the signal model.

    Parameters
    ----------
    df
        Data frame contains the all dataset.
    all_settings
        Complete list of settings
    settings
        Settings for the complete summary section.
    summary
        Summary from the signal model.
    cat_coefs
        Data frame with beta and gamma for each category for fitted linear model
    pair_coefs
        Data frame with beta and gamma for the pairwise category comparisons

    Returns
    -------
    dict
        Summary file contains all necessary information.
    """
    # load summary
    ref_cat = summary["ref_cat"]
    cats = cat_coefs["cat"]
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

    cat_order = all_settings["cat_order"]
    # Output combined score and star rating: max score if non-ordinal,
    # 'averaged' score if ordinal categories
    if cat_order:
        if np.any(np.isnan(pair_coefs["score"])):
            summary["combined_score"] = float("nan")
        else:
            sign = np.sign(pair_coefs["beta"])
            signed_bprf = sign * pair_coefs["log_bprf"]
            max_idx = signed_bprf.abs().idxmax()
            score = float(
                (1 / len(alt_cats))
                * (np.sum(signed_bprf) - 0.5 * signed_bprf[max_idx])
            )
            summary["combined_score"] = score
        summary["combined_star_rating"] = score_to_star_rating(score)
        summary["category_type"] = "ordinal"
    else:
        max_idx = pair_coefs["score"].idxmax()
        summary["combined_score"] = float(pair_coefs.loc[max_idx, "score"])
        summary["combined_star_rating"] = int(
            pair_coefs.loc[max_idx, "star_rating"]
        )
        summary["category_type"] = "non-ordinal"

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
    coefs = np.array(beta_info)
    quantiles = np.asarray(settings["draws"]["quantiles"])
    quantiles = _validate_required_quantiles(quantiles)
    inner_beta_quantiles = norm.ppf(
        quantiles,
        loc=coefs[:, None],
        scale=np.array(inner_beta_sd)[:, None],
    )
    inner_beta_quantiles = np.vstack(inner_beta_quantiles)
    outer_beta_quantiles = norm.ppf(
        quantiles,
        loc=coefs[:, None],
        scale=np.array(outer_beta_sd)[:, None],
    )
    outer_beta_quantiles = np.vstack(outer_beta_quantiles)
    df_inner_quantiles = pd.DataFrame(
        inner_beta_quantiles, columns=list(map(str, quantiles))
    )
    df_inner_quantiles.insert(0, "risk_cat_pair", pair_coefs["pair"])
    df_inner_quantiles.insert(0, "alt_risk_cat", pair_coefs["alt_risk_cat"])
    df_inner_quantiles.insert(0, "ref_risk_cat", pair_coefs["ref_risk_cat"])
    df_outer_quantiles = pd.DataFrame(
        outer_beta_quantiles, columns=list(map(str, quantiles))
    )
    df_outer_quantiles.insert(0, "risk_cat_pair", pair_coefs["pair"])
    df_outer_quantiles.insert(0, "alt_risk_cat", pair_coefs["alt_risk_cat"])
    df_outer_quantiles.insert(0, "ref_risk_cat", pair_coefs["ref_risk_cat"])

    return df_inner_quantiles, df_outer_quantiles


def plot_linear_model(
    df: DataFrame,
    name: str,
    summary: dict,
    cat_coefs: DataFrame,
    pair_coefs: DataFrame,
) -> Figure:
    """Plot the linear model

    Parameters
    ----------
    df
        Data frame contains the training data.
    name
        Name of the pair
    summary
        Completed summary file.
    cat_coefs
        Data frame containing the fitted beta and gamma coefficients
    pair_coefs
        Data frame containing fitted pairwise model parameters

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
        df,
        name,
        pair_coefs,
        ax[0],
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
        label = "★" * int(stars) if stars > 0 else "☆"
        ax[0].text(
            x_text_pos,
            row["y_mid"],
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )
    ax[0].set_xlim(
        xlim[0], x_text_pos + max_label_length * char_spacing + 2 * char_spacing
    )

    # plot funnel
    _plot_funnel(df, summary, cat_coefs, ax[1])

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
    df: DataFrame,
    name: str,
    pair_coefs: DataFrame,
    ax: Axes,
) -> Axes:
    """Plot data points

    Parameters
    ----------
    df
        Data frame contains training data.
    name
        Name of the pair.
    pair_coefs
        Data frame containing the fitted pairwise model coefficients
    ax
        Axes of the figure. Usually corresponding to one panel of a figure.

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
    df: DataFrame, summary: dict, cat_coefs: DataFrame, ax: Axes
) -> Axes:
    """Plot the funnel plot
    Parameters
    ----------
    df
        Data frame that contains training data.
    summary
        Complete summary file.
    cat_coefs
        Data frame with category-specific fitted model coefficients
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


def _validate_cat_order(cat_order: list, cats: pd.Series) -> None:
    """Validate that the category list matches the list of fitted categories.
    Should be provided for ordinal categories only.

    Parameters
    ----------
    cat_order
        list of all risk exposure categories provided in cat_order setting
    cats
        risk exposure categories extracted from dataset

    Returns
    -------
    ValueError if user-provided list of categories does not match the categories in the data
    """
    if cat_order:
        if sorted(cat_order) != sorted(cats.tolist()):
            raise ValueError(
                f"Error: cat_order does not match the expected categories. "
                f"Expected: {sorted(cats.tolist())}, but got: {sorted(cat_order)}"
            )


def _validate_distinct_cov_sets(
    bias: set, interacted: set, non_interacted: set
) -> None:
    """Validate that the lists for each covariate type are distinct, i.e.,
    no covariate is listed as multiple types

    Parameters
    ----------
    bias
        Set of bias covariates
    interacted
        Set of interacted model covariates
    non_interacted
        Set of non-interacted model covariates

    Returns
    -------
    ValueError if a covariate is present in more than one list
    """
    cov_sets = {
        "bias": bias,
        "interacted": interacted,
        "non_interacted": non_interacted,
    }
    for (name1, covs1), (name2, covs2) in combinations(cov_sets.items(), 2):
        overlap = covs1 & covs2
        if overlap:
            raise ValueError(
                f"Covariates defined in both '{name1}' and '{name2}': {overlap}"
            )


def _validate_covs_in_data(df: DataFrame, covs: set) -> None:
    """Validate that all covariates in the settings are present in the data

    Parameters
    ----------
    df
        Input dataframe
    covs
        Set of all covariates listed in settings file

    Returns
    -------
    ValueError if a covariate is included in settings but not found in data
    """
    covs_missing_from_df = covs - set(df.columns)
    if covs_missing_from_df:
        raise ValueError(
            f"The following covariates are specified in settings but not found in the dataframe: {covs_missing_from_df}"
        )


def _validate_binary_bias_covs(df: DataFrame, bias_covs: set) -> None:
    """Validate that all bias covariates are binary

    Parameters
    ----------
    df
        Input dataframe
    bias_covs
        Set of all bias covariates listed in settings file

    Returns
    -------
    ValueError if a bias covariate is not binary
    """
    for cov in bias_covs:
        unique_vals = df[cov].unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Bias covariate '{cov}' is not binary")


def _validate_preselected_subset_bias(
    pre_selected_covs: set, bias_covs: set
) -> None:
    """Validate that all pre-selected covariates are contained in bias covariate list

    Parameters
    ----------
    pre_selected_covs
        Pre-selected bias covariates to include in the model
    bias_covs
        Set of all possible bias covariates listed in settings file

    Returns
    -------
    ValueError if a pre-selected bias cov is not found in list of all bias covariates
    """
    assert pre_selected_covs <= bias_covs, (
        f"pre_selected_covs must be a subset of bias_covs, but had additional non-bias covariates: "
        f"{pre_selected_covs - bias_covs}"
    )


def _validate_cat_order_prior_order_match(
    cat_order: list, prior_order: list
) -> None:
    """For ordinal categories (cat_order provided), if prior_order is specified,
    validate that cat_order and prior_order match exactly.

    Parameters
    ----------
    cat_order
        Setting containing complete list of ordered categories for ordinal risks.
        Must be provided for ordinal categories, will be empty for non-ordinal categories.
    prior_order
        Setting containing list of prior ordering for the categories. May be empty.
        For ordinal categories, cat_order and prior_order must match if both are provided.

    Returns
    -------
    ValueError if both cat_order and prior_order are provided but do not match.
    """
    if cat_order:
        if prior_order:
            flat_prior = list(chain.from_iterable(prior_order))
            if cat_order != flat_prior:
                # If cat_order does not match prior_order, raise error
                raise ValueError(
                    f"cat_order implies an ordinal structure but differs from prior_order:\n"
                    f"cat_order: {cat_order}\nprior_order: {prior_order}"
                )


def _find_covs_to_remove(df: DataFrame, all_covs: set) -> set:
    """Find and remove covariates that are nearly identical (all or all-but-one value the same)

    Parameters
    ----------
    df
        Input dataframe
    all_covs
        Set of all covariates listed in settings file

    Returns
    -------
    set
        Set of covariates with low variability to remove from data and settings files
    """
    covs_to_remove = set()
    for col in all_covs:
        counts = df[col].value_counts()
        if counts.iloc[0] >= len(df[col]) - 1:
            covs_to_remove.add(col)
    return covs_to_remove

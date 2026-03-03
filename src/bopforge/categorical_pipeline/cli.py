import pathlib
import warnings

import numpy as np
from pplkit.io import IOManager

import bopforge.categorical_pipeline.functions as functions
from bopforge.base_pipeline import create_argument_parser, run_pipeline
from bopforge.utils import fill_dict, get_point_estimate_and_UIs

warnings.filterwarnings("ignore")


def pre_processing(result_dir: pathlib.Path) -> None:
    iom = IOManager(result=result_dir)
    name = iom["result"].name

    # load data
    df = iom.load(f"raw-{name}.csv", key="result")
    all_settings = iom.load("settings.yaml", key="result")

    # Preprocess covariates
    df, cov_settings = functions.covariate_preprocessing(df, all_settings)
    all_settings["cov_type"] = cov_settings["cov_type"]
    all_settings["select_bias_covs"] = cov_settings["select_bias_covs"]

    # specify reference category
    unique_cats, counts = np.unique(
        np.hstack([df.ref_risk_cat, df.alt_risk_cat]), return_counts=True
    )
    ref_cat = all_settings["fit_signal_model"]["cat_cov_model"]["ref_cat"]
    if ref_cat:
        ref_cat = ref_cat
    else:
        ref_cat = unique_cats[counts.argmax()]
    # store initial summary outputs
    summary = {
        "name": name,
        "risk_type": str(df.risk_type.values[0]),
        "ref_cat": ref_cat,
    }
    all_settings["fit_signal_model"]["cat_cov_model"]["ref_cat"] = ref_cat

    # Add design matrices for interacted model covariates to data
    df = functions.covariate_design_mat(df, all_settings)

    # Validate ordering constraints if categories are ordinal
    cat_order = all_settings["cat_order"]
    prior_order = all_settings["fit_signal_model"]["cat_cov_model"][
        "prior_order"
    ]
    # Check cat_order is complete
    functions._validate_cat_order(cat_order, unique_cats)
    # For ordinal categories, if prior_order is given validate it matches cat_order
    functions._validate_cat_order_prior_order_match(cat_order, prior_order)

    # save results
    iom.dump(df, f"{name}.csv", key="result")
    iom.dump(all_settings, "settings.yaml", key="result")
    iom.dump(summary, "summary.yaml", key="result")


def fit_signal_model(result_dir: pathlib.Path) -> None:
    """Fit signal model. This step involves, trimming, but does not use a mixed
    effect model. The goal is to get the strength of prior for the covariate
    selection step and identifying all the outliers. A summary file will be
    generated to store the results of signal model.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    pre_processing(result_dir)
    iom = IOManager(result=result_dir)
    name = iom["result"].name

    # load data
    df = iom.load(f"{name}.csv", key="result")

    # load settings and summary
    all_settings = iom.load("settings.yaml", key="result")
    summary = iom.load("summary.yaml", key="result")

    signal_model = functions.get_signal_model(df, all_settings, summary)
    settings = all_settings["fit_signal_model"]
    default_signal_model_fit_model = {
        "outer_step_size": 20.0,
        "outer_max_iter": 100,
        "inner_options": {"gtol": 1e-6, "xtol": 1e-6},
    }
    signal_model.fit_model(
        **fill_dict(
            settings.get("signal_model_fit_model", {}),
            default_signal_model_fit_model,
        )
    )

    df = functions.add_cols(df, signal_model)

    # save results
    iom.dump(df, f"{name}.csv", key="result")
    iom.dump(signal_model, "signal_model.pkl", key="result")
    iom.dump(summary, "summary.yaml", key="result")


def select_bias_covs(result_dir: pathlib.Path) -> None:
    """Select the bias covariates. In this step, we first fit a linear model to
    get the prior strength of the bias-covariates. And then we use `CovFinder`
    to select important bias-covariates. A summary of the result will be
    generated and store in file `cov_finder_result.yaml`.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    iom = IOManager(result=result_dir)
    name = iom["result"].name

    df = iom.load(f"{name}.csv", key="result")

    all_settings = iom.load("settings.yaml", key="result")
    settings = all_settings["select_bias_covs"]

    cov_finder_linear_model = iom.load("signal_model.pkl", key="result")

    cov_finder = functions.get_cov_finder(
        df, all_settings, settings, cov_finder_linear_model
    )
    cov_finder.select_covs(verbose=True)

    cov_finder_result = functions.get_cov_finder_result(
        cov_finder_linear_model, cov_finder
    )

    iom.dump(cov_finder_result, "cov_finder_result.yaml", key="result")
    iom.dump(
        cov_finder_linear_model, "cov_finder_linear_model.pkl", key="result"
    )
    iom.dump(cov_finder, "cov_finder.pkl", key="result")


def fit_linear_model(result_dir: pathlib.Path) -> None:
    """Fit the final linear mixed effect model for the process. We will fit the
    linear model using selected bias covariates in this step. And we will create
    draws and quantiles for the effects. A single panels figure will be plotted
    to show the fit and all the important result information is documented in
    the `summary.yaml` file.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    iom = IOManager(result=result_dir)
    name = iom["result"].name

    df = iom.load(f"{name}.csv", key="result")
    df_train = df[df.is_outlier == 0].copy()

    cov_finder_result = iom.load("cov_finder_result.yaml", key="result")
    all_settings = iom.load("settings.yaml", key="result")
    settings = all_settings["complete_summary"]
    summary = iom.load("summary.yaml", key="result")

    linear_model = functions.get_linear_model(
        df_train, all_settings, cov_finder_result
    )
    linear_model.fit_model()

    cat_coefs = functions.get_cat_coefs(linear_model)
    pair_coefs = functions.get_pair_info(
        all_settings, summary, cat_coefs, linear_model
    )

    summary = functions.get_linear_model_summary(
        df, all_settings, settings, summary, cat_coefs, pair_coefs
    )

    df_cleaned = df.loc[:, ~df.columns.str.startswith("interacted_")]
    df_inner_draws, df_outer_draws = functions.get_draws(settings, pair_coefs)
    df_inner_quantiles, df_outer_quantiles = functions.get_quantiles(
        settings, pair_coefs
    )

    df_summary = get_point_estimate_and_UIs(
        df_inner_quantiles, df_outer_quantiles
    )

    fig = functions.plot_linear_model(
        df,
        name,
        summary,
        cat_coefs,
        pair_coefs,
    )

    iom.dump(linear_model, "linear_model.pkl", key="result")
    iom.dump(summary, "summary.yaml", key="result")
    iom.dump(df_cleaned, f"{name}.csv", key="result")
    iom.dump(df_inner_draws, "inner_draws.csv", key="result")
    iom.dump(df_outer_draws, "outer_draws.csv", key="result")
    iom.dump(df_inner_quantiles, "inner_quantiles.csv", key="result")
    iom.dump(df_outer_quantiles, "outer_quantiles.csv", key="result")
    iom.dump(df_summary, "summary_estimates.csv", key="result")
    iom.dump(cat_coefs, "cat_coefs.csv", key="result")
    iom.dump(pair_coefs, "pair_coefs.csv", na_rep="NaN", key="result")
    fig.savefig(iom["result"] / "linear_model.pdf", bbox_inches="tight")
    cat_order = all_settings["cat_order"]
    if not cat_order:
        fig_panel = functions.plot_linear_panel_model(df, cat_coefs, pair_coefs)
        fig_panel.savefig(
            iom["result"] / "linear_panel_model.pdf", bbox_inches="tight"
        )


ACTION_REGISTRY = {
    "fit_signal_model": fit_signal_model,
    "select_bias_covs": select_bias_covs,
    "fit_linear_model": fit_linear_model,
}


def main(args=None) -> None:
    parser = create_argument_parser(
        "Categorical burden of proof pipeline.",
        actions=list(ACTION_REGISTRY),
    )
    args = parser.parse_args(args)
    run_pipeline(
        args.input,
        args.output,
        args.pairs,
        args.actions,
        args.metadata,
        ACTION_REGISTRY,
    )


if __name__ == "__main__":
    main()

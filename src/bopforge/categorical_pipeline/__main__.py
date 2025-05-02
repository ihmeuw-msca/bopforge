import os
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pplkit.data.interface import DataInterface

import bopforge.categorical_pipeline.functions as functions
from bopforge.utils import ParseKwargs, fill_dict

warnings.filterwarnings("ignore")


def pre_processing(result_folder: Path) -> None:
    dataif = DataInterface(result=result_folder)
    name = dataif.result.name

    # load data
    df = dataif.load_result(f"raw-{name}.csv")
    all_settings = dataif.load_result("settings.yaml")

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

    # save results
    dataif.dump_result(df, f"{name}.csv")
    dataif.dump_result(all_settings, "settings.yaml")
    dataif.dump_result(summary, "summary.yaml")


def fit_signal_model(result_folder: Path) -> None:
    """Fit signal model. This step involves, trimming, but does not use a mixed
    effect model. The goal is to get the strength of prior for the covariate
    selection step and identifying all the outliers. A summary file will be
    generated to store the results of signal model.

    Parameters
    ----------
    dataif
        Data interface in charge of file reading and writing.

    """
    pre_processing(result_folder)
    dataif = DataInterface(result=result_folder)
    name = dataif.result.name

    # load data
    df = dataif.load_result(f"{name}.csv")

    # load settings and summary
    all_settings = dataif.load_result("settings.yaml")
    summary = dataif.load_result("summary.yaml")

    signal_model = functions.get_signal_model(df, all_settings, summary)
    signal_model.fit_model(outer_step_size=200, outer_max_iter=100)

    df = functions.add_cols(df, signal_model)

    # save results
    dataif.dump_result(df, f"{name}.csv")
    dataif.dump_result(signal_model, "signal_model.pkl")
    dataif.dump_result(summary, "summary.yaml")


def select_bias_covs(result_folder: Path) -> None:
    """Select the bias covariates. In this step, we first fit a linear model to
    get the prior strength of the bias-covariates. And then we use `CovFinder`
    to select important bias-covariates. A summary of the result will be
    generated and store in file `cov_finder_result.yaml`.

    Parameters
    ----------
    dataif
        Data interface in charge of file reading and writing.

    """
    dataif = DataInterface(result=result_folder)
    name = dataif.result.name

    df = dataif.load_result(f"{name}.csv")

    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["select_bias_covs"]

    cov_finder_linear_model = dataif.load_result("signal_model.pkl")

    cov_finder = functions.get_cov_finder(
        df, all_settings, settings, cov_finder_linear_model
    )
    cov_finder.select_covs(verbose=True)

    cov_finder_result = functions.get_cov_finder_result(
        cov_finder_linear_model, cov_finder
    )

    dataif.dump_result(cov_finder_result, "cov_finder_result.yaml")
    dataif.dump_result(cov_finder_linear_model, "cov_finder_linear_model.pkl")
    dataif.dump_result(cov_finder, "cov_finder.pkl")


def fit_linear_model(result_folder: Path) -> None:
    """Fit the final linear mixed effect model for the process. We will fit the
    linear model using selected bias covariates in this step. And we will create
    draws and quantiles for the effects. A single panels figure will be plotted
    to show the fit and all the important result information is documented in
    the `summary.yaml` file.

    Parameters
    ----------
    dataif
        Data interface in charge of file reading and writing.

    """
    dataif = DataInterface(result=result_folder)
    name = dataif.result.name

    df = dataif.load_result(f"{name}.csv")
    df_train = df[df.is_outlier == 0].copy()

    cov_finder_result = dataif.load_result("cov_finder_result.yaml")
    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["complete_summary"]
    summary = dataif.load_result("summary.yaml")

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

    fig = functions.plot_linear_model(
        df,
        name,
        summary,
        cat_coefs,
        pair_coefs,
    )

    dataif.dump_result(linear_model, "linear_model.pkl")
    dataif.dump_result(summary, "summary.yaml")
    dataif.dump_result(df_cleaned, f"{name}.csv")
    dataif.dump_result(df_inner_draws, "inner_draws.csv")
    dataif.dump_result(df_outer_draws, "outer_draws.csv")
    dataif.dump_result(df_inner_quantiles, "inner_quantiles.csv")
    dataif.dump_result(df_outer_quantiles, "outer_quantiles.csv")
    dataif.dump_result(cat_coefs, "cat_coefs.csv")
    dataif.dump_result(pair_coefs, "pair_coefs.csv", na_rep="NaN")
    fig.savefig(dataif.result / "linear_model.pdf", bbox_inches="tight")
    cat_order = all_settings["cat_order"]
    if not cat_order:
        fig_panel = functions.plot_linear_panel_model(df, cat_coefs, pair_coefs)
        fig_panel.savefig(
            dataif.result / "linear_panel_model.pdf", bbox_inches="tight"
        )


def run(
    i_dir: str,
    o_dir: str,
    pairs: list[str],
    actions: list[str],
    metadata: dict,
) -> None:
    i_dir, o_dir = Path(i_dir), Path(o_dir)
    # check the input and output folders
    if not i_dir.exists():
        raise FileNotFoundError("input data folder not found")

    o_dir.mkdir(parents=True, exist_ok=True)

    dataif = DataInterface(i_dir=i_dir, o_dir=o_dir)
    settings = dataif.load_i_dir("settings.yaml")

    # check pairs
    all_pairs = [pair for pair in settings.keys() if pair != "default"]
    pairs = pairs or all_pairs
    for pair in pairs:
        data_path = dataif.get_fpath(f"{pair}.csv", key="i_dir")
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file {data_path}")

    # check actions
    # TODO: might be good to use enum here
    all_actions = ["fit_signal_model", "select_bias_covs", "fit_linear_model"]
    actions = actions or all_actions
    invalid_actions = set(actions) - set(all_actions)
    if len(invalid_actions) != 0:
        raise ValueError(f"{list(invalid_actions)} are invalid actions")

    # fit each pair
    for pair in pairs:
        pair_o_dir = o_dir / pair
        pair_o_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(i_dir / f"{pair}.csv", pair_o_dir / f"raw-{pair}.csv")

        if pair not in settings:
            pair_settings = settings["default"]
        else:
            pair_settings = fill_dict(settings[pair], settings["default"])
        pair_settings["metadata"] = metadata
        dataif.dump_o_dir(pair_settings, pair, "settings.yaml")

        np.random.seed(pair_settings["seed"])
        for action in actions:
            globals()[action](pair_o_dir)


def main(args=None) -> None:
    parser = ArgumentParser(description="Categorical burden of proof pipeline.")
    parser.add_argument(
        "-i",
        "--input",
        type=os.path.abspath,
        required=True,
        help="Input data folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=os.path.abspath,
        required=True,
        help="Output result folder",
    )
    parser.add_argument(
        "-p",
        "--pairs",
        required=False,
        default=None,
        nargs="+",
        help="Included pairs, default all pairs",
    )
    parser.add_argument(
        "-a",
        "--actions",
        choices=["fit_signal_model", "select_bias_covs", "fit_linear_model"],
        default=None,
        nargs="+",
        help="Included actions, default all actions",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        nargs="*",
        required=False,
        default={},
        action=ParseKwargs,
        help="User defined metadata",
    )
    args = parser.parse_args(args)

    run(args.input, args.output, args.pairs, args.actions, args.metadata)


if __name__ == "__main__":
    main()

    # # get covariates that need to be removed
    # all_covs = [col for col in df.columns if col.startswith("cov_")]
    # covs_to_remove = [col for col in all_covs if len(df[col].unique()) == 1]

    # # remove from dataframe
    # df.drop(columns=covs_to_remove, inplace=True)

    # # remove from settings
    # all_covs = set(all_covs)
    # covs_to_remove = set(covs_to_remove)
    # subset_covs = all_covs - covs_to_remove
    # # pre-selected bias covariates
    # pre_selected_covs = set(settings["pre_selected_covs"])
    # pre_selected_covs = pre_selected_covs & all_covs
    # pre_selected_covs = pre_selected_covs - covs_to_remove
    # # model covariates
    # interacted_covs = set(model_cov_settings["interacted_covs"])
    # non_interacted_covs = set(model_cov_settings["non_interacted_covs"])
    # interacted_covs = interacted_covs & all_covs
    # non_interacted_covs = non_interacted_covs & all_covs
    # interacted_covs = interacted_covs - covs_to_remove
    # non_interacted_covs = non_interacted_covs - covs_to_remove

    # # Separate model covariates from bias covariates
    # col_bias_covs = list(subset_covs - interacted_covs - non_interacted_covs)

    # settings["pre_selected_covs"] = list(pre_selected_covs)
    # # settings["candidate_bias_covs"] = col_bias_covs
    # model_cov_settings["interacted_covs"] = list(interacted_covs)
    # model_cov_settings["non_interacted_covs"] = list(non_interacted_covs)
    # model_cov_settings["candidate_bias_covs"] = col_bias_covs
    # all_settings["select_bias_covs"]["cov_finder"] = settings
    # all_settings["select_bias_covs"]["model_covs"] = model_cov_settings


# # Create design matrices for interacted covariates
# cats = np.unique(df[["ref_risk_cat", "alt_risk_cat"]].to_numpy().ravel())
# alt_cats_mat = pd.get_dummies(df["alt_risk_cat"], drop_first=False).astype(
#     float
# )
# ref_cats_mat = pd.get_dummies(df["ref_risk_cat"], drop_first=False).astype(
#     float
# )
# for cat in cats:
#     if cat not in alt_cats_mat:
#         alt_cats_mat[cat] = 0.0
#     if cat not in ref_cats_mat:
#         ref_cats_mat[cat] = 0.0
# alt_cats_mat = alt_cats_mat[cats]
# ref_cats_mat = ref_cats_mat[cats]
# design = alt_cats_mat - ref_cats_mat
# model_covs = list(interacted_covs)
# design_matrices = {}
# for cov_name in model_covs:
#     cov_name_key = f"{cov_name}_design"
#     cov_design = design.copy()
#     cat_name_temp = [
#         f"model_{cov_name}_{col}" for col in cov_design.columns
#     ]
#     cov_design.columns = cat_name_temp
#     cov_design[:] = cov_design.to_numpy() * df[cov_name].to_numpy()[:, None]
#     cov_design[cov_design == -0.0] = 0.0
#     design_matrices[cov_name_key] = cov_design

# # Append model covariate design matrices to dataframe
# for cov_name, cov_design in design_matrices.items():
#     df = pd.concat([df, cov_design], axis=1)

import warnings
from pathlib import Path

from pplkit.data.interface import DataInterface

import bopforge.continuous_pipeline.functions as functions
from bopforge.base_pipeline import create_argument_parser, run_pipeline
from bopforge.utils import fill_dict, get_point_estimate_and_UIs

warnings.filterwarnings("ignore")


def pre_processing(result_dir: Path) -> None:
    dataif = DataInterface(result=result_dir)
    name = dataif.result.name
    df = dataif.load_result(f"raw-{name}.csv")
    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["select_bias_covs"]["cov_finder"]

    # get bias covariates that need to be removed
    all_covs = [col for col in df.columns if col.startswith("cov_")]
    covs_to_remove = [col for col in all_covs if len(df[col].unique()) == 1]

    # remove from dataframe
    df.drop(columns=covs_to_remove, inplace=True)

    # remove from settings
    all_covs = set("em" + col[3:] for col in all_covs)
    covs_to_remove = set("em" + col[3:] for col in covs_to_remove)
    pre_selected_covs = set(settings["pre_selected_covs"])
    pre_selected_covs = pre_selected_covs & all_covs
    pre_selected_covs = pre_selected_covs - covs_to_remove

    settings["pre_selected_covs"] = list(pre_selected_covs)
    all_settings["select_bias_covs"]["cov_finder"] = settings

    # save results
    dataif.dump_result(df, f"{name}.csv")
    dataif.dump_result(all_settings, "settings.yaml")


def fit_signal_model(result_dir: Path) -> None:
    """Fit signal model. This step involves, non-linear curve fitting and
    trimming, but does not use a mixed effect model. A single panel plot will be
    created for vetting the fit of the signal and a summary file will be
    generated to store the results of signal model.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    pre_processing(result_dir)
    dataif = DataInterface(result=result_dir)
    name = dataif.result.name

    df = dataif.load_result(f"{name}.csv")

    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["fit_signal_model"]

    signal_model = functions.get_signal_model(settings, df)
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

    df = functions.convert_bc_to_em(df, signal_model)

    summary = functions.get_signal_model_summary(name, all_settings, df)

    fig = functions.plot_signal_model(
        name,
        summary,
        df,
        signal_model,
        show_ref=all_settings["figure"]["show_ref"],
    )

    dataif.dump_result(df, f"{name}.csv")
    dataif.dump_result(signal_model, "signal_model.pkl")
    dataif.dump_result(summary, "summary.yaml")
    fig.savefig(dataif.result / "signal_model.pdf", bbox_inches="tight")


def select_bias_covs(result_dir: Path) -> None:
    """Select the bias covariates. In this step, we first fit a linear model to
    get the prior strength of the bias-covariates. And then we use `CovFinder`
    to select important bias-covariates. A summary of the result will be
    generated and store in file `cov_finder_result.yaml`.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    dataif = DataInterface(result=result_dir)
    name = dataif.result.name

    df = dataif.load_result(f"{name}.csv")
    df = df[df.is_outlier == 0].copy()

    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["select_bias_covs"]

    cov_finder_linear_model = functions.get_cov_finder_linear_model(df)
    cov_finder_linear_model.fit_model()

    cov_finder = functions.get_cov_finder(settings, cov_finder_linear_model)
    cov_finder.select_covs(verbose=True)

    cov_finder_result = functions.get_cov_finder_result(
        cov_finder_linear_model, cov_finder
    )

    # save results
    dataif.dump_result(cov_finder_result, "cov_finder_result.yaml")
    dataif.dump_result(cov_finder_linear_model, "cov_finder_linear_model.pkl")
    dataif.dump_result(cov_finder, "cov_finder.pkl")


def fit_linear_model(result_dir: Path) -> None:
    """Fit the final linear mixed effect model for the process. We will fit the
    linear model using the signal and selected bias covariates in this step.
    And we will create draws and quantiles for the risk curve. A two panels
    figure will be plotted to show the fit and all the important result
    information is documented in the `summary.yaml` file.

    Parameters
    ----------
    result_dir
        Path to the pair's output directory.

    """
    dataif = DataInterface(result=result_dir)
    name = dataif.result.name

    df = dataif.load_result(f"{name}.csv")
    df_train = df[df.is_outlier == 0].copy()

    cov_finder_result = dataif.load_result("cov_finder_result.yaml")
    all_settings = dataif.load_result("settings.yaml")
    settings = all_settings["complete_summary"]
    summary = dataif.load_result("summary.yaml")
    signal_model = dataif.load_result("signal_model.pkl")

    linear_model = functions.get_linear_model(df_train, cov_finder_result)
    linear_model.fit_model()

    summary = functions.get_linear_model_summary(
        settings,
        summary,
        df,
        signal_model,
        linear_model,
    )

    df_inner_draws, df_outer_draws = functions.get_draws(
        settings, summary, signal_model
    )

    df_inner_quantiles, df_outer_quantiles = functions.get_quantiles(
        settings, summary, signal_model
    )

    df_summary = get_point_estimate_and_UIs(
        df_inner_quantiles, df_outer_quantiles
    )

    fig = functions.plot_linear_model(
        name,
        summary,
        df,
        signal_model,
        linear_model,
        show_ref=all_settings["figure"]["show_ref"],
    )

    dataif.dump_result(linear_model, "linear_model.pkl")
    dataif.dump_result(summary, "summary.yaml")
    dataif.dump_result(df_inner_draws, "inner_draws.csv")
    dataif.dump_result(df_outer_draws, "outer_draws.csv")
    dataif.dump_result(df_inner_quantiles, "inner_quantiles.csv")
    dataif.dump_result(df_outer_quantiles, "outer_quantiles.csv")
    dataif.dump_result(df_summary, "summary_estimates.csv")
    fig.savefig(dataif.result / "linear_model.pdf", bbox_inches="tight")


ACTION_REGISTRY = {
    "fit_signal_model": fit_signal_model,
    "select_bias_covs": select_bias_covs,
    "fit_linear_model": fit_linear_model,
}


def main(args=None) -> None:
    parser = create_argument_parser(
        "Continuous burden of proof pipeline.",
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

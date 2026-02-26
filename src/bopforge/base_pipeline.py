"""Shared boilerplate for the continuous, dichotomous, and categorical
burden-of-proof pipelines."""

import argparse
import enum
import os
import pathlib
import shutil
import sys
import traceback
import typing
import warnings

import numpy as np
from pplkit.data.interface import DataInterface

from bopforge.utils import ParseKwargs, fill_dict

warnings.filterwarnings("ignore")


class Style(enum.Enum):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def styled(msg: str, *styles: Style) -> str:
    if not sys.stdout.isatty() or not styles:
        return msg
    prefix = "".join(s.value for s in styles)
    return f"{prefix}{msg}{Style.RESET.value}"


def banner(msg: str, *styles: Style, fill: str = "=") -> str:
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    text = f"{msg}".center(width, fill)
    if Style.BOLD not in styles:
        styles = (*styles, Style.BOLD)
    return styled(text, *styles)


def create_argument_parser(
    description: str, actions: list[str]
) -> argparse.ArgumentParser:
    """Build the CLI argument parser used by all pipelines.

    Parameters
    ----------
    description
        One-line description shown in ``--help`` output, e.g.
        ``"Continuous burden of proof pipeline."``.

    Returns
    -------
    ArgumentParser
        Fully configured argument parser.

    """
    parser = argparse.ArgumentParser(description=description)
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
        choices=actions,
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
    return parser


def run_pipeline(
    i_dir: str | pathlib.Path,
    o_dir: str | pathlib.Path,
    pairs: list[str] | None,
    actions: list[str] | None,
    metadata: dict,
    action_registry: dict[str, typing.Callable[[pathlib.Path], None]],
) -> None:
    """Execute a burden-of-proof pipeline.

    This is the generalised version of the per-pipeline ``run()`` functions.
    It handles directory validation, pair iteration, settings loading,
    action dispatching, and error reporting.

    The error-handling strategy is adopted from the continuous pipeline:
    failures during per-pair setup or individual actions are caught, recorded,
    and reported in a final summary rather than aborting the entire run.

    Parameters
    ----------
    i_dir
        pathlib.Path to the input data folder.
    o_dir
        pathlib.Path to the output results folder (created if it doesn't exist).
    pairs
        List of pair names to process.  ``None`` or empty → all pairs found
        in ``settings.yaml``.
    actions
        List of action names to run.  ``None`` or empty → all actions in
        ``action_registry``.
    metadata
        Arbitrary user-supplied key/value metadata attached to every pair's
        settings.
    action_registry
        Mapping from action name (e.g. ``"fit_signal_model"``) to the
        callable that implements it.  Each callable must accept a single
        :class:`~pathlib.pathlib.Path` argument pointing to the pair's output
        directory.

    """
    i_path, o_path = pathlib.Path(i_dir), pathlib.Path(o_dir)

    if not i_path.exists():
        raise FileNotFoundError("input data folder not found")

    o_path.mkdir(parents=True, exist_ok=True)

    dataif = DataInterface(i_dir=i_path, o_dir=o_path)
    settings = dataif.load_i_dir("settings.yaml")

    all_pairs = [pair for pair in settings.keys() if pair != "default"]
    valid_pairs = pairs or all_pairs
    for pair in valid_pairs:
        data_path = dataif.get_fpath(f"{pair}.csv", key="i_dir")
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file {data_path}")

    all_actions = list(action_registry.keys())
    actions = actions or all_actions
    inactions = set(actions) - set(all_actions)
    if inactions:
        raise ValueError(f"{list(inactions)} are invalid actions")

    failed_pairs: list[dict] = []

    for pair in valid_pairs:
        print("\n" + banner(f" MODELING PAIR: {pair} "))

        pair_o_dir = o_path / pair
        pair_o_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy(i_path / f"{pair}.csv", pair_o_dir / f"raw-{pair}.csv")

            if pair not in settings:
                pair_settings = settings["default"]
            else:
                pair_settings = fill_dict(settings[pair], settings["default"])
            pair_settings["metadata"] = metadata
            dataif.dump_o_dir(pair_settings, pair, "settings.yaml")

            np.random.seed(pair_settings["seed"])

            for action in actions:
                print(styled(f"  > Running action: {action}..."))
                try:
                    action_registry[action](pair_o_dir)
                    print(
                        styled(f"    [SUCCESS] Finished {action}.", Style.GREEN)
                    )
                except Exception as e:
                    tb_str = traceback.format_exc()
                    print(
                        "\n" + banner(" FAILURE ", Style.RED, fill="!"),
                        file=sys.stderr,
                    )
                    print(styled(f"Pair: {pair}", Style.RED), file=sys.stderr)
                    print(
                        styled(f"Action: {action}", Style.RED), file=sys.stderr
                    )
                    print(
                        styled(f"Error Type: {type(e).__name__}", Style.RED),
                        file=sys.stderr,
                    )
                    print(
                        styled(f"Error Details: {str(e)}", Style.RED),
                        file=sys.stderr,
                    )
                    failed_pairs.append(
                        {
                            "pair": pair,
                            "action": action,
                            "error": str(e),
                            "traceback": tb_str,
                        }
                    )
                    # Stop processing remaining actions for this pair
                    break
        except Exception as e:
            # Catching issues that happen before actions
            # (e.g., file copying or settings merging)
            print(
                styled(
                    f"An error occurred during setup for pair '{pair}': {e}",
                    Style.RED,
                ),
                file=sys.stderr,
            )
            failed_pairs.append(
                {"pair": pair, "action": "setup", "error": str(e)}
            )
            continue

    num_success = len(valid_pairs) - len(failed_pairs)
    summary_style = Style.GREEN if not failed_pairs else Style.YELLOW

    print(
        "\n" + banner(" PIPELINE EXECUTION SUMMARY ", summary_style, fill="#")
    )
    print(f"  Total pairs processed:  {len(valid_pairs)}")
    print(styled(f"  Successfully completed: {num_success}", Style.GREEN))
    print(
        styled(
            f"  Failed/Skipped:         {len(failed_pairs)}",
            Style.RED if failed_pairs else Style.GREEN,
        )
    )

    if failed_pairs:
        err_msg = "\n".join(
            ["Stage of Failures:"]
            + [
                f" - {failure['pair']} at {failure['action']}"
                for failure in failed_pairs
            ]
        )
        print(banner("", summary_style, fill="#") + "\n")
        raise RuntimeError(err_msg)

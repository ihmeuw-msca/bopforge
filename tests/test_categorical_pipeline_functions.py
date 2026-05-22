import pytest

from bopforge.categorical_pipeline.functions import _resolve_prior_order


@pytest.mark.parametrize(
    "cat_order, prior_order, expected",
    [
        # bug case (MSCA-749): cat_order set but prior_order empty -> derive
        # prior_order from cat_order so downstream ordinal logic has a value
        (["a", "b", "c"], [], [["a", "b", "c"]]),
        # regression guard: an explicit prior_order must not be overwritten
        (["b", "a", "c"], [["a", "b", "c"]], [["a", "b", "c"]]),
        # non-ordinal risk: both empty, nothing to derive
        ([], [], []),
    ],
)
def test_resolve_prior_order(cat_order, prior_order, expected):
    assert _resolve_prior_order(cat_order, prior_order) == expected

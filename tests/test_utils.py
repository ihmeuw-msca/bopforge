import numpy as np
import pytest

from bopforge.utils import score_to_star_rating


@pytest.mark.parametrize(
    "score, expected_star",
    [
        (float("nan"), 0),
        (-0.1, 1),
        (0.0, 1),
        (0.001, 2),
        (0.1, 2),
        (np.log(1.15), 2),
        (np.log(1.16), 3),
        (0.3, 3),
        (np.log(1.5), 3),
        (np.log(1.51), 4),
        (0.5, 4),
        (np.log(1.85), 4),
        (np.log(1.86), 5),
        (1.0, 5),
        (10.0, 5),
    ],
)
def test_score_to_star_rating(score, expected_star):
    assert score_to_star_rating(score) == expected_star

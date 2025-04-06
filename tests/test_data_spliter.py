"""Test module for data.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_spliter import create_examples_for_machine


@pytest.fixture
def small_df() -> pd.DataFrame:
    """DataFrame for basic tests.

    index: 0  1  2  3  4  5  6  7  8  9
    failure_flag: 0  0  0  1  0  0  1  0  0  0
    feature1: range(10)
    feature2: random floats
    """
    data = {
        "failure_flag": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "feature1": list(range(10)),
        "feature2": np.random.randn(10),
    }
    return pd.DataFrame(data)


def test_basic_functionality(small_df: pd.DataFrame) -> None:
    """Check that the function returns the expected number of examples."""
    seq_len = 3
    horizont = 0

    examples = create_examples_for_machine(small_df, seq_len, horizont)

    assert len(examples) == 4, f"Expected 4 examples, got {len(examples)}"

    # Check the first positive window
    pos_window, pos_label = examples[0]
    assert pos_label == 1, "Expected label=1 for the first window"
    assert list(pos_window.index) == [
        0,
        1,
        2,
    ], "The first positive window has incorrect indices"
    assert len(pos_window) == 3, "Window length should match seq_len"

    # Check the second positive window
    pos_window2, pos_label2 = examples[1]
    assert pos_label2 == 1, "Second window is also positive"
    assert list(pos_window2.index) == [
        3,
        4,
        5,
    ], "Second positive window indices don't match expected"
    assert len(pos_window2) == 3, "Window length should match seq_len"


def test_no_failures() -> None:
    """The function should return 0 examples if there are no failures in the data."""
    df = pd.DataFrame({"failure_flag": [0, 0, 0, 0], "feature1": [10, 20, 30, 40]})
    seq_len = 2
    horizont = 0

    examples = create_examples_for_machine(df, seq_len, horizont)

    assert len(examples) == 0, "Expected 0 examples when no failures are present"


def test_not_enough_negatives(small_df: pd.DataFrame) -> None:
    """Force a scenario with not enough negative rows to match the number of positives."""
    df_reduced = small_df.iloc[[2, 3, 6]].copy()
    # fix the index to [0,1,2] for clarity
    df_reduced.reset_index(drop=True, inplace=True)
    # Now df_reduced has:
    #   index: 0    1    2
    #   fail:  0    1    1
    # Only one non-failure row (index=0).
    seq_len = 1
    horizont = 0

    examples = create_examples_for_machine(df_reduced, seq_len, horizont)
    assert len(examples) == 2, f"Expected 2 examples, got {len(examples)}"

    # Confirm that 2 are positive, 0 is negative
    pos_count = sum(label for (_, label) in examples)
    neg_count = len(examples) - pos_count
    assert pos_count == 2, f"Expected 2 positives, got {pos_count}"
    assert neg_count == 0, f"Expected 0 negative, got {neg_count}"


def test_random_seed_reproducibility(small_df: pd.DataFrame) -> None:
    """Ensure that specifying a random_state leads to consistent negative window selection."""
    seq_len = 3
    rstate = 42
    horizont = 0

    examples_run1 = create_examples_for_machine(
        small_df, seq_len, horizont, random_state=rstate
    )
    examples_run2 = create_examples_for_machine(
        small_df, seq_len, horizont, random_state=rstate
    )

    # The negative windows should be identical in order and content
    negative_windows_run1 = [win for (win, label) in examples_run1 if label == 0]
    negative_windows_run2 = [win for (win, label) in examples_run2 if label == 0]

    assert len(negative_windows_run1) == len(
        negative_windows_run2
    ), "Number of negative windows differs between runs"
    for w1, w2 in zip(negative_windows_run1, negative_windows_run2):
        assert (
            w1.index == w2.index
        ).all(), "Negative window indices differ under same random seed"


@pytest.mark.parametrize("horizon", [0, 1, 2])
def test_horizon_prediction(small_df: pd.DataFrame, horizon: int) -> None:
    """Verify that positive windows end exactly `horizon` steps before failure index.

    - If a window would start at a negative index, it should not exist.
    """
    seq_len = 2
    failure_indices = [3, 6]
    examples = create_examples_for_machine(small_df, seq_len, horizon)

    positives = [(win, int(lbl)) for (win, lbl) in examples if lbl == 1]
    negatives = [(win, int(lbl)) for (win, lbl) in examples if lbl == 0]

    _check_positive_windows(positives, failure_indices, seq_len, horizon)
    _check_negative_windows(negatives, seq_len)


def _check_positive_windows(
    positives: list[tuple[pd.DataFrame, int]],
    failure_indices: list[int],
    seq_len: int,
    horizon: int,
) -> None:
    """Check that positive windows are correctly aligned with failure indices."""
    for fi in failure_indices:
        end_idx = fi - horizon
        start_idx = end_idx - seq_len

        found = any(
            list(win.index) == list(range(start_idx, end_idx))
            for win, lbl in positives
            if lbl == 1 and len(win) == seq_len
        )

        if start_idx < 0:
            assert not found, (
                f"Unexpected window found for failure at index {fi} "
                f"with horizon {horizon} (start_idx < 0)"
            )
        else:
            assert found, (
                f"Expected window not found for failure at index {fi} "
                f"with horizon {horizon}"
            )


def _check_negative_windows(
    negatives: list[tuple[pd.DataFrame, int]], seq_len: int
) -> None:
    """Check that negative windows are of the expected length."""
    for win, lbl in negatives:
        assert lbl == 0, "Negative label expected"
        assert len(win) == seq_len, "Negative window should match seq_len"

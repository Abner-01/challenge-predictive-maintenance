import pytest
import pandas as pd
from src.data import create_examples_for_machine
import numpy as np


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
    examples = create_examples_for_machine(small_df, seq_len)

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
    examples = create_examples_for_machine(df, seq_len=2)

    assert len(examples) == 0, "Expected 0 examples when no failures are present"


def test_not_enough_negatives(small_df) -> None:
    """Force a scenario with not enough negative rows to match the number of positives."""
    df_reduced = small_df.iloc[[2, 3, 6]].copy()
    # fix the index to [0,1,2] for clarity
    df_reduced.reset_index(drop=True, inplace=True)
    # Now df_reduced has:
    #   index: 0    1    2
    #   fail:  0    1    1
    # Only one non-failure row (index=0).

    examples = create_examples_for_machine(df_reduced, seq_len=1)
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

    examples_run1 = create_examples_for_machine(small_df, seq_len, random_state=rstate)
    examples_run2 = create_examples_for_machine(small_df, seq_len, random_state=rstate)

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

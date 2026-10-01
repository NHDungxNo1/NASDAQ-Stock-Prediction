import numpy as np
import pytest

from src.validation import walk_forward_splits


def test_train_always_comes_before_test():
    for train, test in walk_forward_splits(100, initial_train=50, step=10):
        assert train.max() < test.min()


def test_every_row_after_the_first_window_is_forecast_once():
    tested = np.concatenate([te for _, te in walk_forward_splits(103, 50, 10)])
    np.testing.assert_array_equal(tested, np.arange(50, 103))


def test_expanding_vs_sliding_window():
    expanding = [len(tr) for tr, _ in walk_forward_splits(100, 50, 10, expanding=True)]
    sliding = [len(tr) for tr, _ in walk_forward_splits(100, 50, 10, expanding=False)]
    assert expanding == [50, 60, 70, 80, 90]
    assert sliding == [50] * 5


def test_rejects_window_larger_than_data():
    with pytest.raises(ValueError):
        list(walk_forward_splits(10, initial_train=10))

"""Tests for prolif.parallel module."""

from unittest.mock import MagicMock, patch

import pytest

from prolif.parallel import (
    MAX_JOBS,
    MDA_PARALLEL_STRATEGY_THRESHOLD,
    get_mda_parallel_strategy,
    get_n_jobs,
)


class TestGetNJobs:
    """Tests for get_n_jobs function."""

    def test_explicit_n_jobs_returned(self) -> None:
        """When n_jobs is explicitly set, return that value."""
        assert get_n_jobs(4) == 4

    def test_invalid_n_jobs_raises_error(self) -> None:
        """When n_jobs <= 0, raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be > 0"):
            get_n_jobs(0)
        with pytest.raises(ValueError, match="n_jobs must be > 0"):
            get_n_jobs(-1)

    def test_env_variable_used_when_n_jobs_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When n_jobs is None, use PROLIF_N_JOBS env variable."""
        monkeypatch.setenv("PROLIF_N_JOBS", "6")
        assert get_n_jobs() == 6

    def test_cpu_count_used_when_no_env(self) -> None:
        """When n_jobs and env are not set, use cpu_count (capped at MAX_JOBS)."""
        with patch("prolif.parallel.psutil.cpu_count", return_value=16):
            assert get_n_jobs(capped=True) == MAX_JOBS

    def test_returns_none_when_cpu_count_fails(self) -> None:
        """When cpu_count returns None, get_n_jobs returns None."""
        with patch("prolif.parallel.psutil.cpu_count", return_value=None):
            assert get_n_jobs() is None


class TestGetMdaParallelStrategy:
    """Tests for get_mda_parallel_strategy function."""

    def test_explicit_strategy_returned(self) -> None:
        """When strategy is explicitly set, return that value."""
        mock_traj = MagicMock()
        assert get_mda_parallel_strategy("chunk", mock_traj) == "chunk"
        assert get_mda_parallel_strategy("queue", mock_traj) == "queue"

    def test_small_pickle_uses_chunk(self) -> None:
        """When trajectory pickle is small, use 'chunk' strategy."""
        mock_traj = MagicMock()
        # Mock dill.dumps to return a small pickle
        small_pickle = b"x" * (MDA_PARALLEL_STRATEGY_THRESHOLD - 1)
        with patch("prolif.parallel.dill.dumps", return_value=small_pickle):
            assert get_mda_parallel_strategy(None, mock_traj) == "chunk"

    def test_large_pickle_uses_queue(self) -> None:
        """When trajectory pickle is large, use 'queue' strategy."""
        mock_traj = MagicMock()
        # Mock dill.dumps to return a large pickle
        large_pickle = b"x" * (MDA_PARALLEL_STRATEGY_THRESHOLD + 1)
        with patch("prolif.parallel.dill.dumps", return_value=large_pickle):
            assert get_mda_parallel_strategy(None, mock_traj) == "queue"

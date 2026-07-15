"""
Unit test for the replay buffer flush mechanism used in CurriculumCallback.
Validates both the count AND identity of retained transitions.

Key invariant: after flush, the retained transitions must be the MOST RECENT
ones, not the oldest. The previous version of this test only checked counts
and missed a bug where the oldest data was kept instead.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.buffers import ReplayBuffer
from training.train_sac import CurriculumCallback
from gymnasium import spaces

# ─── Buffer attribute list (must match train_sac.py exactly) ───
BUFFER_ARRAYS = ['observations', 'next_observations', 'actions',
                 'rewards', 'dones', 'timeouts']


def _create_test_buffer(buffer_size: int, obs_dim: int = 12, action_dim: int = 2):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return ReplayBuffer(buffer_size, obs_space, action_space)


def _fill_buffer_with_markers(buf, n_transitions):
    """Fill buffer with transitions whose obs[0] = insertion order (0, 1, ..., N-1).
    This lets us verify WHICH transitions survive a flush."""
    for i in range(n_transitions):
        obs = np.zeros(buf.observation_space.shape, dtype=np.float32)
        obs[0] = float(i)  # marker = insertion order
        next_obs = np.zeros(buf.observation_space.shape, dtype=np.float32)
        next_obs[0] = float(i + 1)
        action = np.zeros(buf.action_space.shape, dtype=np.float32)
        reward = float(i)
        done = False
        info = [{}]
        buf.add(obs, next_obs, action, np.array([reward]), np.array([done]), info)


def _flush_buffer(buf, keep_fraction=CurriculumCallback.KEEP_FRACTION):
    """Exact copy of the production flush logic from CurriculumCallback._on_step()."""
    old_pos = buf.pos
    old_full = buf.full

    if old_full:
        keep_count = int(buf.buffer_size * keep_fraction)
        idx = (np.arange(old_pos - keep_count, old_pos) % buf.buffer_size)
    else:
        keep_count = int(old_pos * keep_fraction)
        idx = np.arange(old_pos - keep_count, old_pos)

    for attr_name in BUFFER_ARRAYS:
        if hasattr(buf, attr_name):
            arr = getattr(buf, attr_name)
            arr[:keep_count] = arr[idx]

    buf.pos = keep_count
    buf.full = False
    return keep_count


# ═══════════════════════════════════════════════════════════════
# TEST 1: Non-full buffer — correct count
# ═══════════════════════════════════════════════════════════════
def test_flush_count_nonfull():
    buf = _create_test_buffer(buffer_size=1000)
    _fill_buffer_with_markers(buf, 500)
    assert buf.pos == 500 and not buf.full

    kept = _flush_buffer(buf)

    assert kept == 100  # 20% of 500
    assert buf.pos == 100
    assert buf.full is False
    print(f"  [PASS] Non-full count: 500 -> kept {kept}")


# ═══════════════════════════════════════════════════════════════
# TEST 2: Non-full buffer — retained transitions are MOST RECENT
# ═══════════════════════════════════════════════════════════════
def test_flush_identity_nonfull():
    buf = _create_test_buffer(buffer_size=1000)
    _fill_buffer_with_markers(buf, 500)

    kept = _flush_buffer(buf)

    # The retained obs markers should be [400, 401, ..., 499] — the last 100
    retained_markers = buf.observations[:kept, 0, 0]  # shape is (buf_size, 1, obs_dim)
    expected = np.arange(400, 500, dtype=np.float32)

    assert np.array_equal(retained_markers, expected), \
        f"Expected markers [400..499], got [{retained_markers[0]}..{retained_markers[-1]}]"

    # Every retained marker must be > every discarded marker
    min_retained = retained_markers.min()
    assert min_retained == 400.0, f"Oldest retained should be 400, got {min_retained}"
    print(f"  [PASS] Non-full identity: retained markers [{int(retained_markers[0])}..{int(retained_markers[-1])}]")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Full/wrapped buffer — correct count
# ═══════════════════════════════════════════════════════════════
def test_flush_count_full():
    buf = _create_test_buffer(buffer_size=200)
    _fill_buffer_with_markers(buf, 350)  # overfill → wraps, full=True

    assert buf.full is True

    kept = _flush_buffer(buf)

    assert kept == 40  # 20% of buffer_size=200
    assert buf.pos == 40
    assert buf.full is False
    print(f"  [PASS] Full-buffer count: kept {kept}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: Full/wrapped buffer — retained transitions are MOST RECENT
# ═══════════════════════════════════════════════════════════════
def test_flush_identity_full():
    buf = _create_test_buffer(buffer_size=200)
    _fill_buffer_with_markers(buf, 350)  # writes 0..349, wraps at 200

    # After 350 inserts into a size-200 buffer:
    #   buf.pos = 350 % 200 = 150, buf.full = True
    #   Physical layout: indices [0..149] hold markers [200..349]
    #                    indices [150..199] hold markers [150..199]
    #   Most recent 40: markers [310..349]
    assert buf.pos == 150

    kept = _flush_buffer(buf)

    retained_markers = buf.observations[:kept, 0, 0]
    expected = np.arange(310, 350, dtype=np.float32)

    assert np.array_equal(retained_markers, expected), \
        f"Expected markers [310..349], got [{retained_markers[0]}..{retained_markers[-1]}]"
    print(f"  [PASS] Full-buffer identity: retained markers [{int(retained_markers[0])}..{int(retained_markers[-1])}]")


# ═══════════════════════════════════════════════════════════════
# TEST 5: Post-flush writes succeed
# ═══════════════════════════════════════════════════════════════
def test_post_flush_add():
    buf = _create_test_buffer(buffer_size=1000)
    _fill_buffer_with_markers(buf, 500)
    _flush_buffer(buf)

    # Add 50 more transitions
    for i in range(50):
        obs = np.full(buf.observation_space.shape, 9999.0 + i, dtype=np.float32)
        next_obs = np.full(buf.observation_space.shape, 10000.0 + i, dtype=np.float32)
        action = np.zeros(buf.action_space.shape, dtype=np.float32)
        buf.add(obs, next_obs, action, np.array([0.0]), np.array([False]), [{}])

    assert buf.pos == 150  # 100 kept + 50 new
    assert buf.full is False
    print(f"  [PASS] Post-flush add: pos={buf.pos} (100 kept + 50 new)")


# ═══════════════════════════════════════════════════════════════
# TEST 6: Post-flush sampling succeeds
# ═══════════════════════════════════════════════════════════════
def test_post_flush_sample():
    buf = _create_test_buffer(buffer_size=1000)
    _fill_buffer_with_markers(buf, 500)
    _flush_buffer(buf)

    # Add a few transitions so there's data to sample
    _fill_buffer_with_markers(buf, 10)

    batch = buf.sample(32)
    assert batch.observations.shape == (32, 12)
    assert batch.actions.shape == (32, 2)
    assert batch.rewards.shape == (32, 1)
    print(f"  [PASS] Post-flush sample: batch shape {batch.observations.shape}")


# ═══════════════════════════════════════════════════════════════
# TEST 7: All 6 buffer arrays are reordered consistently
# ═══════════════════════════════════════════════════════════════
def test_all_arrays_consistent():
    """After flush, obs/next_obs/actions/rewards/dones/timeouts should all
    correspond to the same transitions — no mismatched tuples."""
    buf = _create_test_buffer(buffer_size=1000)
    _fill_buffer_with_markers(buf, 500)  # reward[i] = float(i), obs[i][0] = float(i)

    kept = _flush_buffer(buf)

    # obs marker and reward should match for the same index
    for j in range(kept):
        obs_marker = buf.observations[j, 0, 0]
        reward_val = buf.rewards[j, 0]
        assert obs_marker == reward_val, \
            f"Mismatch at index {j}: obs marker={obs_marker}, reward={reward_val}"

    print(f"  [PASS] All arrays consistent: obs markers match rewards for all {kept} transitions")


if __name__ == "__main__":
    print("\n=== Buffer Flush Unit Tests (with identity checks) ===\n")
    test_flush_count_nonfull()
    test_flush_identity_nonfull()
    test_flush_count_full()
    test_flush_identity_full()
    test_post_flush_add()
    test_post_flush_sample()
    test_all_arrays_consistent()
    print("\n=== All 7 tests PASSED ===\n")

"""108. DPO Sequence Loss."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [{"batch": 1, "beta": 0.1}, {"batch": 11, "beta": 0.7}, {"batch": 73, "beta": 1.0}]


def make_inputs(case, rng):
    batch = case["batch"]
    beta = case["beta"]
    chosen = -np.abs(rng.standard_normal(batch, dtype=np.float32))
    rejected = -np.abs(rng.standard_normal(batch, dtype=np.float32))
    ref_chosen = -np.abs(rng.standard_normal(batch, dtype=np.float32))
    ref_rejected = -np.abs(rng.standard_normal(batch, dtype=np.float32))
    return [chosen, rejected, ref_chosen, ref_rejected], (1,), [np.float32(beta), np.uint32(batch)]


def reference_impl(chosen, rejected, ref_chosen, ref_rejected, beta, batch):
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return np.array([np.logaddexp(0, -margin).mean()], np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False

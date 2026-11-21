"""74. GPT-2 Transformer Block."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True

TEST_CASES = [{"seq_len": 1}, {"seq_len": 3}, {"seq_len": 7}]

WEIGHT_SHAPES = {
    "ln1": (768,),
    "lb1": (768,),
    "qkv": (768, 2304),
    "bqkv": (2304,),
    "att": (768, 768),
    "batt": (768,),
    "ln2": (768,),
    "lb2": (768,),
    "fc": (768, 3072),
    "bfc": (3072,),
    "proj": (3072, 768),
    "bproj": (768,),
}


def unpack(weights):
    tensors = {}
    offset = 0
    for name, shape in WEIGHT_SHAPES.items():
        count = int(np.prod(shape))
        tensors[name] = weights[offset : offset + count].reshape(shape)
        offset += count
    return tensors


def layer_norm(x, gain, bias):
    centered = x - x.mean(axis=1, keepdims=True)
    return centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True) + 1e-5) * gain + bias


def make_inputs(case, rng):
    seq_len = case["seq_len"]
    shape = (seq_len, 768)
    x = rng.uniform(-1, 1, shape).astype(np.float32)
    weights = rng.standard_normal(
        sum(int(np.prod(s)) for s in WEIGHT_SHAPES.values()), dtype=np.float32
    ) * np.float32(0.025)
    p = unpack(weights)

    p["ln1"] += 1
    p["ln2"] += 1
    return [x, weights], shape, [np.uint32(seq_len)]


def reference_impl(x, weights, seq_len):
    p = unpack(weights)

    norm = layer_norm(x, p["ln1"], p["lb1"])
    q, k, v = (
        t.reshape(seq_len, 12, 64).transpose(1, 0, 2)
        for t in np.split(norm @ p["qkv"] + p["bqkv"], 3, axis=-1)
    )

    scores = (q @ k.transpose(0, 2, 1)) * 0.125
    prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
    context = ((prob / prob.sum(axis=-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(x.shape)
    hidden = x + context @ p["att"] + p["batt"]
    fc = layer_norm(hidden, p["ln2"], p["lb2"]) @ p["fc"] + p["bfc"]
    gelu = 0.5 * fc * (1 + np.tanh(np.float32(np.sqrt(2 / np.pi)) * (fc + 0.044715 * fc**3)))
    return hidden + gelu @ p["proj"] + p["bproj"]


def dispatch_config(case):
    return 1, 1, 1, 128, 1, 1, True


def workspace_bytes(case):
    return 4 * case["seq_len"] * 6912

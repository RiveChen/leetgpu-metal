"""118. Vision Transformer Patch Embedding."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"batch": 1, "channels": 1, "height": 2, "width": 2, "patch": 2, "embed": 3},
    {"batch": 2, "channels": 3, "height": 8, "width": 12, "patch": 4, "embed": 11},
    {"batch": 2, "channels": 4, "height": 15, "width": 21, "patch": 3, "embed": 19},
]


def make_inputs(case, rng):
    batch = case["batch"]
    channels = case["channels"]
    height = case["height"]
    width = case["width"]
    patch = case["patch"]
    embed = case["embed"]
    images = rng.standard_normal((batch, channels, height, width), dtype=np.float32)
    weight = rng.standard_normal((embed, channels, patch, patch), dtype=np.float32)
    bias = rng.standard_normal(embed, dtype=np.float32)
    cls = rng.standard_normal(embed, dtype=np.float32)
    pos = rng.standard_normal((1 + (height // patch) * (width // patch), embed), dtype=np.float32)
    return (
        [images, weight, bias, cls, pos],
        (batch, *pos.shape),
        [np.uint32(v) for v in (batch, channels, height, width, patch, embed)],
    )


def reference_impl(images, weight, bias, cls, pos):
    b, c, h, w = images.shape
    p = weight.shape[-1]
    tiles = (
        images.reshape(b, c, h // p, p, w // p, p)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(b, -1, c * p * p)
    )
    embedded = tiles @ weight.reshape(len(cls), -1).T + bias
    prefix = np.broadcast_to(cls, (b, 1, len(cls)))
    return np.concatenate([prefix, embedded], axis=1) + pos


def dispatch_config(case):
    return (
        case["batch"]
        * (1 + (case["height"] // case["patch"]) * (case["width"] // case["patch"]))
        * case["embed"],
        1,
        1,
        64,
        1,
        1,
        False,
    )

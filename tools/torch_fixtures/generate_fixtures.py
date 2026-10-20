#!/usr/bin/env python3
"""
Generates reference fixtures from PyTorch for the MDNN parity tests.

WHY THIS EXISTS
---------------
All other tests in MDNN.Tests check the library AGAINST ITSELF: a gradient check
compares the analytic backward with the numerical derivative of the library's OWN forward.
If the forward follows a different convention (different padding, flipped kernel,
different loss scaling), the gradient check passes and the bug stays hidden.

This script produces an external oracle. It builds the same computations in PyTorch with
EXACTLY the same weights and inputs, exports the results to JSON, and the C# test then
compares the numbers element by element.

RUNNING (fixtures are committed, so this is only needed when they change):
    python3 -m venv .venv && .venv/bin/pip install torch numpy
    .venv/bin/python generate_fixtures.py

CI does NOT need Python; it only reads the finished .json files.

VERIFIED CONVENTIONS (why the comparison matches)
-------------------------------------------------
    MDNN MSE  = Σ(v−t)²       ≡ MSELoss(reduction='sum')
    MDNN CE   = −Σ t·ln(s)    ≡ CrossEntropyLoss(reduction='sum'), single sample
    MDNN Conv = cross-correlation without kernel flip ≡ torch Conv2d
    MDNN Adam = w − lr·m̂/(√v̂ + 1e-8)  ≡ torch.optim.Adam (eps OUTSIDE the sqrt)

STORAGE DIFFERENCES (transposed here, not in C#)
------------------------------------------------
    Dense  MDNN neuron[i].Weights[j]      ≡ torch Linear.weight[i][j]
    Conv   MDNN Kernels[f][kh][kw][c]     ≡ torch weight[f][c][kh][kw]
    Conv   MDNN input [H][W][C]           ≡ torch [N][C][H][W]
"""

import json
import os

import torch
import torch.nn.functional as F

# The reference must be in double; otherwise a difference against C# (also double) would
# measure float precision instead of a convention mismatch, which is what we want to measure.
torch.set_default_dtype(torch.float64)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "MDNN", "MDNN.Tests", "Fixtures",
)

ACTS = {
    "linear": lambda t: t,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


def deterministic(shape, scale=0.1, offset=0.03):
    """Spread-out values without an RNG: fixtures must be bit-stable across torch
    versions, otherwise the diff would change on its own."""
    n = 1
    for d in shape:
        n *= d
    vals = [((i * 37 % 23) - 11) * scale + offset for i in range(n)]
    return torch.tensor(vals, dtype=torch.float64).reshape(shape)


def nested(t):
    return t.detach().cpu().tolist()


def write(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  {name}.json")


# ---------------------------------------------------------------- Dense model

def dense_model(name, sizes, activations, loss_kind, weight_scale=0.07, worst_class=None):
    """Fully connected network: sizes = [input, hidden..., output].
    In MDNN it is built as new NeuralNetwork(output) + Layers.Add(hidden); the order
    in `layers` below matches model.Layers.Layers, i.e. hidden → output."""
    x = deterministic((sizes[0],), scale=0.11, offset=-0.05).requires_grad_(True)

    layers = []
    h = x
    for li in range(len(sizes) - 1):
        fan_in, fan_out = sizes[li], sizes[li + 1]
        W = deterministic((fan_out, fan_in), scale=weight_scale, offset=0.02).requires_grad_(True)
        b = deterministic((fan_out,), scale=0.05, offset=-0.01).requires_grad_(True)
        z = h @ W.t() + b
        act = activations[li]
        # softmax is not applied here; it is fused into cross_entropy (same as in MDNN)
        h = z if act == "softmax" else ACTS[act](z)
        layers.append({"units": fan_out, "activation": act, "W": W, "b": b, "z": z})

    out_pre = h  # for softmax these are the logits

    if loss_kind == "mse":
        target = deterministic((sizes[-1],), scale=0.3, offset=0.15)
        loss = F.mse_loss(out_pre, target, reduction="sum")
        output = out_pre
    elif loss_kind == "ce":
        # one-hot target; MDNN gets a one-hot vector, torch a class index
        cls = worst_class if worst_class is not None else 1 % sizes[-1]
        target = torch.zeros(sizes[-1], dtype=torch.float64)
        target[cls] = 1.0
        loss = F.cross_entropy(
            out_pre.unsqueeze(0), torch.tensor([cls]), reduction="sum"
        )
        output = torch.softmax(out_pre, dim=0)
    else:
        raise ValueError(loss_kind)

    loss.backward()

    write(name, {
        "name": name,
        "kind": "dense_model",
        "loss": loss_kind,
        "input": nested(x),
        "target": nested(target),
        "layers": [
            {
                "units": L["units"],
                "activation": L["activation"],
                "W": nested(L["W"]),
                "b": nested(L["b"]),
            }
            for L in layers
        ],
        "expected": {
            "output": nested(output),
            "loss": loss.item(),
            "dW": [nested(L["W"].grad) for L in layers],
            "db": [nested(L["b"].grad) for L in layers],
            "dInput": nested(x.grad),
        },
    })


# ----------------------------------------------------------------- Conv layer

def conv_layer(name, h, w, c, k, f, padding, activation):
    """Isolated Conv layer. Instead of a loss, an artificial output gradient (gradOutput)
    is given, exactly as the gradient-check tests do; this tests the layer on its own,
    so any mismatch points at Conv, not at something after it."""
    x_hwc = deterministic((h, w, c), scale=0.09, offset=0.04)
    x = x_hwc.permute(2, 0, 1).unsqueeze(0).clone().requires_grad_(True)  # [1,C,H,W]

    k_mdnn = deterministic((f, k, k, c), scale=0.13, offset=-0.02)        # [f,kh,kw,c]
    weight = k_mdnn.permute(0, 3, 1, 2).clone().requires_grad_(True)      # [f,c,kh,kw]
    bias = deterministic((f,), scale=0.06, offset=0.01).requires_grad_(True)

    z = F.conv2d(x, weight, bias, stride=1, padding=padding)
    out = ACTS[activation](z)

    oh, ow = out.shape[2], out.shape[3]
    g_hwf = deterministic((oh, ow, f), scale=0.05, offset=0.02)           # [oh,ow,f]
    g = g_hwf.permute(2, 0, 1).unsqueeze(0).clone()                       # [1,f,oh,ow]

    dx, dW, db = torch.autograd.grad(out, [x, weight, bias], grad_outputs=g)

    write(name, {
        "name": name,
        "kind": "conv_layer",
        "h": h, "w": w, "c": c, "k": k, "f": f,
        "padding": padding,
        "activation": activation,
        "input": nested(x_hwc),                                   # [H][W][C]
        "kernel": nested(k_mdnn),                                 # [f][kh][kw][c]
        "bias": nested(bias),
        "gradOutput": nested(g_hwf),                              # [oh][ow][f]
        "expected": {
            "output": nested(out.squeeze(0).permute(1, 2, 0)),     # [oh][ow][f]
            "dKernel": nested(dW.permute(0, 2, 3, 1)),             # [f][kh][kw][c]
            "dBias": nested(db),
            "dInput": nested(dx.squeeze(0).permute(1, 2, 0)),      # [H][W][C]
        },
    })


# -------------------------------------------------------------- MaxPool layer

def maxpool_layer(name, h, w, c, pool):
    x_hwc = deterministic((h, w, c), scale=0.09, offset=0.04)
    x = x_hwc.permute(2, 0, 1).unsqueeze(0).clone().requires_grad_(True)

    out = F.max_pool2d(x, pool)
    oh, ow = out.shape[2], out.shape[3]
    g_hwc = deterministic((oh, ow, c), scale=0.05, offset=0.02)
    g = g_hwc.permute(2, 0, 1).unsqueeze(0).clone()

    (dx,) = torch.autograd.grad(out, [x], grad_outputs=g)

    write(name, {
        "name": name,
        "kind": "maxpool_layer",
        "h": h, "w": w, "c": c, "pool": pool,
        "input": nested(x_hwc),
        "gradOutput": nested(g_hwc),
        "expected": {
            "output": nested(out.squeeze(0).permute(1, 2, 0)),
            "dInput": nested(dx.squeeze(0).permute(1, 2, 0)),
        },
    })


# ------------------------------------------------------------------ Optimizer

def adam_steps(name, n_params, steps, lr):
    """Adam over N steps; catches a bias-correction bug that does not yet show
    after a single step."""
    w0 = deterministic((n_params,), scale=0.2, offset=0.05)
    grads = [deterministic((n_params,), scale=0.05 + 0.01 * s, offset=0.02 * s - 0.01)
             for s in range(steps)]

    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    trace = []
    for s in range(steps):
        opt.zero_grad()
        w.grad = grads[s].clone()
        opt.step()
        trace.append(nested(w))

    write(name, {
        "name": name,
        "kind": "adam",
        "lr": lr,
        "initial": nested(w0),
        "grads": [nested(g) for g in grads],
        "expected": {"afterStep": trace},
    })


if __name__ == "__main__":
    print("generating fixtures into", os.path.normpath(OUT_DIR))

    dense_model("dense_1layer_linear_mse", [4, 3], ["linear"], "mse")
    dense_model("dense_2layer_tanh_mse", [4, 5, 3], ["tanh", "tanh"], "mse")
    dense_model("dense_3layer_relu_mse", [5, 6, 4, 2], ["relu", "relu", "linear"], "mse")
    dense_model("dense_2layer_relu_softmax_ce", [4, 6, 3], ["relu", "softmax"], "ce")
    dense_model("dense_sigmoid_mse", [3, 4, 2], ["sigmoid", "sigmoid"], "mse")

    # SATURATED softmax: the case the softmax+CE fusion exists for in the first place.
    # Large weights push the logits so far apart that the smallest probability drops to
    # ~1e-20. The generic path (-t/(s+1e-15) → Jacobian) LOSES the gradient there: epsilon
    # outweighs s itself. The fused path (s - t) holds. The target is deliberately the LEAST
    # likely class, where the gradient is most needed and also most fragile.
    # Without this fixture a backward refactor could break the fusion without the suite noticing.
    dense_model("dense_softmax_ce_saturated", [4, 8, 3], ["tanh", "softmax"], "ce",
                weight_scale=9.0, worst_class=0)

    conv_layer("conv_valid_1ch_1f", 3, 3, 1, 2, 1, "valid", "linear")
    conv_layer("conv_valid_2ch_3f", 5, 5, 2, 3, 3, "valid", "tanh")
    conv_layer("conv_same_odd_kernel", 5, 5, 1, 3, 2, "same", "tanh")
    conv_layer("conv_same_even_kernel", 5, 5, 1, 2, 2, "same", "tanh")   # ← risk: asymmetric padding
    conv_layer("conv_valid_relu", 6, 6, 2, 3, 2, "valid", "relu")

    maxpool_layer("maxpool_2x2_even", 4, 4, 1, 2)
    maxpool_layer("maxpool_2x2_multichannel", 6, 6, 2, 2)
    maxpool_layer("maxpool_2x2_odd_input", 5, 5, 1, 2)                   # ← risk: division remainder

    adam_steps("adam_5_steps", 4, 5, 0.01)

    print("done")

#!/usr/bin/env python3
"""
Generátor referenčních fixtures z PyTorche pro paritní testy MDNN.

PROČ TOHLE EXISTUJE
-------------------
Všechny ostatní testy v MDNN.Tests ověřují knihovnu PROTI SOBĚ SAMÉ: gradient check
porovná analytický backward s numerickou derivací VLASTNÍHO forwardu. Když je ale
forward konvenčně jinak (jiný padding, otočený kernel, jiné škálování loss), gradient
check projde a chyba se neukáže.

Tenhle skript vyrobí externí orákulum. Postaví v PyTorchi tytéž výpočty s PŘESNĚ
stejnými vahami a vstupy, vyexportuje výsledky do JSON a C# test pak porovná čísla
prvek po prvku.

SPUŠTĚNÍ (fixtures se commitují, takže tohle je potřeba jen při jejich změně):
    python3 -m venv .venv && .venv/bin/pip install torch numpy
    .venv/bin/python generate_fixtures.py

CI Python NEPOTŘEBUJE — čte jen hotové .json.

OVĚŘENÉ KONVENCE (proto sedí porovnání)
---------------------------------------
    MDNN MSE  = Σ(v−t)²       ≡ MSELoss(reduction='sum')
    MDNN CE   = −Σ t·ln(s)    ≡ CrossEntropyLoss(reduction='sum'), jeden vzorek
    MDNN Conv = křížová korelace bez otočení kernelu ≡ torch Conv2d
    MDNN Adam = w − lr·m̂/(√v̂ + 1e-8)  ≡ torch.optim.Adam (eps VEN ze sqrt)

ROZDÍLY V ULOŽENÍ (překlápí se tady, ne v C#)
---------------------------------------------
    Dense  MDNN neuron[i].Weights[j]      ≡ torch Linear.weight[i][j]
    Conv   MDNN Kernel[f][kh][kw][c]      ≡ torch weight[f][c][kh][kw]
    Conv   MDNN vstup [H][W][C]           ≡ torch [N][C][H][W]
"""

import json
import os

import torch
import torch.nn.functional as F

# Reference musí být v double, jinak by rozdíl proti C# (taky double) mířil na
# přesnost floatu, ne na neshodu konvence — což je přesně to, co chceme měřit.
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
    """Rozprostřené hodnoty bez RNG — fixtures musí být bitově stabilní napříč
    verzemi torche, jinak by se diff hýbal sám od sebe."""
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

def dense_model(name, sizes, activations, loss_kind):
    """Plně propojená síť: sizes = [vstup, skrytá..., výstup].
    V MDNN se staví jako new MDNN(výstupní) + Layers.Add(skryté) — pořadí
    v `layers` níž je stejné jako v MDNN.Layers.Layers, tedy skryté → výstupní."""
    x = deterministic((sizes[0],), scale=0.11, offset=-0.05).requires_grad_(True)

    layers = []
    h = x
    for li in range(len(sizes) - 1):
        fan_in, fan_out = sizes[li], sizes[li + 1]
        W = deterministic((fan_out, fan_in), scale=0.07, offset=0.02).requires_grad_(True)
        b = deterministic((fan_out,), scale=0.05, offset=-0.01).requires_grad_(True)
        z = h @ W.t() + b
        act = activations[li]
        # softmax se neaplikuje tady — je fúzovaný do cross_entropy (stejně jako v MDNN)
        h = z if act == "softmax" else ACTS[act](z)
        layers.append({"units": fan_out, "activation": act, "W": W, "b": b, "z": z})

    out_pre = h  # u softmaxu jsou to logity

    if loss_kind == "mse":
        target = deterministic((sizes[-1],), scale=0.3, offset=0.15)
        loss = F.mse_loss(out_pre, target, reduction="sum")
        output = out_pre
    elif loss_kind == "ce":
        # one-hot cíl; MDNN dostává one-hot vektor, torch index třídy
        cls = 1 % sizes[-1]
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
    """Izolovaná Conv vrstva. Místo loss se zadá umělý gradient shora (gradOutput),
    přesně jak to dělají stávající gradient-check testy — tím se vrstva testuje
    samostatně a případná neshoda ukazuje na Conv, ne na něco za ní."""
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
    """Adam přes N kroků se stejným gradientem — chytí chybu v bias correction,
    která se po jednom kroku ještě neprojeví."""
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
    print("generuji fixtures do", os.path.normpath(OUT_DIR))

    dense_model("dense_1layer_linear_mse", [4, 3], ["linear"], "mse")
    dense_model("dense_2layer_tanh_mse", [4, 5, 3], ["tanh", "tanh"], "mse")
    dense_model("dense_3layer_relu_mse", [5, 6, 4, 2], ["relu", "relu", "linear"], "mse")
    dense_model("dense_2layer_relu_softmax_ce", [4, 6, 3], ["relu", "softmax"], "ce")
    dense_model("dense_sigmoid_mse", [3, 4, 2], ["sigmoid", "sigmoid"], "mse")

    conv_layer("conv_valid_1ch_1f", 3, 3, 1, 2, 1, "valid", "linear")
    conv_layer("conv_valid_2ch_3f", 5, 5, 2, 3, 3, "valid", "tanh")
    conv_layer("conv_same_odd_kernel", 5, 5, 1, 3, 2, "same", "tanh")
    conv_layer("conv_same_even_kernel", 5, 5, 1, 2, 2, "same", "tanh")   # ← riziko: asymetrický pad
    conv_layer("conv_valid_relu", 6, 6, 2, 3, 2, "valid", "relu")

    maxpool_layer("maxpool_2x2_even", 4, 4, 1, 2)
    maxpool_layer("maxpool_2x2_multichannel", 6, 6, 2, 2)
    maxpool_layer("maxpool_2x2_odd_input", 5, 5, 1, 2)                   # ← riziko: zbytek při dělení

    adam_steps("adam_5_steps", 4, 5, 0.01)

    print("hotovo")

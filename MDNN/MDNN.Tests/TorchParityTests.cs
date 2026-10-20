using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// Parity tests against PyTorch: an EXTERNAL ORACLE.
///
/// Why separate from the gradient checks: a gradient check compares the analytic backward
/// with the numerical derivative of the library's OWN forward. If the forward follows a different
/// convention (different padding, flipped kernel, different loss scaling), the gradient check
/// passes and the bug stays hidden. These tests compare numbers with an independent implementation.
///
/// The fixtures are produced by tools/torch_fixtures/generate_fixtures.py and are committed,
/// so CI does not need Python.
/// </summary>
public class TorchParityTests
{
    // Both sides compute in double and perform the same operations, so any difference can only
    // come from a different summation order. 1e-9 leaves orders of magnitude of margin; a
    // convention mismatch typically gives 1e-2 and above, not something in between.
    private const double Tol = 1e-9;

    // ------------------------------------------------------------ loading

    private static JsonElement Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Missing fixture '{name}'. Generate it with: tools/torch_fixtures/generate_fixtures.py", path);

        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    private static double[] D1(JsonElement e) =>
        e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    private static double[][] D2(JsonElement e) =>
        e.EnumerateArray().Select(D1).ToArray();

    private static double[][][] D3(JsonElement e) =>
        e.EnumerateArray().Select(D2).ToArray();

    private static double[][][][] D4(JsonElement e) =>
        e.EnumerateArray().Select(D3).ToArray();

    private static ActivationFunction Act(string name) => name switch
    {
        "linear"  => new Linear(),
        "relu"    => new ReLU(),
        "tanh"    => new Tanh(),
        "sigmoid" => new Sigmoid(),
        "softmax" => new Softmax(),
        _ => throw new ArgumentException($"Unknown activation '{name}'"),
    };

    // ------------------------------------------------------------ comparison

    /// <summary>Compares two number sequences and on a mismatch says WHERE and BY HOW MUCH.
    /// Without that, an "expected X, got Y" message over 200 numbers would be useless.</summary>
    private static void Same(string what, IEnumerable<double> expected, IEnumerable<double> actual)
    {
        double[] e = expected.ToArray();
        double[] a = actual.ToArray();

        Assert.True(e.Length == a.Length,
            $"{what}: different element count: PyTorch {e.Length}, NeuralNetwork {a.Length}. " +
            "The output shape probably does not match (padding? axis order?).");

        double worst = 0; int worstIdx = 0;
        for (int i = 0; i < e.Length; i++)
        {
            // absolute and relative at once: near zero the relative error is inflated,
            // for large numbers the absolute one is worthless
            double diff = Math.Abs(e[i] - a[i]) / (1.0 + Math.Abs(e[i]));
            if (diff > worst) { worst = diff; worstIdx = i; }
        }

        // the message is built ONLY on failure: the Assert.True argument is evaluated
        // eagerly, so otherwise it would be formatted for passing tests too
        if (worst >= Tol)
            Assert.Fail(
                $"{what}: mismatch at index {worstIdx}: PyTorch {e[worstIdx]:G17}, " +
                $"NeuralNetwork {a[worstIdx]:G17} (deviation {worst:E2}, threshold {Tol:E0})");
    }

    private static IEnumerable<double> Flat(double[][] a) => a.SelectMany(x => x);
    private static IEnumerable<double> Flat(double[][][] a) => a.SelectMany(Flat);
    private static IEnumerable<double> Flat(double[][][][] a) => a.SelectMany(Flat);

    // ============================================================ DENSE

    [Theory]
    [InlineData("dense_1layer_linear_mse")]
    [InlineData("dense_2layer_tanh_mse")]
    [InlineData("dense_3layer_relu_mse")]
    [InlineData("dense_2layer_relu_softmax_ce")]
    [InlineData("dense_sigmoid_mse")]
    [InlineData("dense_softmax_ce_saturated")]
    public void Dense_model_matches_pytorch(string fixture)
    {
        JsonElement fx = Load(fixture);

        double[] input  = D1(fx.GetProperty("input"));
        double[] target = D1(fx.GetProperty("target"));
        var specs = fx.GetProperty("layers").EnumerateArray().ToArray();

        Loss loss = fx.GetProperty("loss").GetString() == "ce"
            ? new CrossEntropy()
            : new MSE();

        // In MDNN a model is built from the OUTPUT layer; Add() inserts before it.
        var last = specs[^1];
        var model = new NeuralNetwork(
            new Dense(last.GetProperty("units").GetInt32(),
                      Act(last.GetProperty("activation").GetString()!)),
            new SGD(0.01), loss);

        for (int i = 0; i < specs.Length - 1; i++)
            model.Layers.Add(new Dense(specs[i].GetProperty("units").GetInt32(),
                                       Act(specs[i].GetProperty("activation").GetString()!)));

        var x = new Tensor(input);

        // Weight materialization is deferred to the first pass; without this there would
        // be no neurons yet to set the parameters on.
        model.GetResults(x);

        for (int li = 0; li < specs.Length; li++)
        {
            double[][] W = D2(specs[li].GetProperty("W"));
            double[]   b = D1(specs[li].GetProperty("b"));
            var neurons = ((Dense)model.Layers.Layers[li]).Neurons;

            Assert.True(neurons.Count == W.Length,
                $"{fixture}, layer {li}: PyTorch has {W.Length} neurons, NeuralNetwork {neurons.Count}.");

            for (int n = 0; n < neurons.Count; n++)
                neurons[n].SetParamsForTests(W[n], b[n]);
        }

        var expected = fx.GetProperty("expected");

        // --- forward ---
        Same($"{fixture}: output", D1(expected.GetProperty("output")), model.GetResults(x).Data);

        Same($"{fixture}: loss",
             new[] { expected.GetProperty("loss").GetDouble() },
             new[] { model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target) });

        // --- backward (Fit = forward + backprop, WITHOUT a weight update) ---
        model.Train.Fit(x, new Tensor(target));

        double[][][] dW = D3(expected.GetProperty("dW"));
        double[][]   db = D2(expected.GetProperty("db"));

        for (int li = 0; li < specs.Length; li++)
        {
            var neurons = ((Dense)model.Layers.Layers[li]).Neurons;

            Same($"{fixture}: dW layer {li}",
                 Flat(dW[li]),
                 neurons.SelectMany(n => n.gradientsW));

            Same($"{fixture}: dBias layer {li}",
                 db[li],
                 neurons.Select(n => n.gradientsB));
        }
    }

    // ============================================================ CONV

    [Theory]
    [InlineData("conv_valid_1ch_1f")]
    [InlineData("conv_valid_2ch_3f")]
    [InlineData("conv_same_odd_kernel")]
    [InlineData("conv_same_even_kernel")]
    [InlineData("conv_valid_relu")]
    public void Conv_layer_matches_pytorch(string fixture)
    {
        JsonElement fx = Load(fixture);

        int h = fx.GetProperty("h").GetInt32();
        int w = fx.GetProperty("w").GetInt32();
        int c = fx.GetProperty("c").GetInt32();
        int k = fx.GetProperty("k").GetInt32();
        int f = fx.GetProperty("f").GetInt32();
        string padding = fx.GetProperty("padding").GetString()!;

        var conv = new Conv(f, k, Act(fx.GetProperty("activation").GetString()!), padding);
        conv.Build(null, new[] { h, w, c });

        // input [H][W][C] → double[,,]
        double[][][] inNested = D3(fx.GetProperty("input"));
        var input = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int ch = 0; ch < c; ch++)
            input[i, j, ch] = inNested[i][j][ch];

        // kernel [f][kh][kw][c]: same layout as MDNN, just copied in
        double[][][][] kern = D4(fx.GetProperty("kernel"));
        double[] bias = D1(fx.GetProperty("bias"));
        for (int fi = 0; fi < f; fi++)
        {
            for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++)
            for (int ch = 0; ch < c; ch++)
                conv.Kernels[fi][ki][kj][ch] = kern[fi][ki][kj][ch];

            conv.Biases[fi] = bias[fi];
        }

        var outTensor = conv.FeedForward(new Tensor(input));
        var expected = fx.GetProperty("expected");

        Same($"{fixture}: output", Flat(D3(expected.GetProperty("output"))), outTensor.Data);

        // the output gradient is passed straight to Backward as a 3D tensor,
        // the same way the gradient-check tests do it
        double[][][] gNested = D3(fx.GetProperty("gradOutput"));
        int oh = gNested.Length, ow = gNested[0].Length;
        var g = new double[oh, ow, f];
        for (int i = 0; i < oh; i++)
        for (int j = 0; j < ow; j++)
        for (int fi = 0; fi < f; fi++)
            g[i, j, fi] = gNested[i][j][fi];

        var gTensor = new Tensor(g);
        var dInput = conv.Backward(gTensor);

        Same($"{fixture}: dKernel", Flat(D4(expected.GetProperty("dKernel"))), Flat(conv.dKernels));
        Same($"{fixture}: dBias",   D1(expected.GetProperty("dBias")),         conv.dBiases);
        Same($"{fixture}: dInput",  Flat(D3(expected.GetProperty("dInput"))),  dInput.Data);
    }

    // ============================================================ MAXPOOL

    [Theory]
    [InlineData("maxpool_2x2_even")]
    [InlineData("maxpool_2x2_multichannel")]
    [InlineData("maxpool_2x2_odd_input")]
    public void MaxPool_layer_matches_pytorch(string fixture)
    {
        JsonElement fx = Load(fixture);

        int h = fx.GetProperty("h").GetInt32();
        int w = fx.GetProperty("w").GetInt32();
        int c = fx.GetProperty("c").GetInt32();

        var pool = new MaxPool(fx.GetProperty("pool").GetInt32());
        pool.Build(null, new[] { h, w, c });

        double[][][] inNested = D3(fx.GetProperty("input"));
        var input = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int ch = 0; ch < c; ch++)
            input[i, j, ch] = inNested[i][j][ch];

        var outTensor = pool.FeedForward(new Tensor(input));
        var expected = fx.GetProperty("expected");

        Same($"{fixture}: output", Flat(D3(expected.GetProperty("output"))), outTensor.Data);

        double[][][] gNested = D3(fx.GetProperty("gradOutput"));
        int oh = gNested.Length, ow = gNested[0].Length;
        var g = new double[oh, ow, c];
        for (int i = 0; i < oh; i++)
        for (int j = 0; j < ow; j++)
        for (int ch = 0; ch < c; ch++)
            g[i, j, ch] = gNested[i][j][ch];

        var dInput = pool.Backward(new Tensor(g));

        Same($"{fixture}: dInput", Flat(D3(expected.GetProperty("dInput"))), dInput.Data);
    }

    // ============================================================ ADAM

    [Fact]
    public void Adam_matches_pytorch_over_multiple_steps()
    {
        // Several steps on purpose: a bias-correction bug does not show after one step,
        // because the correction terms are trivial on the first step.
        JsonElement fx = Load("adam_5_steps");

        double lr = fx.GetProperty("lr").GetDouble();
        double[] w = D1(fx.GetProperty("initial"));
        double[][] grads = D2(fx.GetProperty("grads"));
        double[][] expected = D2(fx.GetProperty("expected").GetProperty("afterStep"));

        var adam = new Adam(lr);   // the state (m/v/t) sizes itself on the first Update

        for (int step = 0; step < grads.Length; step++)
        {
            for (int i = 0; i < w.Length; i++)
                w[i] = adam.Update(w[i], grads[step][i], i);

            Same($"adam: weights after step {step + 1}", expected[step], w);
        }
    }
}

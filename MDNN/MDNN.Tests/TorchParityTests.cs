using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Paritní testy proti PyTorchi — EXTERNÍ ORÁKULUM.
///
/// Proč zvlášť od gradient checků: gradient check porovná analytický backward
/// s numerickou derivací VLASTNÍHO forwardu. Když je forward konvenčně jinak
/// (jiný padding, otočený kernel, jiné škálování loss), gradient check projde
/// a chyba se neukáže. Tyhle testy porovnávají čísla s nezávislou implementací.
///
/// Fixtures vyrábí tools/torch_fixtures/generate_fixtures.py a jsou commitnuté,
/// takže CI Python nepotřebuje.
/// </summary>
public class TorchParityTests
{
    // Obě strany počítají v double a dělají tytéž operace, takže rozdíl smí být
    // jen z jiného pořadí sčítání. 1e-9 je na to řádově s rezervou; kdyby se
    // lišila konvence, rozdíl je typicky 1e-2 a výš — ne někde mezi.
    private const double Tol = 1e-9;

    // ------------------------------------------------------------ načítání

    private static JsonElement Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Chybí fixture '{name}'. Vyrob ho: tools/torch_fixtures/generate_fixtures.py", path);

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

    private static Activation_func Act(string name) => name switch
    {
        "linear"  => new Linear(),
        "relu"    => new ReLu(),
        "tanh"    => new Tanh(),
        "sigmoid" => new Sigmoid(),
        "softmax" => new Softmax(),
        _ => throw new ArgumentException($"Neznámá aktivace '{name}'"),
    };

    // ------------------------------------------------------------ porovnání

    /// <summary>Porovná dvě posloupnosti čísel a při neshodě řekne KDE a O KOLIK.
    /// Bez toho by hláška „očekáváno X, dostal Y" u 200 čísel k ničemu nebyla.</summary>
    private static void Same(string what, IEnumerable<double> expected, IEnumerable<double> actual)
    {
        double[] e = expected.ToArray();
        double[] a = actual.ToArray();

        Assert.True(e.Length == a.Length,
            $"{what}: jiný počet prvků — PyTorch {e.Length}, MDNN {a.Length}. " +
            "Nejspíš nesedí tvar výstupu (padding? pořadí os?).");

        double worst = 0; int worstIdx = 0;
        for (int i = 0; i < e.Length; i++)
        {
            // absolutní i relativní zároveň: u hodnot kolem nuly je relativní chyba
            // nafouknutá, u velkých čísel je zase absolutní bezcenná
            double diff = Math.Abs(e[i] - a[i]) / (1.0 + Math.Abs(e[i]));
            if (diff > worst) { worst = diff; worstIdx = i; }
        }

        // hláška se staví AŽ při neúspěchu — argument Assert.True se vyhodnocuje
        // dychtivě, takže by se jinak formátovala i u zelených testů
        if (worst >= Tol)
            Assert.Fail(
                $"{what}: neshoda na indexu {worstIdx} — PyTorch {e[worstIdx]:G17}, " +
                $"MDNN {a[worstIdx]:G17} (odchylka {worst:E2}, práh {Tol:E0})");
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
    public void Dense_model_matches_pytorch(string fixture)
    {
        JsonElement fx = Load(fixture);

        double[] input  = D1(fx.GetProperty("input"));
        double[] target = D1(fx.GetProperty("target"));
        var specs = fx.GetProperty("layers").EnumerateArray().ToArray();

        Loss loss = fx.GetProperty("loss").GetString() == "ce"
            ? new CrossEntropy()
            : new MSE();

        // V MDNN se model staví od VÝSTUPNÍ vrstvy; Add() přidává před ni.
        var last = specs[^1];
        var model = new My_DNN.MDNN(
            new Dense(last.GetProperty("units").GetInt32(),
                      Act(last.GetProperty("activation").GetString()!)),
            new SGD(0.01), loss);

        for (int i = 0; i < specs.Length - 1; i++)
            model.Layers.Add(new Dense(specs[i].GetProperty("units").GetInt32(),
                                       Act(specs[i].GetProperty("activation").GetString()!)));

        var x = new Tensor(input);

        // Materializace vah je odložená na první průchod — bez tohohle by ještě
        // neexistovaly neurony, do kterých se dají nastavit parametry.
        model.GetResults(x);

        for (int li = 0; li < specs.Length; li++)
        {
            double[][] W = D2(specs[li].GetProperty("W"));
            double[]   b = D1(specs[li].GetProperty("b"));
            var neurons = ((Dense)model.Layers.Layers[li]).Neurons;

            Assert.True(neurons.Count == W.Length,
                $"{fixture}, vrstva {li}: PyTorch má {W.Length} neuronů, MDNN {neurons.Count}.");

            for (int n = 0; n < neurons.Count; n++)
                neurons[n].SetParamsForTests(W[n], b[n]);
        }

        var expected = fx.GetProperty("expected");

        // --- forward ---
        Same($"{fixture}: výstup", D1(expected.GetProperty("output")), model.GetResults(x).Data);

        Same($"{fixture}: loss",
             new[] { expected.GetProperty("loss").GetDouble() },
             new[] { model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target) });

        // --- backward (Fit = forward + backprop, BEZ update vah) ---
        model.Train.Fit(x, new Tensor(target));

        double[][][] dW = D3(expected.GetProperty("dW"));
        double[][]   db = D2(expected.GetProperty("db"));

        for (int li = 0; li < specs.Length; li++)
        {
            var neurons = ((Dense)model.Layers.Layers[li]).Neurons;

            Same($"{fixture}: dW vrstva {li}",
                 Flat(dW[li]),
                 neurons.SelectMany(n => n.gradientsW));

            Same($"{fixture}: dBias vrstva {li}",
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
        conv.LayerAdjustment(null, new[] { h, w, c });

        // vstup [H][W][C] → double[,,]
        double[][][] inNested = D3(fx.GetProperty("input"));
        var input = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int ch = 0; ch < c; ch++)
            input[i, j, ch] = inNested[i][j][ch];

        // kernel [f][kh][kw][c] — stejný layout jako MDNN, jen se překopíruje dovnitř
        double[][][][] kern = D4(fx.GetProperty("kernel"));
        double[] bias = D1(fx.GetProperty("bias"));
        for (int fi = 0; fi < f; fi++)
        {
            for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++)
            for (int ch = 0; ch < c; ch++)
                conv.Kernel[fi][ki][kj][ch] = kern[fi][ki][kj][ch];

            conv.Biases[fi] = bias[fi];
        }

        var outTensor = conv.FeedForward(new Tensor(input));
        var expected = fx.GetProperty("expected");

        Same($"{fixture}: výstup", Flat(D3(expected.GetProperty("output"))), outTensor.Data);

        // gradient shora se zadává rovnou (3D tvar = „další vrstva není Dense"),
        // stejně jako to dělají stávající gradient-check testy
        double[][][] gNested = D3(fx.GetProperty("gradOutput"));
        int oh = gNested.Length, ow = gNested[0].Length;
        var g = new double[oh, ow, f];
        for (int i = 0; i < oh; i++)
        for (int j = 0; j < ow; j++)
        for (int fi = 0; fi < f; fi++)
            g[i, j, fi] = gNested[i][j][fi];

        var gTensor = new Tensor(g);
        var dInput = conv.CalculateLayerGradients(gTensor, null!);
        conv.BackPropagation(gTensor);

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
        pool.LayerAdjustment(null, new[] { h, w, c });

        double[][][] inNested = D3(fx.GetProperty("input"));
        var input = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int ch = 0; ch < c; ch++)
            input[i, j, ch] = inNested[i][j][ch];

        var outTensor = pool.FeedForward(new Tensor(input));
        var expected = fx.GetProperty("expected");

        Same($"{fixture}: výstup", Flat(D3(expected.GetProperty("output"))), outTensor.Data);

        double[][][] gNested = D3(fx.GetProperty("gradOutput"));
        int oh = gNested.Length, ow = gNested[0].Length;
        var g = new double[oh, ow, c];
        for (int i = 0; i < oh; i++)
        for (int j = 0; j < ow; j++)
        for (int ch = 0; ch < c; ch++)
            g[i, j, ch] = gNested[i][j][ch];

        var dInput = pool.CalculateLayerGradients(new Tensor(g), null!);

        Same($"{fixture}: dInput", Flat(D3(expected.GetProperty("dInput"))), dInput.Data);
    }

    // ============================================================ ADAM

    [Fact]
    public void Adam_matches_pytorch_over_multiple_steps()
    {
        // Přes víc kroků schválně: chyba v bias correction se po jednom kroku
        // ještě neprojeví, protože korekční členy jsou v prvním kroku triviální.
        JsonElement fx = Load("adam_5_steps");

        double lr = fx.GetProperty("lr").GetDouble();
        double[] w = D1(fx.GetProperty("initial"));
        double[][] grads = D2(fx.GetProperty("grads"));
        double[][] expected = D2(fx.GetProperty("expected").GetProperty("afterStep"));

        var adam = new Adam(lr);   // stav (m/v/t) si dorovnává sám při prvním Update

        for (int step = 0; step < grads.Length; step++)
        {
            for (int i = 0; i < w.Length; i++)
                w[i] = adam.Update(w[i], grads[step][i], i);

            Same($"adam: váhy po kroku {step + 1}", expected[step], w);
        }
    }
}

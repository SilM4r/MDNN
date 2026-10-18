using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Stav optimizeru se ukládá s modelem, takže trénink jde přerušit a plynule navázat.
///
/// Dřív soubor nesl jen jméno a hyperparametry optimizeru. Načtený model začínal s Adamem
/// od nuly — vynulované momenty a krok 0, takže první kroky po načtení byly velké jako na
/// začátku tréninku. Hlavní kontrakt teď: N kroků → uložit → načíst → M kroků dá BIT PO BITU
/// totéž co N + M kroků bez přerušení.
/// </summary>
public class OptimizerSaveTests
{
    private static readonly Tensor[] X = Enumerable.Range(0, 16)
        .Select(i => new Tensor(new[] { Math.Sin(i), Math.Cos(i * 0.7), i * 0.05 })).ToArray();
    private static readonly Tensor[] T = Enumerable.Range(0, 16)
        .Select(i => new Tensor(new[] { Math.Sin(i) * 0.5, Math.Cos(i) })).ToArray();

    private static My_DNN.MDNN Mlp(Optimizer optimizer, Loss? loss = null)
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), optimizer, loss ?? new MSE(), seed: 4);
        model.Layers.Add(new Dense(5, new Tanh()));
        model.GetResults(X[0]);
        return model;
    }

    // Pevné pořadí vzorků — bez míchání, ať pozice v generátoru náhody (ta se neukládá)
    // nehraje roli a porovnává se jen stav optimizeru.
    private static void Steps(My_DNN.MDNN model, Tensor[] x, Tensor[] t, int batches)
    {
        for (int b = 0; b < batches; b++)
        {
            for (int k = 0; k < 4; k++)
            {
                int i = (b * 4 + k) % x.Length;
                model.Train.Fit(x[i], t[i]);
            }
            model.Train.UpdateParams();
        }
    }

    private static double[] Parameters(My_DNN.MDNN model)
    {
        var all = new List<double>();
        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is LayerBasedOnNeurons byNeurons)
                foreach (Neuron n in byNeurons.Neurons) { all.AddRange(n.Weights); all.Add(n.Bias); }
            else if (layer is Conv conv)
            {
                foreach (var f in conv.Kernel) foreach (var r in f) foreach (var c in r) all.AddRange(c);
                all.AddRange(conv.Biases);
            }
        }
        return all.ToArray();
    }

    private static string TempPath() => Path.Combine(Path.GetTempPath(), "mdnn_opt_" + Guid.NewGuid().ToString("N") + ".json");

    private static My_DNN.MDNN SaveAndLoad(My_DNN.MDNN model, bool includeState = true,
                                           Optimizer? optimizer = null, Loss? loss = null)
    {
        string path = TempPath();
        try
        {
            model.SaveAsJson(path, includeState);
            return My_DNN.MDNN.LoadModel(path, optimizer: optimizer, loss: loss);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------ navázání tréninku

    public static TheoryData<Func<Optimizer>> BuiltIn()
        => new() { () => new Adam(0.05), () => new Momentum(0.05, 0.9), () => new SGD(0.05) };

    [Theory]
    [MemberData(nameof(BuiltIn))]
    public void Resuming_from_a_file_is_bit_identical_to_uninterrupted_training(Func<Optimizer> optimizer)
    {
        var uninterrupted = Mlp(optimizer());
        Steps(uninterrupted, X, T, 12);

        var first = Mlp(optimizer());
        Steps(first, X, T, 6);
        var resumed = SaveAndLoad(first);
        // druhá půlka dat — Steps počítá index od nuly, takže se posune ručně
        Steps(resumed, X.Skip(8).Concat(X.Take(8)).ToArray(), T.Skip(8).Concat(T.Take(8)).ToArray(), 6);

        Assert.Equal(Parameters(uninterrupted), Parameters(resumed));
    }

    [Fact]
    public void Conv_optimizer_state_survives_save_and_load()
    {
        var img = Enumerable.Range(0, 8).Select(s =>
        {
            var a = new double[4, 4, 2];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int c = 0; c < 2; c++) a[i, j, c] = Math.Sin(s + i + j * 0.5 + c);
            return new Tensor(a);
        }).ToArray();
        var lab = Enumerable.Range(0, 8).Select(s => new Tensor(new[] { s % 2 == 0 ? 1.0 : 0.0, s % 2 == 0 ? 0.0 : 1.0 })).ToArray();

        My_DNN.MDNN Cnn()
        {
            var m = new My_DNN.MDNN(new Dense(2, new Softmax()), new Adam(0.05), new CrossEntropy(), seed: 2);
            m.Layers.Add(new Conv(2, 2, new ReLu(), "valid"));
            m.GetResults(img[0]);
            return m;
        }

        var uninterrupted = Cnn();
        Steps(uninterrupted, img, lab, 6);

        var first = Cnn();
        Steps(first, img, lab, 3);
        var resumed = SaveAndLoad(first);
        Steps(resumed, img.Skip(4).Concat(img.Take(4)).ToArray(), lab.Skip(4).Concat(lab.Take(4)).ToArray(), 3);

        Assert.Equal(Parameters(uninterrupted), Parameters(resumed));
    }

    [Fact]
    public void Without_state_the_file_is_smaller_and_adam_starts_over()
    {
        var reference = Mlp(new Adam(0.05));
        Steps(reference, X, T, 12);

        var first = Mlp(new Adam(0.05));
        Steps(first, X, T, 6);

        string withState = first.SaveAsJsonString();
        string withoutState = first.SaveAsJsonString(includeOptimizerState: false);
        Assert.Contains("OptimizerState", withState);
        Assert.DoesNotContain("OptimizerState", withoutState);
        Assert.True(withoutState.Length * 2 < withState.Length,
            $"bez stavu {withoutState.Length} znaků, se stavem {withState.Length}");

        // Bez stavu je to přesně dřívější chování: navázání NENÍ plynulé.
        var resumed = SaveAndLoad(first, includeState: false);
        Steps(resumed, X.Skip(8).Concat(X.Take(8)).ToArray(), T.Skip(8).Concat(T.Take(8)).ToArray(), 6);
        Assert.NotEqual(Parameters(reference), Parameters(resumed));
    }

    [Fact]
    public void Corrupted_state_is_rejected()
    {
        var adam = new Adam(0.1);
        var state = new Dictionary<string, double[]> { ["m"] = new double[3], ["v"] = new double[2], ["t"] = new double[3] };

        Assert.Throws<ArgumentException>(() => adam.ImportState(state));
        Assert.Throws<ArgumentException>(() => adam.ImportState(new Dictionary<string, double[]> { ["m"] = new double[1] }));
    }

    // ------------------------------------------------------------ změna optimizeru

    [Fact]
    public void Changing_the_context_optimizer_reaches_every_layer()
    {
        // Dřív se změna nepropsala: neurony trénovaly dál SGD, ale do souboru se uložil
        // Adam — po načtení tak model trénoval jiným optimizerem než před uložením.
        var model = Mlp(new SGD(0.05));
        model.Layers.Add(new Conv(1, 1, new Linear(), "valid"));   // jen aby měl i Conv
        model.GetResults(X[0]);
        Steps(model, X, T, 2);

        model.Context.Optimizer = new Adam(0.01);

        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is LayerBasedOnNeurons byNeurons)
                Assert.All(byNeurons.Neurons, n => Assert.IsType<Adam>(n.optimizer));
        }
        Assert.IsType<Adam>(model.Layers.Layers.OfType<Conv>().Single().OptimizerForSave);

        // a uložený model s ním opravdu pokračuje
        var loaded = SaveAndLoad(model);
        Assert.All(((Dense)loaded.Layers.Layers[^1]).Neurons, n => Assert.IsType<Adam>(n.optimizer));
    }

    [Fact]
    public void Reassigning_the_same_optimizer_keeps_its_state()
    {
        var model = Mlp(new Adam(0.05));
        Steps(model, X, T, 3);
        var neuron = ((Dense)model.Layers.Layers[0]).Neurons[0];
        double[] before = neuron.optimizer.ExportState()!["m"];

        model.Context.Optimizer = model.Context.Optimizer;

        Assert.Equal(before, neuron.optimizer.ExportState()!["m"]);
    }

    // ------------------------------------------------------------ vlastní optimizer a loss

    /// <summary>Vlastní optimizer se stavem (zjednodušený momentum).</summary>
    private sealed class Heavy : Optimizer
    {
        private readonly double lr;
        private double[] velocity = Array.Empty<double>();
        public Heavy(double lr) { this.lr = lr; }
        public override string Name => "Heavy";
        public override double[] Hyperparameters => new[] { lr };
        public override Optimizer CreateFresh() => new Heavy(lr);
        public override void EnsureCapacity(int n) { if (velocity.Length < n) Array.Resize(ref velocity, n); }
        public override double Update(double w, double g, int i)
        {
            EnsureCapacity(i + 1);
            velocity[i] = 0.8 * velocity[i] + g;
            return w - lr * velocity[i];
        }
        public override Dictionary<string, double[]> ExportState() => new() { ["velocity"] = (double[])velocity.Clone() };
        public override void ImportState(Dictionary<string, double[]> state) => velocity = (double[])state["velocity"].Clone();
    }

    private sealed class HalfMse : Loss
    {
        public override string Name => "HalfMSE";
        public override double LossFunction(double value, double target) => 0.5 * (value - target) * (value - target);
        public override double DerivativeOfLossFunction(double value, double target) => value - target;
    }

    [Fact]
    public void Custom_optimizer_and_loss_load_when_passed_in_and_keep_their_state()
    {
        var uninterrupted = Mlp(new Heavy(0.05), new HalfMse());
        Steps(uninterrupted, X, T, 12);

        var first = Mlp(new Heavy(0.05), new HalfMse());
        Steps(first, X, T, 6);
        var resumed = SaveAndLoad(first, optimizer: new Heavy(0.05), loss: new HalfMse());
        Steps(resumed, X.Skip(8).Concat(X.Take(8)).ToArray(), T.Skip(8).Concat(T.Take(8)).ToArray(), 6);

        Assert.IsType<HalfMse>(resumed.Loss);
        Assert.Equal(Parameters(uninterrupted), Parameters(resumed));
    }

    [Fact]
    public void Custom_optimizer_without_an_instance_fails_with_a_hint()
    {
        var model = Mlp(new Heavy(0.05));
        var ex = Assert.Throws<ArgumentException>(() => SaveAndLoad(model));
        Assert.Contains("LoadModel", ex.Message);
    }

    [Fact]
    public void Passed_optimizer_must_have_the_saved_name()
    {
        var model = Mlp(new Adam(0.05));
        Assert.Throws<ArgumentException>(() => SaveAndLoad(model, optimizer: new SGD(0.1)));
    }

    [Fact]
    public void Passed_built_in_optimizer_overrides_hyperparameters_but_keeps_state()
    {
        // Dotrénování s nižším learning rate: hyperparametry z předané instance, stav ze souboru.
        var model = Mlp(new Adam(0.05));
        Steps(model, X, T, 3);
        double[] m = ((Dense)model.Layers.Layers[0]).Neurons[0].optimizer.ExportState()!["m"];

        var loaded = SaveAndLoad(model, optimizer: new Adam(0.001));
        var optimizer = ((Dense)loaded.Layers.Layers[0]).Neurons[0].optimizer;

        Assert.Equal(0.001, optimizer.Hyperparameters[0]);
        Assert.Equal(m, optimizer.ExportState()!["m"]);
    }

    [Fact]
    public void TrainLoop_with_custom_optimizer_and_loss_restores_the_best_model()
    {
        // Na konci TrainLoop se obnovuje nejlepší model ze snapshotu. Ten hledal optimizer
        // a loss podle jména, takže s vlastními spadl na ArgumentException.
        var x = Enumerable.Range(0, 20).Select(i => new[] { Math.Sin(i), Math.Cos(i) }).ToArray();
        var y = x.Select(r => new[] { r[0] * r[1] }).ToArray();
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new Heavy(0.02), new HalfMse(), seed: 3);
        model.Train.ShowLossChartInTrainLoop = false;
        model.Train.ShowModelInfoIntrainLoop = false;
        model.Train.AutoSaveInTrainLoop = false;

        model.Train.TrainLoop(x, y, 5, 4);   // TestNeuralNetworkAfterTraining zůstává zapnutý

        Assert.IsType<HalfMse>(model.Loss);
        Assert.IsType<Heavy>(((Dense)model.Layers.Layers[0]).Neurons[0].optimizer);
    }
}

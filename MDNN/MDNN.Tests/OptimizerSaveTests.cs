using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// The optimizer state is saved with the model, so training can be interrupted and resumed seamlessly.
///
/// Previously the file only held the optimizer's name and hyperparameters. A loaded model started
/// Adam from scratch (zeroed moments, step 0), so the first steps after loading were as large as at
/// the start of training. The main contract now: N steps → save → load → M steps gives BIT FOR BIT
/// the same result as N + M uninterrupted steps.
/// </summary>
public class OptimizerSaveTests
{
    private static readonly Tensor[] X = Enumerable.Range(0, 16)
        .Select(i => new Tensor(new[] { Math.Sin(i), Math.Cos(i * 0.7), i * 0.05 })).ToArray();
    private static readonly Tensor[] T = Enumerable.Range(0, 16)
        .Select(i => new Tensor(new[] { Math.Sin(i) * 0.5, Math.Cos(i) })).ToArray();

    private static NeuralNetwork Mlp(Optimizer optimizer, Loss? loss = null)
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), optimizer, loss ?? new MSE(), seed: 4);
        model.Layers.Add(new Dense(5, new Tanh()));
        model.GetResults(X[0]);
        return model;
    }

    // Fixed sample order, no shuffling, so the random generator position (which is not saved)
    // plays no role and only the optimizer state is compared.
    private static void Steps(NeuralNetwork model, Tensor[] x, Tensor[] t, int batches)
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

    private static double[] Parameters(NeuralNetwork model)
    {
        var all = new List<double>();
        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is NeuronLayer byNeurons)
                foreach (Neuron n in byNeurons.Neurons) { all.AddRange(n.Weights); all.Add(n.Bias); }
            else if (layer is Conv conv)
            {
                foreach (var f in conv.Kernels) foreach (var r in f) foreach (var c in r) all.AddRange(c);
                all.AddRange(conv.Biases);
            }
        }
        return all.ToArray();
    }

    private static string TempPath() => Path.Combine(Path.GetTempPath(), "mdnn_opt_" + Guid.NewGuid().ToString("N") + ".json");

    private static NeuralNetwork SaveAndLoad(NeuralNetwork model, bool includeState = true,
                                           Optimizer? optimizer = null, Loss? loss = null)
    {
        string path = TempPath();
        try
        {
            model.SaveAsJson(path, includeState);
            return NeuralNetwork.LoadModel(path, optimizer: optimizer, loss: loss);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------ resuming training

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
        // second half of the data; Steps counts the index from zero, so shift it manually
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

        NeuralNetwork Cnn()
        {
            var m = new NeuralNetwork(new Dense(2, new Softmax()), new Adam(0.05), new CrossEntropy(), seed: 2);
            m.Layers.Add(new Conv(2, 2, new ReLU(), "valid"));
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
            $"without state {withoutState.Length} chars, with state {withState.Length}");

        // Without state this is exactly the old behaviour: resuming is NOT seamless.
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

    // ------------------------------------------------------------ changing the optimizer

    [Fact]
    public void Changing_the_context_optimizer_reaches_every_layer()
    {
        // Previously the change did not propagate: neurons kept training with SGD, but Adam was
        // saved to the file, so after loading the model trained with a different optimizer.
        var model = Mlp(new SGD(0.05));
        model.Layers.Add(new Conv(1, 1, new Linear(), "valid"));   // just so it has a Conv too
        model.GetResults(X[0]);
        Steps(model, X, T, 2);

        model.Context.Optimizer = new Adam(0.01);

        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is NeuronLayer byNeurons)
                Assert.All(byNeurons.Neurons, n => Assert.IsType<Adam>(n.Optimizer));
        }
        Assert.IsType<Adam>(model.Layers.Layers.OfType<Conv>().Single().OptimizerForSave);

        // and the saved model really continues with it
        var loaded = SaveAndLoad(model);
        Assert.All(((Dense)loaded.Layers.Layers[^1]).Neurons, n => Assert.IsType<Adam>(n.Optimizer));
    }

    [Fact]
    public void Reassigning_the_same_optimizer_keeps_its_state()
    {
        var model = Mlp(new Adam(0.05));
        Steps(model, X, T, 3);
        var neuron = ((Dense)model.Layers.Layers[0]).Neurons[0];
        double[] before = neuron.Optimizer.ExportState()!["m"];

        model.Context.Optimizer = model.Context.Optimizer;

        Assert.Equal(before, neuron.Optimizer.ExportState()!["m"]);
    }

    // ------------------------------------------------------------ custom optimizer and loss

    /// <summary>Custom optimizer with state (simplified momentum).</summary>
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
        // Fine-tuning with a lower learning rate: hyperparameters from the passed instance, state from the file.
        var model = Mlp(new Adam(0.05));
        Steps(model, X, T, 3);
        double[] m = ((Dense)model.Layers.Layers[0]).Neurons[0].Optimizer.ExportState()!["m"];

        var loaded = SaveAndLoad(model, optimizer: new Adam(0.001));
        var optimizer = ((Dense)loaded.Layers.Layers[0]).Neurons[0].Optimizer;

        Assert.Equal(0.001, optimizer.Hyperparameters[0]);
        Assert.Equal(m, optimizer.ExportState()!["m"]);
    }

    [Fact]
    public void TrainLoop_with_custom_optimizer_and_loss_restores_the_best_model()
    {
        // At the end of TrainLoop the best model is restored from a snapshot. That used to look up
        // the optimizer and loss by name, so with custom ones it failed with ArgumentException.
        var x = Enumerable.Range(0, 20).Select(i => new[] { Math.Sin(i), Math.Cos(i) }).ToArray();
        var y = x.Select(r => new[] { r[0] * r[1] }).ToArray();
        var model = new NeuralNetwork(new Dense(1, new Linear()), new Heavy(0.02), new HalfMse(), seed: 3);
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;

        model.Train.TrainLoop(x, y, 5, 4);   // TestAfterTraining stays on

        Assert.IsType<HalfMse>(model.Loss);
        Assert.IsType<Heavy>(((Dense)model.Layers.Layers[0]).Neurons[0].Optimizer);
    }

    // Loading used to overwrite the global GeneralNeuralNetworkSettings.DefaultOptimizer, which layer
    // constructors cloned from. A model loaded with Adam thus changed the default optimizer for the
    // whole process, and two loads on two threads could swap it under each other.
    [Fact]
    public void Loading_does_not_touch_the_global_default_optimizer()
    {
        Optimizer before = GeneralNeuralNetworkSettings.DefaultOptimizer;

        var loaded = SaveAndLoad(Mlp(new Adam(0.05)));

        Assert.Same(before, GeneralNeuralNetworkSettings.DefaultOptimizer);
        Assert.IsType<Adam>(((Dense)loaded.Layers.Layers[0]).Neurons[0].Optimizer);
    }

    // The instance passed to LoadModel is a PROTOTYPE: layers get clones of it and the state from
    // the file is loaded into those. The prototype itself must not be written to; anyone holding
    // the instance would share it, and one layer's state would look like the whole model's state.
    [Fact]
    public void Optimizer_passed_to_LoadModel_stays_an_empty_prototype()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 4);
        model.Layers.Add(new Conv(2, 2, new Tanh()));
        var image = new Tensor(new double[3, 3, 1] { { { 0.1 }, { 0.2 }, { 0.3 } }, { { 0.4 }, { 0.5 }, { 0.6 } }, { { 0.7 }, { 0.8 }, { 0.9 } } });
        for (int i = 0; i < 3; i++)
        {
            model.Train.Fit(image, new Tensor(new[] { 0.5, -0.5 }));
            model.Train.UpdateParams();
        }

        var prototype = new Adam(0.05);
        var loaded = SaveAndLoad(model, optimizer: prototype);

        Assert.Empty(prototype.ExportState()!["m"]);
        Assert.NotEmpty(((Conv)loaded.Layers.Layers[0]).OptimizerForSave.ExportState()!["m"]);
    }
}

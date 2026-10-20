using System;
using System.Linq;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// Optimizer state (Adam's moments, Momentum's velocity) belongs to the weights: as long as the
/// weights survive, so must the state.
///
/// Previously `MaterializeParameters` ALWAYS cloned the optimizer from the Context, even when the
/// neurons were not rebuilt at all. The old `Clone_optimizer` rebuilt the optimizer from its name
/// and hyperparameters, so `m`, `v`, `t` were thrown away. There were two triggers:
///
///   1. `SetInputShape()`, which `SimpleTrainLoop` calls on EVERY run
///   2. `Add`/`Insert`/`RemoveAt`: `MaterializeAll()` walks all layers, so the state was lost
///      even for layers the rewiring did not touch at all
///
/// The second one hurts code that changes the architecture incrementally (e.g. an AutoML runner):
/// every step silently discarded the accumulated moments of the whole model.
/// </summary>
public class OptimizerStateTests
{
    private static readonly Tensor X = new Tensor(new double[] { 1.0, 0.5 });
    private static readonly Tensor T = new Tensor(new double[] { 1.0 });

    private static void Step(NeuralNetwork model)
    {
        model.Train.Fit(X, T);
        model.Train.UpdateParams();
    }

    /// <summary>Model with a warmed-up Adam; after five steps the moments are no longer trivial.</summary>
    private static NeuralNetwork Warm(int hidden = 0)
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        if (hidden > 0) model.Layers.Add(new Dense(hidden, new ReLU()));

        for (int i = 0; i < 5; i++) Step(model);
        return model;
    }

    private static double Weight(NeuralNetwork model, int layer = 0)
        => ((Dense)model.Layers.Layers[layer]).Neurons[0].Weights[0];

    private static Optimizer OptimizerOf(NeuralNetwork model, int layer)
        => ((Dense)model.Layers.Layers[layer]).Neurons[0].Optimizer;

    // ------------------------------------------------------------ rewiring that changes nothing

    [Fact]
    public void Rewiring_that_changes_nothing_keeps_optimizer_state()
    {
        var reference = Warm();
        Step(reference);

        var rewired = Warm();
        rewired.Layers.SetInputShape();   // the shape does not change, it is only "rewired"
        Step(rewired);

        Assert.Equal(Weight(reference), Weight(rewired));
    }

    [Fact]
    public void Rewiring_that_changes_nothing_keeps_the_same_optimizer_instance()
    {
        var model = Warm();
        Optimizer before = OptimizerOf(model, 0);

        model.Layers.SetInputShape();

        Assert.Same(before, OptimizerOf(model, 0));
    }

    // ------------------------------------------------------------ architecture change

    [Fact]
    public void Adding_a_layer_keeps_the_optimizer_of_untouched_layers()
    {
        // Layers: [Dense(3) hidden, Dense(1) output]. Add(Dense(4)) inserts in the middle, so the
        // OUTPUT layer gets a different input count (3 → 4) and must be rebuilt, while the first
        // one does not change at all.
        var model = Warm(hidden: 3);
        Optimizer untouched = OptimizerOf(model, 0);
        Optimizer rebuilt = OptimizerOf(model, 1);

        model.Layers.Add(new Dense(4, new ReLU()));
        model.GetResults(X);   // triggers MaterializeAll

        Assert.Same(untouched, OptimizerOf(model, 0));
        Assert.NotSame(rebuilt, OptimizerOf(model, 2));   // the output layer is now index 2
    }

    [Fact]
    public void Removing_a_layer_keeps_the_optimizer_of_untouched_layers()
    {
        var model = Warm(hidden: 3);
        model.Layers.Add(new Dense(4, new ReLU()));
        model.GetResults(X);

        Optimizer untouched = OptimizerOf(model, 0);

        model.Layers.RemoveAt(1);
        model.GetResults(X);

        Assert.Same(untouched, OptimizerOf(model, 0));
    }

    // ------------------------------------------------------------ full training

    [Fact]
    public void SimpleTrainLoop_in_two_halves_matches_one_run()
    {
        // `SimpleTrainLoop` calls SetInputShape() unconditionally on every run,
        // so the second half used to start with a reset Adam.
        double[][] x = Enumerable.Range(1, 8).Select(i => new[] { i / 10.0 }).ToArray();
        double[][] y = x.Select(r => new[] { r[0] * 2 }).ToArray();

        NeuralNetwork Build()
        {
            var m = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.05), new MSE(), seed: 3);
            m.Train.AutoSave = false;
            return m;
        }

        var oneGo = Build();
        var inHalves = Build();

        var original = Console.Out;
        Console.SetOut(System.IO.TextWriter.Null);
        try
        {
            oneGo.Train.SimpleTrainLoop(x, y, 10, 4);

            inHalves.Train.SimpleTrainLoop(x, y, 5, 4);
            inHalves.Train.SimpleTrainLoop(x, y, 5, 4);
        }
        finally
        {
            Console.SetOut(original);
        }

        Assert.Equal(10u, oneGo.Train.CurrentEpoch);
        Assert.Equal(10u, inHalves.Train.CurrentEpoch);
        Assert.Equal(Weight(oneGo), Weight(inHalves));
    }

    // ------------------------------------------------------------ guards against overdoing it

    [Fact]
    public void Freshly_built_neurons_use_the_models_optimizer_not_the_global_default()
    {
        // In its constructor a neuron takes a clone of the GLOBAL optimizer (SGD with lr 0.0001).
        // Materialization must replace it with the one from the Context; otherwise a model with
        // Adam would train with step 0.0001 and it would look like "the model does not learn".
        // The hidden layer uses Tanh, not ReLU: with this initialization ReLU outputs all zeros,
        // so the output weights' gradient would be zero and even a correct Adam would not move.
        var model = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        model.Layers.Add(new Dense(3, new Tanh()));
        model.GetResults(X);                // layers are materialized only here

        double before = Weight(model, 1);   // output layer
        Step(model);
        double moved = Math.Abs(Weight(model, 1) - before);

        // Adam's first step is roughly lr (m̂/(√v̂+ε) ≈ ±1); the global SGD would give ~1e-4.
        Assert.True(moved > 0.05, $"the weight moved only by {moved:G6}; looks like the global SGD, not the model's Adam");
    }

    [Fact]
    public void Two_models_with_adam_do_not_share_momentum()
    {
        // Models are independent: every neuron must have its OWN optimizer instance.
        var a = Warm();
        var b = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        b.GetResults(X);   // materialization

        Assert.NotSame(OptimizerOf(a, 0), OptimizerOf(b, 0));

        // `b` is fresh, so its first step must be a full Adam step, regardless of
        // `a` having five steps behind it.
        double before = Weight(b);
        Step(b);

        Assert.True(Math.Abs(Weight(b) - before) > 0.05);
    }

    // ------------------------------------------------------------ EnsureCapacity

    /// <summary>Length of the optimizer's state array (Adam.m, Momentum.velocity).</summary>
    /// <remarks>Reflection on a private field is deliberate: the state size is not observable
    /// behaviour, and adding a public property just for a test is not wanted.</remarks>
    private static int StateLength(Optimizer optimizer)
    {
        string field = optimizer is Adam ? "m" : "velocity";
        var info = optimizer.GetType().GetField(field,
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return ((double[])info.GetValue(optimizer)!).Length;
    }

    public static TheoryData<Optimizer> StatefulOptimizers()
        => new() { new Adam(0.1), new Momentum(0.1, 0.9) };

    [Theory]
    [MemberData(nameof(StatefulOptimizers))]
    public void EnsureCapacity_keeps_state_that_is_already_accumulated(Optimizer optimizer)
    {
        // Growing the array mid-training must not reset the moments of parameters already in progress.
        Optimizer reference = optimizer.CreateFresh();

        double a = 1.0, b = 1.0;
        for (int step = 0; step < 3; step++)
        {
            a = optimizer.Update(a, 0.5, 0);
            b = reference.Update(b, 0.5, 0);
        }

        optimizer.EnsureCapacity(100);
        optimizer.EnsureCapacity(1);   // a smaller request must not shrink the array

        Assert.Equal(reference.Update(b, 0.5, 0), optimizer.Update(a, 0.5, 0));
        Assert.Equal(100, StateLength(optimizer));
    }

    [Fact]
    public void Dense_sizes_optimizer_state_exactly_to_weights_plus_bias()
    {
        // Guards against capacity doubling coming back (commit f6b8e05): 785 weights inflated
        // the state to 1024 elements and every UpdateParams cost ~3 % extra.
        var model = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        Step(model);

        Assert.Equal(X.Data.Length + 1, StateLength(OptimizerOf(model, 0)));
    }

    [Fact]
    public void Conv_sizes_optimizer_state_exactly_to_kernels_plus_biases()
    {
        var model = new NeuralNetwork(new Conv(2, 2, new Linear(), "valid"), new Adam(0.1), new MSE(), seed: 1);
        var image = new double[4, 4, 1];
        for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            image[i, j, 0] = 0.1 * (i + j);

        model.Train.Fit(new Tensor(image), new Tensor(new double[3 * 3 * 2]));
        model.Train.UpdateParams();

        var conv = model.Layers.Layers[0];
        var optimizer = (Optimizer)conv.GetType().GetField("optimizer",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(conv)!;

        // 2 filters × (2×2×1 kernel + bias)
        Assert.Equal(2 * (2 * 2 * 1 + 1), StateLength(optimizer));
    }
}

using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Losses;
using MDNN.Optimizers;
using System.Text.Json.Nodes;

namespace MDNN.Tests;

/// <summary>
/// Building a layer: a freshly constructed layer is only a DESCRIPTION (no weights), the model
/// builds it once the shape of the data is known, and a built model rejects data of a different
/// shape unless the user asks for a rebuild (Rebuild / RebuildIncompatible).
///
/// Previously the description was encoded in placeholder parameters (neurons with zero weights,
/// kernels with zero channels), the unknown shape was the sentinel {0}, five places built the
/// model each in their own way, and a different shape of data either silently re-randomized the
/// first layer or was multiplied with a part of its weights.
/// </summary>
public class BuildLifecycleTests : GradientCheckTestBase
{
    private static NeuralNetwork Quiet(NeuralNetwork model)
    {
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        model.Train.TestAfterTraining = false;
        model.Train.ConsoleReportCount = 1;
        return model;
    }

    private static T Silent<T>(Func<T> action)
    {
        var original = Console.Out;
        Console.SetOut(TextWriter.Null);
        try { return action(); }
        finally { Console.SetOut(original); }
    }

    private static void Silent(Action action) => Silent(() => { action(); return 0; });

    private static (double[][] x, double[][] y) Data(int width, int count = 24, int seed = 3)
    {
        var random = new Random(seed);
        double[][] x = Enumerable.Range(0, count)
            .Select(_ => Enumerable.Range(0, width).Select(_ => random.NextDouble()).ToArray())
            .ToArray();
        double[][] y = x.Select(v => new[] { v.Sum() / width }).ToArray();
        return (x, y);
    }

    private static NeuralNetwork TrainedMlp(int inputWidth)
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 5));
        model.Layers.Add(new Dense(4, new Tanh()));
        var (x, y) = Data(inputWidth);
        Silent(() => model.Train.SimpleTrainLoop(x, y, 2, 4));
        return model;
    }

    private static double[] WeightsOf(Layer layer) =>
        ((NeuronLayer)layer).Neurons.SelectMany(n => n.Weights.Append(n.Bias)).ToArray();

    // ------------------------------------------------------------ description before building

    [Fact]
    public void A_new_layer_is_only_a_description()
    {
        var dense = new Dense(64, new ReLU());
        var conv = new Conv(8, 3, new ReLU(), "same");

        Assert.False(dense.IsBuilt);
        Assert.Null(dense.InputShape);
        Assert.Equal(64, dense.NeuronCount);
        Assert.Empty(dense.Neurons);          // previously 64 placeholder neurons with zero weights

        Assert.False(conv.IsBuilt);
        Assert.Equal(8, conv.Filters);
        Assert.Equal(3, conv.KernelSize);
        Assert.Empty(conv.Kernels);           // previously kernels with zero input channels
    }

    [Fact]
    public void The_first_pass_builds_the_model_and_records_its_input()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), seed: 1);
        model.Layers.Add(new Dense(3, new ReLU()));
        Assert.Null(model.Context.InputShape);

        model.GetResults(new Tensor(new double[4, 5]));

        Assert.Equal(new[] { 4, 5 }, model.Context.InputShape);
        Assert.Equal(new[] { 20 }, model.Layers.Layers[0].InputShape);   // Dense flattens
        Assert.All(model.Layers.Layers, l => Assert.True(l.IsBuilt));
    }

    [Fact]
    public void A_loaded_model_knows_the_input_it_is_built_for()
    {
        var model = TrainedMlp(3);
        var loaded = NeuralNetwork.LoadModel(SaveToTemp(model));

        Assert.Equal(new[] { 3 }, loaded.Context.InputShape);
    }

    [Fact]
    public void A_dense_model_takes_the_same_data_flat_or_shaped()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), seed: 1);
        model.GetResults(new Tensor(new double[4, 5]));

        model.GetResults(new Tensor(new double[20]));   // the same 20 numbers, flat — fine for Dense
    }

    // ------------------------------------------------------------ data of a different shape

    [Fact]
    public void A_shorter_input_into_a_built_dense_is_an_error_not_a_partial_dot_product()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.GetResults(new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }));

        // previously: computed from the first 4 of 8 weights, without a word
        var ex = Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[] { 1, 2, 3, 4 })));
        Assert.Contains("Rebuild", ex.Message);
    }

    [Fact]
    public void A_longer_input_into_a_built_dense_is_an_error_not_IndexOutOfRange()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.GetResults(new Tensor(new double[] { 1, 2, 3, 4 }));

        Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[8])));
    }

    [Fact]
    public void SimpleTrainLoop_on_wider_data_is_an_error_and_keeps_the_trained_weights()
    {
        var model = TrainedMlp(3);
        double[] before = WeightsOf(model.Layers.Layers[0]);
        var (wider, y) = Data(5);

        // previously: the first layer silently got 5 new random weights per neuron
        var ex = Assert.Throws<ArgumentException>(() => Silent(() => model.Train.SimpleTrainLoop(wider, y, 1, 4)));
        Assert.Contains("RebuildIncompatible", ex.Message);
        Assert.Equal(before, WeightsOf(model.Layers.Layers[0]));
    }

    [Fact]
    public void TrainLoop_on_wider_data_is_an_error()
    {
        var model = TrainedMlp(3);
        var (wider, y) = Data(5);

        Assert.Throws<ArgumentException>(() => Silent(() => model.Train.TrainLoop(wider, y, 1, 4)));
    }

    [Fact]
    public void SetInputShape_cannot_change_the_shape_of_a_built_model()
    {
        var model = TrainedMlp(3);

        Assert.Throws<ArgumentException>(() => model.Layers.SetInputShape(new int[] { 5 }));
        model.Layers.SetInputShape(new int[] { 3 });   // the same shape is fine
    }

    // ------------------------------------------------------------ rebuilding on purpose

    [Fact]
    public void Rebuild_gives_every_layer_new_weights_for_the_new_shape()
    {
        var model = TrainedMlp(3);
        double[] outputBefore = WeightsOf(model.Layers.Layers[1]);

        model.Layers.Rebuild();
        Assert.All(model.Layers.Layers, l => Assert.False(l.IsBuilt));
        Assert.Null(model.Context.InputShape);

        var (wider, y) = Data(5);
        Silent(() => model.Train.SimpleTrainLoop(wider, y, 1, 4));

        Assert.Equal(5, ((Dense)model.Layers.Layers[0]).Neurons[0].Weights.Length);
        Assert.NotEqual(outputBefore, WeightsOf(model.Layers.Layers[1]));   // the output layer too
    }

    [Fact]
    public void RebuildIncompatible_keeps_the_layers_that_still_fit()
    {
        var model = TrainedMlp(3);
        double[] outputBefore = WeightsOf(model.Layers.Layers[1]);

        model.Layers.RebuildIncompatible();
        model.GetResults(new Tensor(new double[5]));

        Assert.Equal(5, ((Dense)model.Layers.Layers[0]).Neurons[0].Weights.Length);   // rebuilt
        Assert.Equal(outputBefore, WeightsOf(model.Layers.Layers[1]));                // kept, bit for bit
        Assert.Equal(new[] { 5 }, model.Context.InputShape);
    }

    [Fact]
    public void Both_rebuilds_can_build_right_away_for_a_given_shape()
    {
        var a = TrainedMlp(3);
        a.Layers.Rebuild(new int[] { 6 });
        Assert.All(a.Layers.Layers, l => Assert.True(l.IsBuilt));
        Assert.Equal(new[] { 6 }, a.Layers.Layers[0].InputShape);

        var b = TrainedMlp(3);
        b.Layers.RebuildIncompatible(new int[] { 6 });
        Assert.All(b.Layers.Layers, l => Assert.True(l.IsBuilt));
        Assert.Equal(new[] { 6 }, b.Layers.Layers[0].InputShape);
    }

    [Fact]
    public void Rebuild_with_a_seed_is_reproducible()
    {
        double[] Run()
        {
            var model = TrainedMlp(3);
            model.Layers.Rebuild(new int[] { 5 });
            return WeightsOf(model.Layers.Layers[0]);
        }

        Assert.Equal(Run(), Run());
    }

    // ------------------------------------------------------------ changing the structure

    [Fact]
    public void A_layer_that_does_not_fit_is_rejected_before_the_model_changes()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.Layers.Add(new Conv(2, 3, new ReLU()));
        model.GetResults(new Tensor(new double[4, 4, 1]));

        // a 5×5 kernel does not fit the 2×2 output of the first Conv
        Assert.Throws<ArgumentException>(() => model.Layers.Add(new Conv(2, 5, new ReLU())));

        Assert.Equal(2, model.Layers.Layers.Count);                 // previously the layer was already in the list
        Assert.All(model.Layers.Layers, l => Assert.True(l.IsBuilt));
        model.GetResults(new Tensor(new double[4, 4, 1]));          // and the model still works
    }

    [Fact]
    public void A_model_that_cannot_be_built_draws_nothing_and_builds_nothing()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.Layers.Add(new Dense(3, new ReLU()));
        model.Layers.Add(new Conv(2, 5, new ReLU()));   // after the Dense: [3] → 2×2 guess, too small

        Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[3])));

        Assert.All(model.Layers.Layers, l => Assert.False(l.IsBuilt));   // the chain is planned first
        model.Context.Seed = 7;   // and the generator was not touched, so the seed can still be set
    }

    [Fact]
    public void Inserting_a_layer_rebuilds_only_the_layer_whose_input_changed()
    {
        var model = TrainedMlp(3);
        Layer firstBefore = model.Layers.Layers[0];
        double[] outputBefore = WeightsOf(model.Layers.Layers[1]);

        model.Layers.Insert(0, new Dense(7, new ReLU()));   // the old first layer now gets 7 inputs
        Assert.False(firstBefore.IsBuilt);
        model.GetResults(new Tensor(new double[3]));

        Assert.Equal(7, ((Dense)firstBefore).Neurons[0].Weights.Length);
        Assert.Equal(outputBefore, WeightsOf(model.Layers.Layers[2]));   // its input (4) did not change
    }

    [Fact]
    public void Saving_a_model_that_is_not_built_says_so()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);

        var ex = Assert.Throws<InvalidOperationException>(() => model.SaveAsJsonString());
        Assert.Contains("not built", ex.Message);
    }

    // ------------------------------------------------------------ restoring the best model in place

    private sealed class Swish : ElementwiseActivationFunction
    {
        public override string Name => "Swish";
        public override double Apply(double x) => x / (1 + Math.Exp(-x));
        public override double Derivative(double x)
        {
            double s = 1 / (1 + Math.Exp(-x));
            return s + x * s * (1 - s);
        }
    }

    [Fact]
    public void TrainLoop_with_a_custom_activation_finishes()
    {
        // previously: the whole training ran and then restoring the best model crashed, because the
        // activation was looked up by name ("Swish is not built in")
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 2));
        var swish = new Swish();
        model.Layers.Add(new Dense(4, swish));
        var (x, y) = Data(3, 40);

        Silent(() => model.Train.TrainLoop(x, y, 3, 4));

        Assert.Same(swish, model.Layers.Layers[0].Activation);
    }

    [Fact]
    public void After_TrainLoop_the_users_layer_is_still_the_models_layer()
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.05), new MSE(), seed: 2));
        var hidden = new Dense(4, new ReLU());
        model.Layers.Add(hidden);
        LayerManager layers = model.Layers;
        var (x, y) = Data(3, 40);

        Silent(() => model.Train.TrainLoop(x, y, 6, 4));

        // previously: new objects from the snapshot; `hidden` kept the weights of the LAST epoch
        Assert.Same(hidden, model.Layers.Layers[0]);
        Assert.Same(layers, model.Layers);
    }

    [Fact]
    public void Restoring_in_place_gives_the_same_model_as_loading_the_snapshot()
    {
        var model = Quiet(new NeuralNetwork(new Dense(2, new Softmax()), new Adam(0.05), new CrossEntropy(), seed: 4));
        model.Layers.Add(new Dense(5, new ReLU()));
        var (x, _) = Data(3, 40);
        double[][] y = x.Select(v => v[0] > 0.5 ? new[] { 1.0, 0.0 } : new[] { 0.0, 1.0 }).ToArray();

        // Trained with manual steps, not TrainLoop: TrainLoop itself ends with an in-place restore,
        // so the snapshot would already have gone through the code under test — and a bug there
        // (e.g. the optimizer state not being loaded) would hit both sides of the comparison alike.
        void Steps(int from, int count)
        {
            for (int i = from; i < from + count; i++)
            {
                model.Train.Fit(new Tensor(x[i]), new Tensor(y[i]));
                if (i % 4 == 3) model.Train.UpdateParams();
            }
        }

        Steps(0, 16);
        string snapshot = model.SaveAsJsonString();
        Steps(16, 16);   // move away from the snapshot

        model.LoadWeightsFromString(snapshot);
        var reference = NeuralNetwork.LoadModel(SaveToTemp(snapshot));

        for (int i = 0; i < model.Layers.Layers.Count; i++)
        {
            Assert.Equal(WeightsOf(reference.Layers.Layers[i]), WeightsOf(model.Layers.Layers[i]));
        }

        // the optimizer state goes with it: one more identical step gives identical weights
        var sample = new Tensor(x[0]);
        var target = new Tensor(y[0]);
        model.Train.Fit(sample, target); model.Train.UpdateParams();
        reference.Train.Fit(sample, target); reference.Train.UpdateParams();
        Assert.Equal(WeightsOf(reference.Layers.Layers[0]), WeightsOf(model.Layers.Layers[0]));
    }

    [Fact]
    public void LoadWeightsInPlace_with_a_different_architecture_still_replaces_the_layers()
    {
        var small = TrainedMlp(3);
        var other = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 9));
        other.Layers.Add(new Dense(6, new ReLU()));
        other.Layers.Add(new Dense(2, new ReLU()));
        other.GetResults(new Tensor(new double[5]));

        small.LoadWeightsInPlace(SaveToTemp(other));

        Assert.Equal(3, small.Layers.Layers.Count);
        Assert.Equal(new[] { 5 }, small.Context.InputShape);
    }

    // ------------------------------------------------------------ parallel training

    [Fact]
    public void A_layer_added_to_a_trained_model_is_trained_in_parallel_too()
    {
        var model = TrainedMlp(3);
        model.Layers.Add(new Dense(6, new Tanh()));
        model.Train.Workers = 3;
        var (x, y) = Data(3, 40);

        Silent(() => model.Train.TrainLoop(x, y, 2, 8));

        // previously: the replicas built their own copies of the new layer and trained those; the
        // original kept 6 placeholder neurons with ZERO weights and the training was lost
        var added = (Dense)model.Layers.Layers[1];
        Assert.True(added.IsBuilt);
        Assert.Equal(4, added.Neurons[0].Weights.Length);
        Assert.Equal(6, ((Dense)model.Layers.Layers[2]).Neurons[0].Weights.Length);
    }

    // ------------------------------------------------------------ self-CR fixes

    [Fact]
    public void A_model_of_layers_built_beforehand_can_be_trained()
    {
        // previously: every layer was built, so nothing recorded the model's input and training
        // refused to start ("The input layer must have at least one input")
        var output = new Dense(1, new Linear());
        output.Build(new[] { 3 });
        var model = Quiet(new NeuralNetwork(output, new SGD(0.1), new MSE(), seed: 1));

        model.Train.Fit(new Tensor(new double[] { 1, 2, 3 }), new Tensor(new double[] { 1 }));
        model.Train.UpdateParams();

        Assert.Equal(new[] { 3 }, model.Context.InputShape);
    }

    [Fact]
    public void A_layer_built_beforehand_for_another_shape_is_reported()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        var hidden = new Dense(4, new ReLU());
        hidden.Build(new[] { 5 });
        model.Layers.Add(hidden);

        var ex = Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[3])));
        Assert.Contains("built for input 5", ex.Message);
    }

    // The model file without its envelope (old format v0, no checksum), so it can be damaged on purpose.
    private static JsonNode FileOf(NeuralNetwork model) =>
        JsonNode.Parse(model.SaveAsJsonString())!["Model"]!.DeepClone();

    private static NeuralNetwork TrainedWithAdam()
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 1));
        model.Layers.Add(new Dense(3, new ReLU()));
        for (int i = 0; i < 3; i++)
        {
            model.Train.Fit(new Tensor(new double[] { 1, 2, 3 }), new Tensor(new double[] { 1 }));
            model.Train.UpdateParams();
        }
        return model;
    }

    [Theory]
    [InlineData("optimizer state")]
    [InlineData("dropout")]
    public void A_damaged_file_leaves_the_model_untouched(string damage)
    {
        var model = TrainedWithAdam();
        JsonNode file = FileOf(model);
        file["Layers"]![0]!["Neurons"]![0]!["Weights"]![0] = 123.0;   // would show a partial load
        if (damage == "optimizer state")
            file["Layers"]![1]!["Neurons"]![0]!["OptimizerState"]!["m"] = new JsonArray(1.0);   // wrong length
        else
            file["Layers"]![1]!["DropoutRate"] = 1.5;

        double[] before = WeightsOf(model.Layers.Layers[0]);
        Optimizer optimizerBefore = model.Context.Optimizer;

        Assert.ThrowsAny<ArgumentException>(() => model.LoadWeightsFromString(file.ToJsonString()));

        // previously: the first layer already had the file's weights when the second one threw
        Assert.Equal(before, WeightsOf(model.Layers.Layers[0]));
        Assert.Same(optimizerBefore, model.Context.Optimizer);
    }

    [Fact]
    public void After_a_failed_load_a_new_optimizer_still_reaches_the_layers()
    {
        var model = TrainedWithAdam();
        JsonNode file = FileOf(model);
        file["Layers"]![1]!["Neurons"]![0]!["OptimizerState"]!["m"] = new JsonArray(1.0);
        Assert.ThrowsAny<ArgumentException>(() => model.LoadWeightsFromString(file.ToJsonString()));

        model.Context.Optimizer = new SGD(0.5);

        // previously: the change-notification stayed disconnected and the neurons kept their Adam
        Assert.All(((Dense)model.Layers.Layers[0]).Neurons, n => Assert.IsType<SGD>(n.Optimizer));
    }

    [Fact]
    public void A_failed_load_of_another_architecture_leaves_the_model_untouched()
    {
        var model = TrainedWithAdam();
        var other = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 2);
        other.GetResults(new Tensor(new double[4]));
        JsonNode file = FileOf(other);
        file["Layers"]![0]!["Name_of_activation_function"] = "NoSuchActivation";

        Layer first = model.Layers.Layers[0];
        Optimizer optimizerBefore = model.Context.Optimizer;

        Assert.ThrowsAny<ArgumentException>(() => model.LoadWeightsFromString(file.ToJsonString()));

        Assert.Same(first, model.Layers.Layers[0]);
        Assert.Same(optimizerBefore, model.Context.Optimizer);   // previously: the file's SGD
        model.Context.Optimizer = new Momentum(0.1, 0.9);
        Assert.All(((Dense)first).Neurons, n => Assert.IsType<Momentum>(n.Optimizer));
    }

    [Fact]
    public void After_Clear_the_model_takes_data_of_any_shape_again()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), seed: 1);
        model.GetResults(new Tensor(new double[8]));

        model.Layers.Clear(new Dense(3, new Linear()));
        Assert.Null(model.Context.InputShape);   // nothing built is left, so nothing to protect

        // previously (this branch): "The model is built for input 8"; on main: 8 weights × 4 inputs
        model.GetResults(new Tensor(new double[4]));
        Assert.Equal(4, ((Dense)model.Layers.Layers[0]).Neurons[0].Weights.Length);
    }

    [Fact]
    public void Loading_weights_cancels_a_pending_RebuildIncompatible()
    {
        var model = TrainedMlp(3);
        string file = model.SaveAsJsonString();

        model.Layers.RebuildIncompatible();
        model.LoadWeightsFromString(file);

        // previously: the pending rebuild threw the just-loaded weights away on the next sample
        Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[5])));
    }

    [Fact]
    public void A_rebuilt_RNN_starts_from_an_empty_hidden_state()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.Layers.Add(new RNN(3, new Tanh()));
        var x = new Tensor(new double[] { 0.5, -0.3 });
        for (int i = 0; i < 5; i++) model.GetResults(x);   // builds up a hidden state

        model.Layers.Rebuild(new[] { 2 });
        double[] first = model.GetResults(x).Data;
        model.ResetSequence();
        double[] fromEmptyState = model.GetResults(x).Data;

        // previously: the first step after the rebuild used h(t-1) of the DISCARDED weights
        Assert.Equal(fromEmptyState, first);
    }

    [Fact]
    public void Building_for_a_zero_size_is_rejected()
    {
        var dense = new Dense(4);
        var conv = new Conv(2, 3, new ReLU(), "same");

        // previously: a "built" Dense with zero weights per neuron
        Assert.Throws<ArgumentException>(() => dense.Build(new[] { 0 }));
        Assert.Throws<ArgumentException>(() => conv.Build(new[] { 0, 0, 1 }));
        Assert.False(dense.IsBuilt);
        Assert.False(conv.IsBuilt);
    }

    // ------------------------------------------------------------ helpers

    private static string SaveToTemp(NeuralNetwork model)
    {
        string path = Path.Combine(Path.GetTempPath(), "mdnn_build_" + Guid.NewGuid().ToString("N") + ".json");
        model.SaveAsJson(path);
        return path;
    }

    private static string SaveToTemp(string json)
    {
        string path = Path.Combine(Path.GetTempPath(), "mdnn_build_" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(path, json);
        return path;
    }
}

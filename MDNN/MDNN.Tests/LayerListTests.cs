using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Losses;
using MDNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// A model can be given its hidden layers as a list (the output layer separately, or from the data), and its architecture can be
/// replaced as a whole with <c>Layers.ReplaceAll</c>. Every layer instance may be in one model, at one
/// position — the same instance twice shares the state of one pass and the gradients come out wrong
/// (measured: the opposite sign).
/// </summary>
public class LayerListTests : GradientCheckTestBase
{
    private static double[] Parameters(NeuralNetwork model) =>
        model.Layers.Layers.OfType<NeuronLayer>()
             .SelectMany(l => l.Neurons.SelectMany(n => n.Weights.Append(n.Bias)))
             .ToArray();

    private static void TrainFewSteps(NeuralNetwork model)
    {
        var random = new Random(5);
        for (int i = 0; i < 12; i++)
        {
            double[] x = { random.NextDouble(), random.NextDouble(), random.NextDouble() };
            model.Train.Fit(new Tensor(x), new Tensor(new[] { x.Sum() }));
            if (i % 4 == 3) model.Train.UpdateParams();
        }
    }

    // ------------------------------------------------------------ the list constructor

    [Fact]
    public void A_model_from_a_list_is_the_same_as_one_assembled_with_Add()
    {
        var fromList = new NeuralNetwork(
            [new Dense(5, new ReLU()), new Dense(4, new Tanh())],
            new Adam(0.01), new MSE(), seed: 8, outputLayer: new Dense(1, new Linear()));

        var withAdd = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 8);
        withAdd.Layers.Add(new Dense(5, new ReLU()));
        withAdd.Layers.Add(new Dense(4, new Tanh()));

        TrainFewSteps(fromList);
        TrainFewSteps(withAdd);

        Assert.Equal(Parameters(withAdd), Parameters(fromList));   // bit for bit
    }

    [Fact]
    public void The_list_holds_the_hidden_layers_and_outputLayer_the_output_layer()
    {
        var hidden = new Dense(5, new ReLU());
        var output = new Dense(2, new Softmax());

        var model = new NeuralNetwork(new Layer[] { hidden }, new SGD(0.1), new CrossEntropy(), outputLayer: output);

        Assert.Same(hidden, model.Layers.Layers[0]);
        Assert.Same(output, model.Layers.Layers[1]);
        Assert.False(hidden.IsOutputLayer);
        Assert.True(output.IsOutputLayer);
        Assert.Equal("[Dense,Dense]", model.Schema);
    }

    [Fact]
    public void More_layers_can_still_be_added_to_a_model_from_a_list()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU())], seed: 1, outputLayer: new Dense(1, new Linear()));

        model.Layers.Add(new Dense(3, new Tanh()));   // just before the output layer

        Assert.Equal(new[] { 5, 3, 1 }, model.Layers.Layers.Cast<Dense>().Select(d => d.NeuronCount));
    }

    [Fact]
    public void An_invalid_list_is_rejected()
    {
        Assert.Throws<ArgumentNullException>(() => new NeuralNetwork((IEnumerable<Layer>)null!));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork(new Layer[] { new Dense(2), null! }));

        // an empty list of HIDDEN layers is fine: the model is just its output layer
        Assert.Single(new NeuralNetwork(Array.Empty<Layer>()).Layers.Layers);
    }

    // ------------------------------------------------------------ one instance, one place

    [Fact]
    public void The_same_instance_twice_in_a_list_is_rejected()
    {
        var shared = new Dense(3, new Tanh());

        var ex = Assert.Throws<ArgumentException>(() =>
            new NeuralNetwork([shared, shared]));
        Assert.Contains("already in the model", ex.Message);
    }

    [Fact]
    public void Adding_or_inserting_an_instance_that_is_already_in_the_model_is_rejected()
    {
        var shared = new Dense(3, new Tanh());
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.Layers.Add(shared);

        // previously allowed: two positions sharing weights and buffers → gradient with the wrong sign
        Assert.Throws<ArgumentException>(() => model.Layers.Add(shared));
        Assert.Throws<ArgumentException>(() => model.Layers.Hidden.Insert(0, shared));
        Assert.Throws<ArgumentException>(() => model.Layers.Add(model.Layers.Layers[1]));   // the output layer too

        Assert.Equal(2, model.Layers.Layers.Count);   // the model did not change
    }

    [Fact]
    public void A_layer_of_another_model_is_rejected()
    {
        var shared = new Dense(3, new Tanh());
        var first = new NeuralNetwork([shared]);
        var second = new NeuralNetwork(new Dense(1, new Linear()));

        var ex = Assert.Throws<ArgumentException>(() => second.Layers.Add(shared));
        Assert.Contains("another model", ex.Message);
        Assert.Throws<ArgumentException>(() => new NeuralNetwork([shared]));
        Assert.Same(first.Context, shared.Context);
    }

    [Fact]
    public void A_removed_layer_can_go_into_another_model()
    {
        var moved = new Dense(3, new Tanh());
        var first = new NeuralNetwork([moved]);
        var second = new NeuralNetwork(new Dense(1, new Linear()));

        first.Layers.Hidden.RemoveAt(0);
        Assert.Null(moved.Context);   // detached

        second.Layers.Add(moved);
        Assert.Same(second.Context, moved.Context);
    }

    // ------------------------------------------------------------ Replace

    [Fact]
    public void ReplaceAll_gives_the_model_exactly_the_new_layers_and_keeps_its_settings()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU())], new Adam(0.01), new MSE(), seed: 3, outputLayer: new Dense(1, new Linear()));
        TrainFewSteps(model);
        Optimizer optimizer = model.Context.Optimizer;
        Layer oldOutput = model.Layers.Layers[1];

        var hidden = new Dense(7, new Tanh());
        var output = new Dense(2, new Linear());
        model.Layers.ReplaceAll([hidden, output]);

        Assert.Equal(new Layer[] { hidden, output }, model.Layers.Layers);
        Assert.True(output.IsOutputLayer);
        Assert.Same(optimizer, model.Context.Optimizer);
        Assert.Null(model.Context.InputShape);   // nothing built is left
        Assert.Null(oldOutput.Context);          // the old layers are detached

        model.GetResults(new Tensor(new double[] { 1, 2, 3, 4 }));   // the new architecture takes any input
        Assert.Equal(4, hidden.Neurons[0].Weights.Length);
    }

    [Fact]
    public void ReplaceAll_may_reuse_a_layer_of_the_model_and_it_keeps_its_weights()
    {
        var first = new Dense(5, new ReLU());
        var model = new NeuralNetwork([first], new Adam(0.01), new MSE(), seed: 3, outputLayer: new Dense(1, new Linear()));
        TrainFewSteps(model);
        double[] before = first.Neurons.SelectMany(n => n.Weights).ToArray();

        model.Layers.ReplaceAll([first, new Dense(2, new Tanh()), new Dense(1, new Linear())]);

        Assert.True(first.IsBuilt);   // its input (3) did not change
        Assert.Equal(before, first.Neurons.SelectMany(n => n.Weights).ToArray());
    }

    [Fact]
    public void ReplaceAll_rejects_an_invalid_list_and_leaves_the_model_as_it_was()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU())], seed: 3, outputLayer: new Dense(1, new Linear()));
        var layers = model.Layers.Layers.ToArray();
        var shared = new Dense(2);

        Assert.Throws<ArgumentException>(() => model.Layers.ReplaceAll([shared, shared]));
        Assert.Throws<ArgumentException>(() => model.Layers.ReplaceAll(Array.Empty<Layer>()));

        Assert.Equal(layers, model.Layers.Layers);
        Assert.All(layers, l => Assert.Same(model.Context, l.Context));
    }

    // ------------------------------------------------------------ hidden layers vs the output layer

    private static NeuralNetwork ThreeLayers(out Dense first, out Dense second, out Dense output)
    {
        first = new Dense(5, new ReLU());
        second = new Dense(4, new Tanh());
        output = new Dense(1, new Linear());
        return new NeuralNetwork([first, second], new Adam(0.01), new MSE(), seed: 3, outputLayer: output);
    }

    [Fact]
    public void Hidden_and_Output_split_the_model()
    {
        var model = ThreeLayers(out var first, out var second, out var output);

        Assert.Equal(new Layer[] { first, second }, model.Layers.Hidden);
        Assert.Equal(2, model.Layers.Hidden.Count);
        Assert.Same(second, model.Layers.Hidden[1]);
        Assert.Same(output, model.Layers.Output);
        Assert.Equal(new Layer[] { first, second, output }, model.Layers.Layers);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Hidden[2]);   // not the output layer
    }

    [Fact]
    public void Hidden_Insert_at_the_end_goes_before_the_output_layer_never_after()
    {
        var model = ThreeLayers(out _, out _, out var output);
        var added = new Dense(3, new ReLU());

        // previously Insert(Count, x) on the whole chain made x the new output layer
        model.Layers.Hidden.Insert(model.Layers.Hidden.Count, added);

        Assert.Same(added, model.Layers.Hidden[2]);
        Assert.Same(output, model.Layers.Output);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Hidden.Insert(4, new Dense(2)));
    }

    [Fact]
    public void Hidden_RemoveAt_cannot_remove_the_output_layer()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU())], outputLayer: new Dense(1, new Linear()));
        Layer output = model.Layers.Output;

        // previously RemoveAt(last) removed the output layer and the hidden one took its place
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Hidden.RemoveAt(1));

        model.Layers.Hidden.RemoveAt(0);
        Assert.Empty(model.Layers.Hidden);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Hidden.RemoveAt(0));
        Assert.Same(output, model.Layers.Output);
    }

    [Fact]
    public void Hidden_Clear_keeps_the_output_layer()
    {
        var model = ThreeLayers(out var first, out _, out var output);

        model.Layers.Hidden.Clear();

        Assert.Equal(new Layer[] { output }, model.Layers.Layers);
        Assert.True(output.IsOutputLayer);
        Assert.Null(first.Context);   // detached
    }

    [Fact]
    public void Hidden_Replace_keeps_the_output_layer()
    {
        var model = ThreeLayers(out _, out _, out var output);
        var hidden = new Dense(7, new Tanh());

        model.Layers.Hidden.Replace([hidden]);

        Assert.Equal(new Layer[] { hidden, output }, model.Layers.Layers);
        Assert.Throws<ArgumentException>(() => model.Layers.Hidden.Replace([output]));   // the output layer is not hidden
    }

    [Fact]
    public void ReplaceOutput_swaps_only_the_output_layer_and_the_hidden_ones_keep_their_weights()
    {
        var model = ThreeLayers(out var first, out var second, out var oldOutput);
        TrainFewSteps(model);
        double[] hiddenBefore = Parameters(model).Take(5 * 4 + 4 * 6).ToArray();   // first (5×(3+1)) + second (4×(5+1))

        var newOutput = new Dense(2, new Linear());
        model.Layers.ReplaceOutput(newOutput);

        Assert.Same(newOutput, model.Layers.Output);
        Assert.True(newOutput.IsOutputLayer);
        Assert.False(second.IsOutputLayer);
        Assert.Null(oldOutput.Context);   // detached
        Assert.True(first.IsBuilt);
        Assert.True(second.IsBuilt);

        var y = model.GetResults(new Tensor(new double[] { 1, 2, 3 }));
        Assert.Equal(2, y.Data.Length);
        Assert.Equal(hiddenBefore, Parameters(model).Take(hiddenBefore.Length).ToArray());
    }

    [Fact]
    public void ReplaceOutput_rejects_a_hidden_layer_and_ignores_the_current_output_layer()
    {
        var model = ThreeLayers(out var first, out _, out var output);

        Assert.Throws<ArgumentException>(() => model.Layers.ReplaceOutput(first));
        model.Layers.ReplaceOutput(output);   // no-op

        Assert.Equal(3, model.Layers.Layers.Count);
        Assert.Same(output, model.Layers.Output);
    }

    [Fact]
    public void Setting_a_hidden_layer_by_index_replaces_just_that_layer()
    {
        var model = ThreeLayers(out var first, out var second, out var output);
        TrainFewSteps(model);
        double[] firstBefore = first.Neurons.SelectMany(n => n.Weights).ToArray();
        double[] outputBefore = output.Neurons.SelectMany(n => n.Weights).ToArray();

        var replacement = new Dense(4, new ReLU());   // same width as `second` → the output layer's input stays
        model.Layers.Hidden[1] = replacement;

        Assert.Equal(new Layer[] { first, replacement, output }, model.Layers.Layers);
        Assert.Null(second.Context);   // the old one is detached
        Assert.False(replacement.IsBuilt);

        model.GetResults(new Tensor(new double[] { 1, 2, 3 }));
        Assert.Equal(5, replacement.Neurons[0].Weights.Length);
        Assert.Equal(firstBefore, first.Neurons.SelectMany(n => n.Weights).ToArray());     // kept
        Assert.Equal(outputBefore, output.Neurons.SelectMany(n => n.Weights).ToArray());   // kept
    }

    [Fact]
    public void Setting_a_wider_hidden_layer_rebuilds_the_next_layer()
    {
        var model = ThreeLayers(out _, out _, out var output);
        TrainFewSteps(model);

        model.Layers.Hidden[1] = new Dense(9, new ReLU());

        Assert.False(output.IsBuilt);   // its input changed from 4 to 9
        model.GetResults(new Tensor(new double[] { 1, 2, 3 }));
        Assert.Equal(9, output.Neurons[0].Weights.Length);
    }

    [Fact]
    public void The_index_cannot_reach_the_output_layer()
    {
        var model = ThreeLayers(out _, out _, out var output);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Hidden[2] = new Dense(1, new Linear()));
        Assert.Same(output, model.Layers.Output);
    }

    [Fact]
    public void Setting_by_index_checks_for_duplicates_but_allows_the_same_layer_back()
    {
        var model = ThreeLayers(out var first, out var second, out var output);

        Assert.Throws<ArgumentException>(() => model.Layers.Hidden[0] = second);   // already at position 1
        Assert.Throws<ArgumentException>(() => model.Layers.Hidden[0] = output);   // the output layer

        model.Layers.Hidden[0] = first;   // no-op
        Assert.Equal(3, model.Layers.Layers.Count);
        Assert.Same(model.Context, first.Context);
    }

    [Fact]
    public void Layers_Add_is_a_shortcut_for_Hidden_Add()
    {
        var model = ThreeLayers(out _, out _, out var output);
        var added = new Dense(3, new ReLU());

        model.Layers.Add(added);

        Assert.Same(added, model.Layers.Hidden[^1]);
        Assert.Same(output, model.Layers.Output);
    }
}

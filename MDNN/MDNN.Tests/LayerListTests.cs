using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Losses;
using MDNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// A model can be given its layers as a list (last = output layer), and its architecture can be
/// replaced as a whole with <c>Layers.Replace</c>. Every layer instance may be in one model, at one
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
            [new Dense(5, new ReLU()), new Dense(4, new Tanh()), new Dense(1, new Linear())],
            new Adam(0.01), new MSE(), seed: 8);

        var withAdd = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), new MSE(), seed: 8);
        withAdd.Layers.Add(new Dense(5, new ReLU()));
        withAdd.Layers.Add(new Dense(4, new Tanh()));

        TrainFewSteps(fromList);
        TrainFewSteps(withAdd);

        Assert.Equal(Parameters(withAdd), Parameters(fromList));   // bit for bit
    }

    [Fact]
    public void The_last_layer_of_the_list_is_the_output_layer()
    {
        var hidden = new Dense(5, new ReLU());
        var output = new Dense(2, new Softmax());

        var model = new NeuralNetwork(new Layer[] { hidden, output }, new SGD(0.1), new CrossEntropy());

        Assert.Same(hidden, model.Layers.Layers[0]);
        Assert.Same(output, model.Layers.Layers[1]);
        Assert.False(hidden.IsOutputLayer);
        Assert.True(output.IsOutputLayer);
        Assert.Equal("[Dense,Dense]", model.Schema);
    }

    [Fact]
    public void More_layers_can_still_be_added_to_a_model_from_a_list()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU()), new Dense(1, new Linear())], seed: 1);

        model.Layers.Add(new Dense(3, new Tanh()));   // just before the output layer

        Assert.Equal(new[] { 5, 3, 1 }, model.Layers.Layers.Cast<Dense>().Select(d => d.NeuronCount));
    }

    [Fact]
    public void An_invalid_list_is_rejected()
    {
        Assert.Throws<ArgumentNullException>(() => new NeuralNetwork((IEnumerable<Layer>)null!));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork(Array.Empty<Layer>()));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork(new Layer[] { new Dense(2), null! }));
    }

    // ------------------------------------------------------------ one instance, one place

    [Fact]
    public void The_same_instance_twice_in_a_list_is_rejected()
    {
        var shared = new Dense(3, new Tanh());

        var ex = Assert.Throws<ArgumentException>(() =>
            new NeuralNetwork([shared, shared, new Dense(1, new Linear())]));
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
        Assert.Throws<ArgumentException>(() => model.Layers.Insert(0, shared));
        Assert.Throws<ArgumentException>(() => model.Layers.Add(model.Layers.Layers[1]));   // the output layer too

        Assert.Equal(2, model.Layers.Layers.Count);   // the model did not change
    }

    [Fact]
    public void A_layer_of_another_model_is_rejected()
    {
        var shared = new Dense(3, new Tanh());
        var first = new NeuralNetwork([shared, new Dense(1, new Linear())]);
        var second = new NeuralNetwork(new Dense(1, new Linear()));

        var ex = Assert.Throws<ArgumentException>(() => second.Layers.Add(shared));
        Assert.Contains("another model", ex.Message);
        Assert.Throws<ArgumentException>(() => new NeuralNetwork([shared, new Dense(1, new Linear())]));
        Assert.Same(first.Context, shared.Context);
    }

    [Fact]
    public void A_removed_layer_can_go_into_another_model()
    {
        var moved = new Dense(3, new Tanh());
        var first = new NeuralNetwork([moved, new Dense(1, new Linear())]);
        var second = new NeuralNetwork(new Dense(1, new Linear()));

        first.Layers.RemoveAt(0);
        Assert.Null(moved.Context);   // detached

        second.Layers.Add(moved);
        Assert.Same(second.Context, moved.Context);
    }

    // ------------------------------------------------------------ Replace

    [Fact]
    public void Replace_gives_the_model_exactly_the_new_layers_and_keeps_its_settings()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU()), new Dense(1, new Linear())], new Adam(0.01), new MSE(), seed: 3);
        TrainFewSteps(model);
        Optimizer optimizer = model.Context.Optimizer;
        Layer oldOutput = model.Layers.Layers[1];

        var hidden = new Dense(7, new Tanh());
        var output = new Dense(2, new Linear());
        model.Layers.Replace([hidden, output]);

        Assert.Equal(new Layer[] { hidden, output }, model.Layers.Layers);
        Assert.True(output.IsOutputLayer);
        Assert.Same(optimizer, model.Context.Optimizer);
        Assert.Null(model.Context.InputShape);   // nothing built is left
        Assert.Null(oldOutput.Context);          // the old layers are detached

        model.GetResults(new Tensor(new double[] { 1, 2, 3, 4 }));   // the new architecture takes any input
        Assert.Equal(4, hidden.Neurons[0].Weights.Length);
    }

    [Fact]
    public void Replace_may_reuse_a_layer_of_the_model_and_it_keeps_its_weights()
    {
        var first = new Dense(5, new ReLU());
        var model = new NeuralNetwork([first, new Dense(1, new Linear())], new Adam(0.01), new MSE(), seed: 3);
        TrainFewSteps(model);
        double[] before = first.Neurons.SelectMany(n => n.Weights).ToArray();

        model.Layers.Replace([first, new Dense(2, new Tanh()), new Dense(1, new Linear())]);

        Assert.True(first.IsBuilt);   // its input (3) did not change
        Assert.Equal(before, first.Neurons.SelectMany(n => n.Weights).ToArray());
    }

    [Fact]
    public void Replace_rejects_an_invalid_list_and_leaves_the_model_as_it_was()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU()), new Dense(1, new Linear())], seed: 3);
        var layers = model.Layers.Layers.ToArray();
        var shared = new Dense(2);

        Assert.Throws<ArgumentException>(() => model.Layers.Replace([shared, shared]));
        Assert.Throws<ArgumentException>(() => model.Layers.Replace(Array.Empty<Layer>()));

        Assert.Equal(layers, model.Layers.Layers);
        Assert.All(layers, l => Assert.Same(model.Context, l.Context));
    }

    [Fact]
    public void Clear_with_an_output_layer_is_Replace_with_one_layer()
    {
        var model = new NeuralNetwork([new Dense(5, new ReLU()), new Dense(1, new Linear())], seed: 3);
        var output = new Dense(2, new Linear());

        model.Layers.Clear(output);

        Assert.Equal(new Layer[] { output }, model.Layers.Layers);
        Assert.True(output.IsOutputLayer);
    }
}

using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Losses;
using MDNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// The output layer a model gets when none is given: an ordinary Dense whose size comes from the
/// targets (like the input adapts to the data) and whose activation comes from the loss
/// (CrossEntropy → Softmax, MSE → Linear). Created with <c>Dense.FromTargets()</c>.
/// </summary>
public class AutoOutputTests : GradientCheckTestBase
{
    private static T Quiet<T>(T model) where T : NeuralNetwork
    {
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        model.Train.TestAfterTraining = false;
        model.Train.ConsoleReportCount = 1;
        return model;
    }

    private static void Silent(Action action)
    {
        var original = Console.Out;
        Console.SetOut(TextWriter.Null);
        try { action(); }
        finally { Console.SetOut(original); }
    }

    // 3 inputs; `classes` one-hot classes
    private static (double[][] x, double[][] y) Classes(int classes, int count = 30, int seed = 4)
    {
        var random = new Random(seed);
        double[][] x = Enumerable.Range(0, count).Select(_ => new[] { random.NextDouble(), random.NextDouble(), random.NextDouble() }).ToArray();
        double[][] y = x.Select(v => { var t = new double[classes]; t[(int)(v.Sum() * 10) % classes] = 1; return t; }).ToArray();
        return (x, y);
    }

    private static double[] Parameters(NeuralNetwork model) =>
        model.Layers.Layers.OfType<NeuronLayer>()
             .SelectMany(l => l.Neurons.SelectMany(n => n.Weights.Append(n.Bias)))
             .ToArray();

    private static Dense AutoOutput(NeuralNetwork model) => Assert.IsType<Dense>(model.Layers.Output);

    // ------------------------------------------------------------ size and activation from the data

    [Fact]
    public void A_classifier_gets_its_output_from_the_targets_and_Softmax_from_CrossEntropy()
    {
        var model = Quiet(new NeuralNetwork(new Adam(0.01), new CrossEntropy(), seed: 1));
        model.Layers.Add(new Dense(8, new ReLU()));
        var (x, y) = Classes(3);

        Silent(() => model.Train.TrainLoop(x, y, 2, 4));

        Dense output = AutoOutput(model);
        Assert.True(output.SizeFromTargets);
        Assert.Equal(3, output.NeuronCount);
        Assert.IsType<Softmax>(output.Activation);
    }

    [Fact]
    public void A_regression_gets_a_Linear_output_from_MSE()
    {
        var model = Quiet(new NeuralNetwork([new Dense(6, new Tanh())], new SGD(0.05), new MSE(), seed: 1));
        var random = new Random(2);
        double[][] x = Enumerable.Range(0, 20).Select(_ => new[] { random.NextDouble(), random.NextDouble() }).ToArray();
        double[][] y = x.Select(v => new[] { v[0] + v[1], v[0] - v[1] }).ToArray();

        Silent(() => model.Train.TrainLoop(x, y, 2, 4));

        Assert.Equal(2, AutoOutput(model).NeuronCount);
        Assert.IsType<Linear>(AutoOutput(model).Activation);
    }

    [Fact]
    public void The_automatic_output_is_the_same_model_as_the_equivalent_explicit_one()
    {
        var (x, y) = Classes(3);

        var automatic = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 7));
        var explicitOutput = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 7,
                                                     outputLayer: new Dense(3, new Softmax())));

        Silent(() => automatic.Train.TrainLoop(x, y, 3, 4));
        Silent(() => explicitOutput.Train.TrainLoop(x, y, 3, 4));

        Assert.Equal(Parameters(explicitOutput), Parameters(automatic));   // bit for bit
    }

    [Fact]
    public void Manual_Fit_gives_the_output_its_size_too()
    {
        var model = new NeuralNetwork(new SGD(0.1), new MSE(), seed: 1);

        model.Train.Fit(new Tensor(new double[] { 1, 2 }), new Tensor(new double[] { 0.5, 0.1, 0.2, 0.3 }));
        model.Train.UpdateParams();

        Assert.Equal(4, AutoOutput(model).NeuronCount);
    }

    [Fact]
    public void Sequences_size_the_output_per_step()
    {
        var model = Quiet(new NeuralNetwork([new RNN(3, new Tanh())], new Adam(0.01), new MSE(), seed: 1));
        var x = new double[8, 4, 2];
        var y = new double[8, 4];

        Silent(() => model.Train.TrainLoop(x, y, 1, 2, isSequence: true));

        Assert.Equal(1, AutoOutput(model).NeuronCount);   // targets [n, steps] → one value per step
    }

    [Fact]
    public void Parallel_training_works_with_the_automatic_output()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1));
        model.Train.Workers = 3;
        var (x, y) = Classes(4, 40);

        Silent(() => model.Train.TrainLoop(x, y, 2, 8));

        Assert.Equal(4, AutoOutput(model).NeuronCount);
    }

    // ------------------------------------------------------------ the constructors

    [Fact]
    public void The_list_holds_only_hidden_layers_with_or_without_an_output_layer()
    {
        var hidden = new Dense(5, new ReLU());
        var withAuto = new NeuralNetwork([hidden]);
        Assert.Same(hidden, Assert.Single(withAuto.Layers.Hidden));
        Assert.True(AutoOutput(withAuto).SizeFromTargets);

        var otherHidden = new Dense(5, new ReLU());
        var output = new Dense(2, new Softmax());
        var withOutput = new NeuralNetwork([otherHidden], outputLayer: output);
        Assert.Same(otherHidden, Assert.Single(withOutput.Layers.Hidden));
        Assert.Same(output, withOutput.Layers.Output);
    }

    [Fact]
    public void The_constructor_with_an_output_layer_still_works_as_before()
    {
        var output = new Dense(2, new Linear());
        var model = new NeuralNetwork(output, new SGD(0.1), new MSE(), seed: 1);

        Assert.Same(output, model.Layers.Output);
        Assert.False(output.SizeFromTargets);
        Assert.Empty(model.Layers.Hidden);
    }

    // ------------------------------------------------------------ before the targets are known

    [Fact]
    public void Prediction_before_training_says_the_output_waits_for_targets()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())], seed: 1);

        var ex = Assert.Throws<InvalidOperationException>(() => model.GetResults(new Tensor(new double[] { 1, 2 })));
        Assert.Contains("targets", ex.Message);
    }

    [Fact]
    public void SetInputShape_builds_the_hidden_layers_and_leaves_the_output_for_the_targets()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())], seed: 1);

        model.Layers.SetInputShape(new[] { 3 });

        Assert.True(model.Layers.Hidden[0].IsBuilt);
        Assert.False(model.Layers.Output.IsBuilt);
        Assert.Equal(new[] { 3 }, model.Context.InputShape);
    }

    [Fact]
    public void Info_before_training_says_where_the_output_comes_from()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())], new SGD(0.1), new CrossEntropy());
        var writer = new StringWriter();
        var original = Console.Out;
        Console.SetOut(writer);
        try { model.Info(); }
        finally { Console.SetOut(original); }

        Assert.Contains("from the targets", writer.ToString());
        Assert.Contains("from the loss", writer.ToString());
    }

    // ------------------------------------------------------------ the activation for other losses

    private sealed class HalfMse : Loss
    {
        public override string Name => "HalfMse";
        public override double LossFunction(double output, double target) => 0.5 * (output - target) * (output - target);
        public override double DerivativeOfLossFunction(double output, double target) => output - target;
    }

    [Fact]
    public void A_custom_loss_needs_the_activation_to_be_given()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())], new SGD(0.1), new HalfMse(), seed: 1);

        // no silent guess — the library does not substitute a function behind the user's back
        var ex = Assert.Throws<InvalidOperationException>(() =>
            model.Train.Fit(new Tensor(new double[] { 1, 2 }), new Tensor(new double[] { 0.5 })));
        Assert.Contains("HalfMse", ex.Message);

        model.Layers.ReplaceOutput(Dense.FromTargets(new Sigmoid()));
        model.Train.Fit(new Tensor(new double[] { 1, 2 }), new Tensor(new double[] { 0.5 }));
        Assert.IsType<Sigmoid>(model.Layers.Output.Activation);
    }

    [Fact]
    public void Setting_the_activation_on_the_automatic_output_layer_keeps_its_size_automatic()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())], new SGD(0.1), new HalfMse(), seed: 1);

        model.Layers.Output.Activation = new Sigmoid();
        model.Train.Fit(new Tensor(new double[] { 1, 2 }), new Tensor(new double[] { 0.5, 0.5 }));

        Assert.Equal(2, AutoOutput(model).NeuronCount);
        Assert.False(AutoOutput(model).ActivationFromLoss);
    }

    // ------------------------------------------------------------ a different number of outputs

    [Fact]
    public void Targets_of_another_size_are_an_error_and_RebuildIncompatible_rebuilds_only_the_output()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1));
        var (x, y3) = Classes(3);
        Silent(() => model.Train.TrainLoop(x, y3, 2, 4));
        double[] hiddenBefore = ((Dense)model.Layers.Hidden[0]).Neurons.SelectMany(n => n.Weights).ToArray();

        var (_, y4) = Classes(4);
        var ex = Assert.Throws<ArgumentException>(() => Silent(() => model.Train.TrainLoop(x, y4, 1, 4)));
        Assert.Contains("RebuildIncompatible", ex.Message);

        model.Layers.RebuildIncompatible();
        Silent(() => model.Train.TrainLoop(x, y4, 1, 4));

        Assert.Equal(4, AutoOutput(model).NeuronCount);
        Assert.Equal(hiddenBefore.Length, ((Dense)model.Layers.Hidden[0]).Neurons.SelectMany(n => n.Weights).Count());
    }

    [Fact]
    public void RebuildIncompatible_keeps_the_hidden_weights_when_only_the_targets_change()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1));
        var (x, y3) = Classes(3);
        Silent(() => model.Train.TrainLoop(x, y3, 2, 4));
        double[] hiddenBefore = ((Dense)model.Layers.Hidden[0]).Neurons.SelectMany(n => n.Weights).ToArray();

        model.Layers.RebuildIncompatible();
        var (_, y4) = Classes(4);
        model.Train.Fit(new Tensor(x[0]), new Tensor(y4[0]));   // no update — just the rebuild

        Assert.Equal(hiddenBefore, ((Dense)model.Layers.Hidden[0]).Neurons.SelectMany(n => n.Weights).ToArray());
        Assert.Equal(4, AutoOutput(model).NeuronCount);
    }

    [Fact]
    public void Rebuild_lets_the_next_targets_decide_the_output_again()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1));
        var (x, y3) = Classes(3);
        Silent(() => model.Train.TrainLoop(x, y3, 1, 4));

        model.Layers.Rebuild();
        Assert.Equal(0, AutoOutput(model).NeuronCount);

        var (_, y5) = Classes(5);
        Silent(() => model.Train.TrainLoop(x, y5, 1, 4));
        Assert.Equal(5, AutoOutput(model).NeuronCount);
    }

    // ------------------------------------------------------------ going back to the automatic output

    [Fact]
    public void ReplaceOutput_with_FromTargets_goes_back_to_the_automatic_output()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1,
                                            outputLayer: new Dense(3, new Softmax())));
        var (x, y3) = Classes(3);
        Silent(() => model.Train.TrainLoop(x, y3, 1, 4));

        model.Layers.ReplaceOutput(Dense.FromTargets());
        var (_, y6) = Classes(6);
        Silent(() => model.Train.TrainLoop(x, y6, 1, 4));

        Assert.Equal(6, AutoOutput(model).NeuronCount);
        Assert.True(model.Layers.Hidden[0].IsBuilt);
    }

    [Fact]
    public void FromTargets_can_only_be_the_output_layer()
    {
        var model = new NeuralNetwork([new Dense(4, new ReLU())]);

        Assert.Throws<ArgumentException>(() => model.Layers.Add(Dense.FromTargets()));
        Assert.Throws<ArgumentException>(() => model.Layers.Hidden[0] = Dense.FromTargets());
        Assert.Throws<ArgumentException>(() => new NeuralNetwork([Dense.FromTargets()]));
        Assert.Throws<ArgumentException>(() => model.Layers.ReplaceAll([Dense.FromTargets(), new Dense(2)]));
    }

    // ------------------------------------------------------------ saving

    [Fact]
    public void A_loaded_model_keeps_its_automatic_output()
    {
        var model = Quiet(new NeuralNetwork([new Dense(8, new ReLU())], new Adam(0.01), new CrossEntropy(), seed: 1));
        var (x, y3) = Classes(3);
        Silent(() => model.Train.TrainLoop(x, y3, 1, 4));

        var loaded = Quiet(NeuralNetwork.LoadModel(Save(model)));
        Assert.True(AutoOutput(loaded).SizeFromTargets);
        Assert.Equal(3, AutoOutput(loaded).NeuronCount);

        // it still knows it may take another size from the targets
        loaded.Layers.RebuildIncompatible();
        var (_, y4) = Classes(4);
        Silent(() => loaded.Train.TrainLoop(x, y4, 1, 4));
        Assert.Equal(4, AutoOutput(loaded).NeuronCount);
    }

    [Fact]
    public void An_ordinary_output_layer_is_saved_without_the_new_keys()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), seed: 1);
        model.GetResults(new Tensor(new double[] { 1, 2 }));

        string json = model.SaveAsJsonString();

        Assert.DoesNotContain("SizeFromTargets", json);
        Assert.DoesNotContain("ActivationFromLoss", json);
    }

    private static string Save(NeuralNetwork model)
    {
        string path = Path.Combine(Path.GetTempPath(), "mdnn_auto_" + Guid.NewGuid().ToString("N") + ".json");
        model.SaveAsJson(path);
        return path;
    }
}

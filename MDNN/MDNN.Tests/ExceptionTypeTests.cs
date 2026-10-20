using System;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// Error states throw a specific exception type, not a plain <see cref="Exception"/>. Callers can
/// catch only what they can handle (bad argument vs. bad state vs. unsupported), and code that
/// catches <c>Exception</c> keeps working unchanged.
/// </summary>
public class ExceptionTypeTests
{
    private static NeuralNetwork Model()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        return model;
    }

    [Fact]
    public void Layer_position_out_of_range()
    {
        var model = Model();
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Insert(5, new Dense(2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.RemoveAt(3));
    }

    [Fact]
    public void Inputs_and_targets_with_different_sample_counts()
    {
        // Note: LabeledData already rejects this case (its own ArgumentException), so the check
        // in Train.CheckTensorShapes is never reached. The test guards the resulting exception
        // type, not that particular line; it would have passed before the conversion too.
        var model = Model();
        Assert.Throws<ArgumentException>(() =>
            model.Train.TrainLoop(new double[,] { { 1 }, { 2 }, { 3 } }, new double[,] { { 1 }, { 2 } }, 1, 1));
    }

    [Fact]
    public void Invalid_split_ratio_is_a_configuration_error()
    {
        var model = Model();
        model.Train.TrainSplitRatio = 1.5;
        var x = new double[20, 1];
        var y = new double[20, 1];

        Assert.Throws<InvalidOperationException>(() => model.Train.TrainLoop(x, y, 1, 1));
    }

    private sealed class Identity : ParameterlessLayer
    {
        private int[] shape = { 0 };
        public override string Name => "Identity";
        public override int[] InputShape => shape;
        public override int[] OutputShape => shape;
        public override Tensor? Output => null;
        public override Tensor? RawOutput => null;
        public override Tensor FeedForward(Tensor input_values) => input_values;
        public override void WireShapes(int? n = null, int[]? input = null) { if (input != null) shape = input; }
        public override void MaterializeParameters() { }
        public override bool IsMaterialized => true;
        public override Tensor Backward(Tensor g, bool p = false, bool c = true) => g;
    }

    [Fact]
    public void Saving_a_custom_layer_is_not_supported()
    {
        var model = Model();
        model.Layers.Insert(0, new Identity());
        model.GetResults(new Tensor(new double[] { 1.0 }));

        var ex = Assert.Throws<NotSupportedException>(() => model.SaveAsJsonString());
        Assert.Contains("Identity", ex.Message);
    }

    // Conv and MaxPool parameters used to be unchecked: an unknown padding silently behaved like
    // "valid", and a zero or negative size failed only deep inside an allocation or a division by zero.
    [Theory]
    [InlineData("smae")]
    [InlineData("full")]
    [InlineData("")]
    public void Conv_rejects_unknown_padding(string padding)
    {
        Assert.Throws<ArgumentException>(() => new Conv(2, 3, new ReLU(), padding));
    }

    [Fact]
    public void Conv_padding_ignores_case()
    {
        Assert.Equal("same", new Conv(2, 3, new ReLU(), "SAME").Padding);
    }

    [Fact]
    public void Conv_and_MaxPool_reject_non_positive_sizes()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv(2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MaxPool(0));
    }

    [Fact]
    public void Conv_kernel_larger_than_input_is_rejected_with_valid_padding()
    {
        var model = Model();
        model.Layers.Add(new Conv(2, 5, new ReLU(), "valid"));

        var ex = Assert.Throws<ArgumentException>(() => model.GetResults(new Tensor(new double[3, 3, 1])));
        Assert.Contains("does not fit", ex.Message);
    }
}

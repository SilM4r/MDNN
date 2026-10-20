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
/// What a layer returns to the outside is a SNAPSHOT, not a window into its working buffer.
///
/// Previously both `Dense` and `RNN` returned `new Tensor(output)`, where `output` is an array the
/// layer overwrites on every pass. `Tensor` copies its data only on the first access to `.Data`,
/// so a result kept from the first forward pass CHANGED RETROACTIVELY after the second one:
///
///     var a = model.GetResults(x1);
///     var b = model.GetResults(x2);
///     a.Data  →  values for x2
///
/// Inside the library this did no harm (`.Data` is always read immediately), but it is public API.
/// Conv and MaxPool did not have the problem in `FeedForward` (the jagged-to-tensor conversion
/// copied), but they did in `Output`, and also reported a wrong `Shape` because `double[][][]` has Rank 1.
/// </summary>
public class LayerOutputTests
{
    private static double[,,] Image(int h, int w, int c, double seed)
    {
        var img = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int k = 0; k < c; k++)
            img[i, j, k] = ((i * w + j) * c + k) * 0.01 + seed;
        return img;
    }

    // ------------------------------------------------------------ GetResults must not overwrite the previous result

    [Fact]
    public void Dense_result_survives_the_next_forward()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1);

        Tensor first = model.GetResults(new Tensor(new double[] { 1, 0, 0 }));
        Tensor second = model.GetResults(new Tensor(new double[] { 0, 0, 1 }));

        Assert.NotEqual(first.Data, second.Data);
    }

    [Fact]
    public void RNN_result_survives_the_next_forward()
    {
        var model = new NeuralNetwork(new RNN(3, new Tanh()), new SGD(0.01), new MSE(), seed: 1);

        Tensor first = model.GetResults(new Tensor(new double[] { 1, 0 }));
        Tensor second = model.GetResults(new Tensor(new double[] { 0, 1 }));

        Assert.NotEqual(first.Data, second.Data);
    }

    // ------------------------------------------------------------ Output is a snapshot too

    [Theory]
    [InlineData(false)]   // Output
    [InlineData(true)]    // RawOutput
    public void Dense_layer_output_property_is_a_snapshot(bool raw)
    {
        var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(new double[] { 1, 0, 0 }));
        Tensor snapshot = (raw ? layer.RawOutput : layer.Output)!;

        model.GetResults(new Tensor(new double[] { 0, 0, 1 }));
        Tensor now = (raw ? layer.RawOutput : layer.Output)!;

        Assert.NotEqual(snapshot.Data, now.Data);
    }

    [Fact]
    public void RNN_layer_output_property_is_a_snapshot()
    {
        var model = new NeuralNetwork(new RNN(3, new Tanh()), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(new double[] { 1, 0 }));
        Tensor snapshot = layer.Output!;

        model.GetResults(new Tensor(new double[] { 0, 1 }));

        Assert.NotEqual(snapshot.Data, layer.Output!.Data);
    }

    [Fact]
    public void Conv_layer_output_property_is_a_snapshot()
    {
        var model = new NeuralNetwork(new Conv(2, 2, new Tanh(), "valid"), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor snapshot = layer.Output!;

        model.GetResults(new Tensor(Image(4, 4, 1, 5.0)));

        Assert.NotEqual(snapshot.Data, layer.Output!.Data);
    }

    // ------------------------------------------------------------ Output reports the correct shape

    [Fact]
    public void Conv_layer_output_reports_the_real_shape()
    {
        // `double[][][]` has Rank 1, so `new Tensor(array)` inferred a Shape of just [outH].
        // The data was right, the Shape was not; it shows when Conv is the output layer.
        var model = new NeuralNetwork(new Conv(2, 2, new Linear(), "valid"), new SGD(0.01), new MSE(), seed: 1);

        Tensor fromForward = model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor fromProperty = model.Layers.Layers[0].Output!;

        Assert.Equal(new[] { 3, 3, 2 }, fromForward.Shape);
        Assert.Equal(fromForward.Shape, fromProperty.Shape);
        Assert.Equal(fromForward.Data, fromProperty.Data);
    }

    [Fact]
    public void MaxPool_layer_output_reports_the_real_shape()
    {
        var model = new NeuralNetwork(new MaxPool(2), new SGD(0.01), new MSE(), seed: 1);

        Tensor fromForward = model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor fromProperty = model.Layers.Layers[0].Output!;

        Assert.Equal(new[] { 2, 2, 1 }, fromForward.Shape);
        Assert.Equal(fromForward.Shape, fromProperty.Shape);
    }

    [Fact]
    public void Dense_result_reports_the_real_shape()
    {
        var model = new NeuralNetwork(new Dense(4, new Linear()), new SGD(0.01), new MSE(), seed: 1);

        Tensor output = model.GetResults(new Tensor(new double[] { 1, 0, 0 }));

        Assert.Equal(new[] { 4 }, output.Shape);
    }

    // ------------------------------------------------------------ values must not change

    [Fact]
    public void Snapshotting_does_not_change_what_the_layers_compute()
    {
        // Guard that the copying did not change the numbers as well.
        var model = new NeuralNetwork(new Dense(3, new Softmax()), new SGD(0.01), new MSE(), seed: 42);
        model.Layers.Add(new Conv(2, 3, new Tanh(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.Layers.Add(new Dense(6, new Tanh()));

        Tensor x = new Tensor(Image(8, 8, 1, 0.0));

        double[] a = model.GetResults(x).Data;
        double[] b = model.GetResults(x).Data;

        Assert.Equal(a, b);
        Assert.Equal(1.0, a.Sum(), 12);   // softmax output is still a distribution
    }
}

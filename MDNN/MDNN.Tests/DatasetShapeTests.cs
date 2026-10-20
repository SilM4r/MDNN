using System;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// How many dimensions a dataset for <c>TrainLoop</c> may have.
///
/// A sample has at most 3 dimensions ([h, w, c] for Conv), so outside sequence mode a dataset
/// is at most 4D, and at most 5D in sequence mode. Previously the check already rejected 4D:
/// images with channels [n, h, w, c] (exactly what Conv takes) could not be trained through
/// TrainLoop, and the only workaround was to flatten them to [n, h·w·c].
/// </summary>
public class DatasetShapeTests
{
    private static NeuralNetwork Cnn()
    {
        var model = new NeuralNetwork(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 3);
        model.Layers.Add(new Conv(2, 3, new ReLU(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        return model;   // TestAfterTraining left on: valid and test sets get exercised too
    }

    private static double Pixel(int s, int i, int j) => Math.Sin(s + i * 0.7 + j * 0.3);

    private static double[,] Labels(int n)
    {
        var y = new double[n, 2];
        for (int s = 0; s < n; s++) y[s, s % 2] = 1;
        return y;
    }

    [Fact]
    public void Four_dimensional_image_dataset_trains()
    {
        var x = new double[12, 6, 6, 3];
        for (int s = 0; s < 12; s++)
        for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
        for (int c = 0; c < 3; c++)
            x[s, i, j, c] = Pixel(s, i, j) + c;

        var model = Cnn();
        model.Train.TrainLoop(x, Labels(12), 2, 4);

        Assert.Equal(new[] { 6, 6, 3 }, model.Layers.Layers[0].InputShape);
    }

    [Fact]
    public void Single_channel_4d_dataset_trains_exactly_like_the_3d_one()
    {
        // [n, h, w, 1] and [n, h, w] are the same data; Conv expands a 2D sample to [h, w, 1].
        var flat = new double[12, 6, 6];
        var withChannel = new double[12, 6, 6, 1];
        for (int s = 0; s < 12; s++)
        for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            flat[s, i, j] = withChannel[s, i, j, 0] = Pixel(s, i, j);

        var a = Cnn();
        var b = Cnn();
        a.Train.TrainLoop(flat, Labels(12), 2, 4);
        b.Train.TrainLoop(withChannel, Labels(12), 2, 4);

        var ka = ((Conv)a.Layers.Layers[0]).Kernels;
        var kb = ((Conv)b.Layers.Layers[0]).Kernels;
        Assert.Equal(ka.Length, kb.Length);
        for (int f = 0; f < ka.Length; f++)
        for (int i = 0; i < ka[f].Length; i++)
        for (int j = 0; j < ka[f][i].Length; j++)
            Assert.Equal(ka[f][i][j], kb[f][i][j]);
    }

    [Fact]
    public void Five_dimensional_non_sequential_dataset_is_rejected()
    {
        var model = Cnn();
        Assert.Throws<ArgumentException>(() =>
            model.Train.TrainLoop(new double[4, 6, 6, 1, 1], Labels(4), 1, 2));
    }

    [Fact]
    public void Six_dimensional_sequential_dataset_is_rejected()
    {
        var model = Cnn();
        Assert.Throws<ArgumentException>(() =>
            model.Train.TrainLoop(new double[2, 3, 6, 6, 1, 1], new double[2, 3, 2], 1, 2, isSequence: true));
    }
}

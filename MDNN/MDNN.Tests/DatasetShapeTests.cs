using System;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Kolik rozměrů smí mít dataset pro <c>TrainLoop</c>.
///
/// Vzorek má nejvýš 3 rozměry ([h, w, c] pro Conv), takže mimo sekvenční režim je dataset
/// nejvýš 4D a v sekvenčním nejvýš 5D. Dřív kontrola odmítala už 4D: obrázky s kanály
/// [n, h, w, c] — přesně to, co Conv bere — přes TrainLoop natrénovat nešly a jediné
/// obejití bylo zploštit je na [n, h·w·c].
/// </summary>
public class DatasetShapeTests
{
    private static My_DNN.MDNN Cnn()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 3);
        model.Layers.Add(new Conv(2, 3, new ReLu(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.Train.ShowLossChartInTrainLoop = false;
        model.Train.ShowModelInfoIntrainLoop = false;
        model.Train.AutoSaveInTrainLoop = false;
        return model;   // TestNeuralNetworkAfterTraining nechán zapnutý: projde i valid a test
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

        Assert.Equal(new[] { 6, 6, 3 }, model.Layers.Layers[0].Input_size_and_shape);
    }

    [Fact]
    public void Single_channel_4d_dataset_trains_exactly_like_the_3d_one()
    {
        // [n, h, w, 1] a [n, h, w] jsou tatáž data — Conv si 2D vzorek doplní na [h, w, 1].
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

        var ka = ((Conv)a.Layers.Layers[0]).Kernel;
        var kb = ((Conv)b.Layers.Layers[0]).Kernel;
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

using System;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// `new Tensor(pole)` musí dát správný tvar pro každý vstup, který knihovna přijímá.
///
/// Dřív konstruktor bral tvar z `Rank`, a ten je u zubatého pole vždycky 1: z
/// `double[4][4][1]` vznikl Shape [4] nad 16 prvky. Tvar a data se rozešly a vrstvy,
/// které čtou plochá data podle tvaru, počítaly nad nesmyslem — Conv bez chyby,
/// Dense s IndexOutOfRange. Správně to uměl jen `ConvertArrayToTensor`.
/// </summary>
public class TensorConstructionTests
{
    private static double[][][] JaggedImage(int h, int w, int c)
    {
        var img = new double[h][][];
        for (int i = 0; i < h; i++)
        {
            img[i] = new double[w][];
            for (int j = 0; j < w; j++)
            {
                img[i][j] = new double[c];
                for (int k = 0; k < c; k++) img[i][j][k] = (i * w + j) * c + k;
            }
        }
        return img;
    }

    private static double[,,] MultiImage(int h, int w, int c)
    {
        var img = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int k = 0; k < c; k++)
            img[i, j, k] = (i * w + j) * c + k;
        return img;
    }

    [Fact]
    public void Jagged_3d_array_gets_its_full_shape()
    {
        var tensor = new Tensor(JaggedImage(4, 3, 2));

        Assert.Equal(new[] { 4, 3, 2 }, tensor.Shape);
        Assert.Equal(new Tensor(MultiImage(4, 3, 2)).Data, tensor.Data);   // row-major jako multidim
    }

    [Fact]
    public void Jagged_2d_array_gets_its_full_shape()
    {
        var tensor = new Tensor(new double[][] { new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 } });

        Assert.Equal(new[] { 2, 3 }, tensor.Shape);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, tensor.Data);
    }

    [Fact]
    public void Ragged_array_is_rejected()
    {
        var ragged = new double[][] { new[] { 1.0, 2.0 }, new[] { 3.0 } };

        Assert.Throws<ArgumentException>(() => new Tensor(ragged));
    }

    [Fact]
    public void Int_jagged_array_is_converted_to_double()
    {
        var tensor = new Tensor(new int[][] { new[] { 1, 2 }, new[] { 3, 4 } });

        Assert.Equal(new[] { 2, 2 }, tensor.Shape);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, tensor.Data);
        Assert.IsType<double[,]>(tensor.GetOriginalData());
    }

    [Fact]
    public void Int_multidim_array_is_converted_to_double()
    {
        var tensor = new Tensor(new int[,] { { 1, 2 }, { 3, 4 } });

        Assert.Equal(new[] { 2, 2 }, tensor.Shape);
        Assert.IsType<double[,]>(tensor.GetOriginalData());
    }

    [Fact]
    public void Unsupported_element_type_is_rejected_with_a_clear_exception()
    {
        // Dřív FormatException z Convert.ToDouble("a") až při prvním čtení .Data.
        Assert.Throws<ArgumentException>(() => new Tensor(new string[] { "a" }));
    }

    [Fact]
    public void Conv_gives_the_same_result_for_jagged_and_multidim_input()
    {
        var model = new My_DNN.MDNN(new Conv(2, 2, new Linear(), "valid"), new SGD(0.01), new MSE(), seed: 1);

        Tensor fromMulti = model.GetResults(new Tensor(MultiImage(4, 4, 1)));
        Tensor fromJagged = model.GetResults(new Tensor(JaggedImage(4, 4, 1)));

        Assert.Equal(new[] { 3, 3, 2 }, fromJagged.Shape);
        Assert.Equal(fromMulti.Data, fromJagged.Data);
    }
}

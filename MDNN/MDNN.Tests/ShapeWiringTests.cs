using System;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Zapojování tvarů: první průchod projde řetěz zleva doprava, každá vrstva dostane
/// výstupní tvar té předchozí — a od té chvíle je tvar DANÝ.
///
/// Tyhle testy hlídají obě půlky té věty:
///   1. zapojení musí zvládnout všechny tvary vstupu, které zvládne forward
///      (dřív ne: `WireShapes` uměl rank 1 a 3, `FeedForward` navíc rank 2)
///   2. po zapojení musí jiný tvar SPADNOUT, ne se tiše dopočítat z toho, co zbylo
///      v bufferu po minulém vzorku
///
/// Před opravou byl projev té druhé půlky tenhle: vrstva zapojená na 4×4 dostala
/// vstup 3×3 samých nul a vrátila čísla z předchozího vzorku. Bez výjimky.
/// </summary>
public class ShapeWiringTests
{
    private static My_DNN.MDNN ModelWith(params Layer[] hidden)
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1);
        foreach (Layer l in hidden) model.Layers.Add(l);
        return model;
    }

    private static double[,,] Filled(int h, int w, int c, double value)
    {
        var img = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int k = 0; k < c; k++)
            img[i, j, k] = value;
        return img;
    }

    // ------------------------------------------------------------ 1) zapojení celého řetězu

    [Fact]
    public void First_forward_wires_the_whole_chain_left_to_right()
    {
        var model = ModelWith(new Conv(2, 3, new Tanh(), "valid"), new MaxPool(2), new Dense(6, new Tanh()));

        Assert.All(model.Layers.Layers, l => Assert.True(l.Input_size_and_shape[0] <= 0));

        model.GetResults(new Tensor(Filled(8, 8, 1, 0.5)));

        var layers = model.Layers.Layers;
        Assert.Equal(new[] { 8, 8, 1 }, layers[0].Input_size_and_shape);
        Assert.Equal(new[] { 6, 6, 2 }, layers[0].Output_size_and_shape);
        Assert.Equal(new[] { 6, 6, 2 }, layers[1].Input_size_and_shape);
        Assert.Equal(new[] { 3, 3, 2 }, layers[1].Output_size_and_shape);
        Assert.Equal(new[] { 18 }, layers[2].Input_size_and_shape);   // 3*3*2 zploštěno pro Dense
        Assert.Equal(new[] { 6 }, layers[3].Input_size_and_shape);
        Assert.All(layers, l => Assert.True(l.IsMaterialized));
    }

    // ------------------------------------------------------------ 2) zapojení umí totéž co forward

    [Theory]
    [InlineData(3)]   // [4,4,1] — kanálová dimenze zadaná
    [InlineData(2)]   // [4,4]   — 2D obrázek bez kanálů: TOHLE dřív padalo
    [InlineData(1)]   // [16]    — plochý vektor
    public void Conv_as_first_layer_accepts_every_rank_the_forward_accepts(int rank)
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        Tensor input = rank switch
        {
            3 => new Tensor(new double[4, 4, 1]),
            2 => new Tensor(new double[4, 4]),
            _ => new Tensor(new double[16]),
        };

        Tensor output = model.GetResults(input);

        // valid + kernel 2 nad 4×4 → 3×3×1 → Dense(2)
        Assert.Equal(2, output.Data.Length);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].Input_size_and_shape);
    }

    [Fact]
    public void MaxPool_as_first_layer_accepts_a_2D_input_too()
    {
        var model = ModelWith(new MaxPool(2));

        Tensor output = model.GetResults(new Tensor(new double[4, 4]));

        Assert.Equal(2, output.Data.Length);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].Input_size_and_shape);
    }

    [Fact]
    public void Unsupported_rank_is_rejected_with_a_message_that_says_what_is_wrong()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        var ex = Assert.ThrowsAny<Exception>(() => model.GetResults(new Tensor(new double[2, 4, 4, 1])));

        // Podstata testu: hláška nesmí být prázdná. Dřív tady byl `throw new Exception()`
        // úplně bez textu a uživatel neměl jak zjistit, co se stalo.
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
        Assert.Contains("CNN", ex.Message);
    }

    // ------------------------------------------------------------ 3) po zapojení je tvar daný

    [Fact]
    public void Conv_rejects_a_smaller_input_instead_of_reusing_the_stale_buffer()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        // první vzorek naplní vstupní buffer stovkami
        model.GetResults(new Tensor(Filled(4, 4, 1, 100.0)));

        // druhý vzorek je menší; před opravou se přepsal jen jeho roh a zbytek
        // se dopočítal ze stovek, které tam zbyly — tiše, bez výjimky
        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        Assert.Contains("4×4×1", ex.Message);
        Assert.Contains("3×3×1", ex.Message);
    }

    [Fact]
    public void MaxPool_rejects_a_smaller_input_instead_of_reusing_the_stale_buffer()
    {
        var model = ModelWith(new MaxPool(2));

        model.GetResults(new Tensor(Filled(4, 4, 1, 100.0)));

        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        Assert.Contains("4×4×1", ex.Message);
    }

    [Fact]
    public void Conv_rejects_an_input_with_a_different_channel_count()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        model.GetResults(new Tensor(Filled(4, 4, 2, 1.0)));

        Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(4, 4, 1, 1.0))));
    }

    [Fact]
    public void Conv_rejects_a_bigger_input_with_a_clear_message_not_IndexOutOfRange()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        model.GetResults(new Tensor(Filled(4, 4, 1, 1.0)));

        // Dřív: IndexOutOfRangeException z hloubi kopírovací smyčky, bez zmínky o tvarech.
        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(5, 5, 1, 1.0))));

        Assert.Contains("4×4×1", ex.Message);
        Assert.Contains("5×5×1", ex.Message);
    }

    [Fact]
    public void The_right_shape_keeps_working_after_a_rejected_one()
    {
        // Odmítnutý vstup nesmí vrstvu rozhodit — tvary se nemění, model jede dál.
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        double[] before = model.GetResults(new Tensor(Filled(4, 4, 1, 0.25))).Data;

        Assert.ThrowsAny<Exception>(() => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        double[] after = model.GetResults(new Tensor(Filled(4, 4, 1, 0.25))).Data;

        Assert.Equal(before, after);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].Input_size_and_shape);
    }

    // ------------------------------------------------------------ 4) forward nesahá na vstup volajícího

    [Fact]
    public void Conv_does_not_reshape_the_tensor_it_was_given()
    {
        // Vrstva si tvar normalizuje pro SEBE — tenzor patří volajícímu a nesmí se pod ním
        // změnit. Dřív `FeedForward` volal TensorValues.Reshape(...), takže po průchodu
        // modelem měl uživatelův tenzor najednou jiný Shape, než jaký mu zadal.
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        var input = new Tensor(new double[16]);
        int[] shapeBefore = (int[])input.Shape.Clone();

        model.GetResults(input);

        Assert.Equal(shapeBefore, input.Shape);
    }

    [Fact]
    public void MaxPool_does_not_reshape_the_tensor_it_was_given()
    {
        var model = ModelWith(new MaxPool(2));

        var input = new Tensor(new double[4, 4]);
        int[] shapeBefore = (int[])input.Shape.Clone();

        model.GetResults(input);

        Assert.Equal(shapeBefore, input.Shape);
    }

    [Fact]
    public void The_same_tensor_can_be_fed_twice_and_gives_the_same_answer()
    {
        // Nejpraktičtější projev: kdo si vstup schová a pošle ho podruhé, musí dostat totéž.
        //
        // POZOR, tohle je GUARD, ne důkaz — prochází i před opravou. Přepsaný tvar
        // ([9] → [3,3,1]) se totiž podruhé znormalizuje sám na sebe, takže se ta mutace
        // navenek neprojeví. Důkaz nese dvojice testů výš, které tvar kontrolují přímo.
        var model = ModelWith(new Conv(1, 2, new Tanh(), "valid"));

        var input = new Tensor(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });

        double[] first = model.GetResults(input).Data;
        double[] second = model.GetResults(input).Data;

        Assert.Equal(first, second);
    }

    // ------------------------------------------------------------ 5) trénink tím projde

    [Fact]
    public void Training_a_CNN_over_2D_samples_runs()
    {
        // Dataset [vzorky, výška, šířka] → per-vzorek [výška, šířka], tedy rank 2.
        // Přesně ten případ, který dřív spadl na prázdné výjimce při zapojování.
        var model = ModelWith(new Conv(1, 3, new Tanh(), "valid"));
        model.Train.ShowLossChartInTrainLoop = false;
        model.Train.AutoSaveInTrainLoop = false;
        model.Train.ShowModelInfoIntrainLoop = false;
        model.Train.TestNeuralNetworkAfterTraining = false;

        var x = new double[20, 8, 8];
        var y = new double[20, 2];
        for (int n = 0; n < 20; n++)
        {
            for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                x[n, i, j] = (n + i * 8 + j) * 0.001;
            y[n, n % 2] = 1;
        }

        var original = Console.Out;
        Console.SetOut(TextWriter.Null);
        try { model.Train.TrainLoop(x, y, 2, 4); }
        finally { Console.SetOut(original); }

        Assert.Equal(2u, model.Train.CurrentEpoch);
        Assert.Equal(new[] { 8, 8, 1 }, model.Layers.Layers[0].Input_size_and_shape);
    }
}

using System;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// Co vrstva vrací navenek, je SNÍMEK — ne okno do jejího pracovního bufferu.
///
/// Dřív `Dense` i `RNN` vracely `new Tensor(output)`, kde `output` je pole, které si vrstva
/// přepisuje při každém průchodu. `Tensor` si data kopíruje až při prvním sáhnutí na `.Data`,
/// takže výsledek schovaný z prvního forwardu se po druhém forwardu ZPĚTNĚ ZMĚNIL:
///
///     var a = model.GetResults(x1);
///     var b = model.GetResults(x2);
///     a.Data  →  hodnoty pro x2
///
/// Uvnitř knihovny to nekousalo (všude se `.Data` čte hned), ale je to veřejné API.
/// Conv a MaxPool problém neměly ve `FeedForward` (Tensor.FromJagged3D kopíruje), zato
/// v `Layer_output` ano — a k tomu hlásily špatný `Shape`, protože `double[][][]` má Rank 1.
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

    // ------------------------------------------------------------ GetResults nesmí přepisovat minulý výsledek

    [Fact]
    public void Dense_result_survives_the_next_forward()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1);

        Tensor first = model.GetResults(new Tensor(new double[] { 1, 0, 0 }));
        Tensor second = model.GetResults(new Tensor(new double[] { 0, 0, 1 }));

        Assert.NotEqual(first.Data, second.Data);
    }

    [Fact]
    public void RNN_result_survives_the_next_forward()
    {
        var model = new My_DNN.MDNN(new RNN(3, new Tanh()), new SGD(0.01), new MSE(), seed: 1);

        Tensor first = model.GetResults(new Tensor(new double[] { 1, 0 }));
        Tensor second = model.GetResults(new Tensor(new double[] { 0, 1 }));

        Assert.NotEqual(first.Data, second.Data);
    }

    // ------------------------------------------------------------ Layer_output je taky snímek

    [Theory]
    [InlineData(false)]   // Layer_output
    [InlineData(true)]    // Layer_raw_output
    public void Dense_layer_output_property_is_a_snapshot(bool raw)
    {
        var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(new double[] { 1, 0, 0 }));
        Tensor snapshot = (raw ? layer.Layer_raw_output : layer.Layer_output)!;

        model.GetResults(new Tensor(new double[] { 0, 0, 1 }));
        Tensor now = (raw ? layer.Layer_raw_output : layer.Layer_output)!;

        Assert.NotEqual(snapshot.Data, now.Data);
    }

    [Fact]
    public void RNN_layer_output_property_is_a_snapshot()
    {
        var model = new My_DNN.MDNN(new RNN(3, new Tanh()), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(new double[] { 1, 0 }));
        Tensor snapshot = layer.Layer_output!;

        model.GetResults(new Tensor(new double[] { 0, 1 }));

        Assert.NotEqual(snapshot.Data, layer.Layer_output!.Data);
    }

    [Fact]
    public void Conv_layer_output_property_is_a_snapshot()
    {
        var model = new My_DNN.MDNN(new Conv(2, 2, new Tanh(), "valid"), new SGD(0.01), new MSE(), seed: 1);
        Layer layer = model.Layers.Layers[0];

        model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor snapshot = layer.Layer_output!;

        model.GetResults(new Tensor(Image(4, 4, 1, 5.0)));

        Assert.NotEqual(snapshot.Data, layer.Layer_output!.Data);
    }

    // ------------------------------------------------------------ Layer_output hlásí správný tvar

    [Fact]
    public void Conv_layer_output_reports_the_real_shape()
    {
        // `double[][][]` má Rank 1, takže `new Tensor(pole)` odvodil Shape jen [outH].
        // Data byla správně, Shape ne — projeví se, když je Conv výstupní vrstva.
        var model = new My_DNN.MDNN(new Conv(2, 2, new Linear(), "valid"), new SGD(0.01), new MSE(), seed: 1);

        Tensor fromForward = model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor fromProperty = model.Layers.Layers[0].Layer_output!;

        Assert.Equal(new[] { 3, 3, 2 }, fromForward.Shape);
        Assert.Equal(fromForward.Shape, fromProperty.Shape);
        Assert.Equal(fromForward.Data, fromProperty.Data);
    }

    [Fact]
    public void MaxPool_layer_output_reports_the_real_shape()
    {
        var model = new My_DNN.MDNN(new MaxPool(2), new SGD(0.01), new MSE(), seed: 1);

        Tensor fromForward = model.GetResults(new Tensor(Image(4, 4, 1, 0.0)));
        Tensor fromProperty = model.Layers.Layers[0].Layer_output!;

        Assert.Equal(new[] { 2, 2, 1 }, fromForward.Shape);
        Assert.Equal(fromForward.Shape, fromProperty.Shape);
    }

    [Fact]
    public void Dense_result_reports_the_real_shape()
    {
        var model = new My_DNN.MDNN(new Dense(4, new Linear()), new SGD(0.01), new MSE(), seed: 1);

        Tensor output = model.GetResults(new Tensor(new double[] { 1, 0, 0 }));

        Assert.Equal(new[] { 4 }, output.Shape);
    }

    // ------------------------------------------------------------ hodnoty se nesmí změnit

    [Fact]
    public void Snapshotting_does_not_change_what_the_layers_compute()
    {
        // Pojistka proti tomu, aby se s kopírováním nezměnila i čísla.
        var model = new My_DNN.MDNN(new Dense(3, new Softmax()), new SGD(0.01), new MSE(), seed: 42);
        model.Layers.Add(new Conv(2, 3, new Tanh(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.Layers.Add(new Dense(6, new Tanh()));

        Tensor x = new Tensor(Image(8, 8, 1, 0.0));

        double[] a = model.GetResults(x).Data;
        double[] b = model.GetResults(x).Data;

        Assert.Equal(a, b);
        Assert.Equal(1.0, a.Sum(), 12);   // softmax výstup je pořád rozdělení
    }
}

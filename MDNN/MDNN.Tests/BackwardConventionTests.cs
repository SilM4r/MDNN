using System;
using System.Collections.Generic;
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
/// Konvence B: <c>Backward(gradOutput) → gradInput</c> místo dnešní trojice
/// CalculateLayerGradients + SeedOutputGradient + BackPropagation.
///
/// Tyhle testy hlídají PŘECHOD, ne správnost matematiky — tu už hlídají gradient checky
/// a paritní testy proti PyTorchi. Tady jde o jedinou otázku: dává nová cesta
/// BITOVĚ STEJNÝ výsledek jako stará? Ne „podobný" — stejný. Kdyby se povolila jen
/// blízkost, prošla by i tichá záměna konvence (kdo dodá derivaci aktivace, jak se
/// zaokrouhluje pořadí sčítání), a přesně to je riziko tohohle refaktoru.
/// </summary>
public class BackwardConventionTests
{
    /// <summary>Gradienty všech parametrů modelu v pevném pořadí — otisk stavu po backwardu.</summary>
    private static double[] Fingerprint(My_DNN.MDNN model)
    {
        var values = new List<double>();
        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is LayerBasedOnNeurons neuronLayer)
            {
                foreach (Neuron n in neuronLayer.Neurons)
                {
                    values.AddRange(n.gradientsW);
                    values.Add(n.gradientsB);
                }
            }
            else if (layer is Conv conv)
            {
                foreach (var filter in conv.dKernels)
                foreach (var row in filter)
                foreach (var col in row)
                    values.AddRange(col);

                values.AddRange(conv.dBiases);
            }
            // MaxPool nemá parametry — do otisku nepřispívá, ale jeho správnost se projeví
            // v gradientech vrstev POD ním, které tam jsou.
        }
        return values.ToArray();
    }

    /// <summary>Projde model novou cestou: jeden Backward řetěz shora dolů.</summary>
    private static Tensor RunNewPath(My_DNN.MDNN model, Tensor x, Tensor target, bool fused = false)
    {
        Tensor output = model.GetResults(x);
        var layers = model.Layers.Layers;
        var grad = new Tensor(OutputGradient(model, output, target));

        grad = layers[^1].Backward(grad, preActivation: fused);
        for (int i = layers.Count - 2; i >= 0; i--)
            grad = layers[i].Backward(grad);

        return grad;
    }

    private static void AssertBitIdentical(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]),
                $"{what}, parametr {i}: stará cesta {expected[i]:G17}, nová {actual[i]:G17}");
    }

    /// <summary>∂L/∂output výstupní vrstvy — přesně to, co dnes počítá Gradient
    /// před předáním do SeedOutputGradient.</summary>
    private static double[] OutputGradient(My_DNN.MDNN model, Tensor output, Tensor target)
    {
        double[] d = new double[output.Data.Length];
        for (int i = 0; i < d.Length; i++)
            d[i] = model.Context.Loss.DerivativeOfLossFunction(output.Data[i], target.Data[i]);
        return d;
    }

    private static My_DNN.MDNN Build(int seed, Loss loss, Activation_func outAct, params int[] hidden)
    {
        var model = new My_DNN.MDNN(new Dense(3, outAct), new SGD(0.01), loss, seed: seed);
        foreach (int h in hidden) model.Layers.Add(new Dense(h, new Tanh()));
        return model;
    }

    [Theory]
    [InlineData(4)]              // jen výstupní vrstva
    [InlineData(4, 5)]           // jedna skrytá
    [InlineData(4, 6, 5)]        // dvě skryté
    public void Dense_backward_is_bit_identical_to_old_path(params int[] shape)
    {
        int[] hidden = shape.Skip(1).ToArray();
        var x = new Tensor(Enumerable.Range(0, shape[0]).Select(i => 0.4 - i * 0.17).ToArray());
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        // ---- STARÁ cesta ----
        var oldModel = Build(9, new MSE(), new Tanh(), hidden);
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        // ---- NOVÁ cesta: ruční Backward řetěz shora dolů ----
        var newModel = Build(9, new MSE(), new Tanh(), hidden);
        Tensor output = newModel.GetResults(x);

        var layers = newModel.Layers.Layers;
        var grad = new Tensor(OutputGradient(newModel, output, target));

        for (int i = layers.Count - 1; i >= 0; i--)
            grad = layers[i].Backward(grad);

        double[] actual = Fingerprint(newModel);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]),
                $"parametr {i}: stará cesta {expected[i]:G17}, nová {actual[i]:G17}");
    }

    [Fact]
    public void Dense_backward_is_bit_identical_with_fused_softmax_cross_entropy()
    {
        // Fúze: loss vrací rovnou ∂L/∂z = s − t, takže se derivace aktivace PŘESKAKUJE.
        // V nové konvenci to řeší preActivation, ne druhý vstupní bod.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });
        var target = new Tensor(new double[] { 0.0, 1.0, 0.0 });

        var oldModel = Build(12, new CrossEntropy(), new Softmax(), 6);
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Build(12, new CrossEntropy(), new Softmax(), 6);
        Tensor output = newModel.GetResults(x);

        var layers = newModel.Layers.Layers;
        var grad = new Tensor(OutputGradient(newModel, output, target));

        // výstupní vrstva: gradient je UŽ ∂L/∂z
        grad = layers[^1].Backward(grad, preActivation: true);
        for (int i = layers.Count - 2; i >= 0; i--)
            grad = layers[i].Backward(grad);

        double[] actual = Fingerprint(newModel);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]),
                $"parametr {i}: stará cesta {expected[i]:G17}, nová {actual[i]:G17}");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    public void Dense_backward_is_bit_identical_with_dropout(double rate)
    {
        // Dropout musí projít TOUTÉŽ maskou. Aby obě cesty dostaly stejnou masku,
        // losuje se ze stejného seedu a maska se pak zmrazí.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        My_DNN.MDNN Prepare()
        {
            var m = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 33);
            m.Layers.Add(new Dense(10, new Tanh(), dropout: rate));
            m.Context.IsTraining = true;
            m.GetResults(x);                                     // vylosuje masku
            foreach (Dense d in m.Layers.Layers.Cast<Dense>())
                d.FreezeDropoutMaskForTests = true;              // od teď se nelosuje
            return m;
        }

        var oldModel = Prepare();
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Prepare();
        Tensor output = newModel.GetResults(x);
        var layers = newModel.Layers.Layers;
        var grad = new Tensor(OutputGradient(newModel, output, target));
        for (int i = layers.Count - 1; i >= 0; i--)
            grad = layers[i].Backward(grad);

        double[] actual = Fingerprint(newModel);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]),
                $"rate={rate}, parametr {i}: stará {expected[i]:G17}, nová {actual[i]:G17}");
    }

    // ============================================================ Conv / MaxPool / RNN

    private static double[,,] Image(int h, int w, int c)
    {
        var img = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int k = 0; k < c; k++)
            img[i, j, k] = ((i * w + j) * c + k) * 0.013 - 0.2;
        return img;
    }

    [Theory]
    [InlineData("valid", 2, 1)]
    [InlineData("valid", 3, 2)]
    [InlineData("same", 3, 2)]
    [InlineData("same", 2, 2)]      // sudý kernel = asymetrický padding, nejcitlivější případ
    public void Conv_backward_is_bit_identical_to_old_path(string padding, int kernel, int filters)
    {
        var x = new Tensor(Image(6, 6, 2));
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        My_DNN.MDNN Build()
        {
            var m = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 44);
            m.Layers.Add(new Conv(filters, kernel, new Tanh(), padding));
            m.Layers.Add(new Dense(5, new Tanh()));
            return m;
        }

        var oldModel = Build();
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Build();
        RunNewPath(newModel, x, target);

        AssertBitIdentical(expected, Fingerprint(newModel), $"Conv {padding} k={kernel} f={filters}");
    }

    [Fact]
    public void Conv_maxpool_dense_backward_is_bit_identical_to_old_path()
    {
        // Nejdelší řetěz: Conv → MaxPool → Dense → Dense(out). Prochází přesně těmi
        // větvemi „další vrstva je Dense", které konvence B ruší.
        var x = new Tensor(Image(8, 8, 1));
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        My_DNN.MDNN Build()
        {
            var m = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 55);
            m.Layers.Add(new Conv(2, 3, new Tanh(), "valid"));
            m.Layers.Add(new MaxPool(2));
            m.Layers.Add(new Dense(6, new Tanh()));
            return m;
        }

        var oldModel = Build();
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Build();
        RunNewPath(newModel, x, target);

        AssertBitIdentical(expected, Fingerprint(newModel), "Conv→MaxPool→Dense");
    }

    [Fact]
    public void Conv_as_output_layer_is_bit_identical_to_old_path()
    {
        // Výstupní Conv: stará cesta ho musí obsloužit hookem SeedOutputGradient,
        // nová ho dostane úplně stejně jako každou jinou vrstvu. Přesně tenhle případ
        // se kdysi TIŠE NEUČIL (bug 2c-1), takže si zaslouží vlastní test.
        var x = new Tensor(Image(5, 5, 1));

        My_DNN.MDNN Build()
        {
            var m = new My_DNN.MDNN(new Conv(2, 3, new Tanh(), "valid"), new SGD(0.01), new MSE(), seed: 66);
            return m;
        }

        var probe = Build();
        Tensor shape = probe.GetResults(x);
        var target = new Tensor(shape.Data.Select(v => v * 0.5 + 0.1).ToArray(), shape.Shape);

        var oldModel = Build();
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Build();
        RunNewPath(newModel, x, target);

        AssertBitIdentical(expected, Fingerprint(newModel), "výstupní Conv");
    }

    [Fact]
    public void RNN_backward_is_bit_identical_to_old_path()
    {
        // RTRL: gImm bylo ve staré cestě SKRYTÉ POLE jen proto, že si ho musely předat
        // dvě metody. V konvenci B je to lokálka — a musí z toho vyjít totéž.
        var x = new Tensor(new double[] { 0.5, -0.3 });
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        My_DNN.MDNN Build()
        {
            var m = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 77);
            m.Layers.Add(new RNN(4, new Tanh()));
            return m;
        }

        var oldModel = Build();
        oldModel.Train.Fit(x, target);
        double[] expected = Fingerprint(oldModel);

        var newModel = Build();
        RunNewPath(newModel, x, target);

        AssertBitIdentical(expected, Fingerprint(newModel), "RNN");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]      // POVINNĚ i s dropoutem: jinak není žádná delta přesně nulová
    [InlineData(0.8)]      // a větev pro vypnutý neuron se v gradInput nikdy neprojde
    public void Backward_returns_gradient_with_respect_to_input(double rate)
    {
        // ∂L/∂input je to, co konvence B navíc VRACÍ — dnes si ho vrstva pod Dense
        // musí dopočítat sama z cizích vah. Ověřeno numericky proti perturbaci vstupu.
        //
        // Proč i s dropoutem: negativní kontrola ukázala, že varianta jen bez dropoutu
        // NEODHALÍ chybu ve zpracování nulových delt (záměna continue/break v akumulaci
        // gradInput prošla). Bez vypnutých neuronů se ta větev prostě nikdy nevykoná.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        var model = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 21);
        model.Layers.Add(new Dense(9, new Tanh(), dropout: rate));

        // Zmrazená maska: numerická derivace musí měřit pokaždé TUTÉŽ funkci.
        model.Context.IsTraining = true;
        model.GetResults(x);
        foreach (Dense d in model.Layers.Layers.Cast<Dense>())
            d.FreezeDropoutMaskForTests = true;

        if (rate > 0)
        {
            double[] mask = ((Dense)model.Layers.Layers[0]).DropoutMaskForTests!;
            Assert.Contains(mask, m => m == 0.0);   // pojistka: opravdu je co vypnout
        }

        Tensor output = model.GetResults(x);

        var layers = model.Layers.Layers;
        var grad = new Tensor(OutputGradient(model, output, target));
        for (int i = layers.Count - 1; i >= 0; i--)
            grad = layers[i].Backward(grad);

        double[] analytic = grad.Data;

        for (int j = 0; j < x.Data.Length; j++)
        {
            double orig = x.Data[j];
            const double eps = 1e-6;

            x.Data[j] = orig + eps;
            double lossPlus = model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);
            x.Data[j] = orig - eps;
            double lossMinus = model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);
            x.Data[j] = orig;

            double numeric = (lossPlus - lossMinus) / (2 * eps);
            Assert.True(Numeric.RelErr(analytic[j], numeric) < 1e-5,
                $"∂L/∂input[{j}]: analytic {analytic[j]:G17}, numeric {numeric:G17}");
        }
    }
}

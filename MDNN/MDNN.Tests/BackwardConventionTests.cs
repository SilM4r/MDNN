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
        }
        return values.ToArray();
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

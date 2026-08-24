using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Testy regularizace (L2 weight decay, dropout).
///
/// Zásada z Fáze 1 platí i tady: nová schopnost se nepřidává bez gradient checku.
/// U L2 to znamená ověřit, že efektivní gradient odpovídá derivaci účelové funkce
/// VČETNĚ trestného členu — tedy loss + λ/2·Σw².
/// </summary>
public class RegularizationTests
{
    // ============================================================ L2

    /// <summary>Součet čtverců všech regularizovaných parametrů — tedy vah, NE biasů.
    /// Musí sedět s tím, co dělá Neuron.Update_weights_bias, jinak testujeme něco jiného.</summary>
    private static double SumOfSquaredWeights(My_DNN.MDNN model)
    {
        double sum = 0;
        foreach (var layer in model.Layers.Layers)
        {
            if (layer is Dense dense)
                foreach (var n in dense.Neurons)
                    foreach (double w in n.Weights)
                        sum += w * w;
        }
        return sum;
    }

    [Theory]
    [InlineData(0.0)]      // vypnuto — musí vyjít stejně jako bez regularizace
    [InlineData(0.01)]
    [InlineData(0.5)]      // schválně velké, ať trestný člen převáží nad loss
    public void L2_effective_gradient_matches_numeric(double lambda)
    {
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });
        var target = new Tensor(new double[] { 0.0, 1.0 });

        var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 7);
        model.Layers.Add(new Dense(4, new Tanh()));
        model.Context.L2 = lambda;

        // dávka o jednom vzorku → gradientsW se nedělí ničím jiným než 1
        model.Train.Fit(x, target);

        var neurons = new List<Neuron>();
        foreach (var layer in model.Layers.Layers)
            neurons.AddRange(((Dense)layer).Neurons);

        // efektivní gradient = to, co se doopravdy předá optimizeru
        var analytic = neurons
            .Select(n => n.Weights.Select((w, i) => n.gradientsW[i] + lambda * w).ToArray())
            .ToList();

        // účelová funkce VČETNĚ trestu — bez toho by se čísla nikdy nepotkala
        Func<double> objectiveAt = () =>
            model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data)
            + lambda / 2.0 * SumOfSquaredWeights(model);

        double maxRel = 0; int worstI = -1, worstJ = -1;
        for (int i = 0; i < neurons.Count; i++)
        for (int j = 0; j < neurons[i].Weights.Length; j++)
        {
            double num = Numeric.ParamGrad(objectiveAt, ref neurons[i].Weights[j]);
            double rel = Numeric.RelErr(analytic[i][j], num);
            if (rel > maxRel) { maxRel = rel; worstI = i; worstJ = j; }
        }

        Assert.True(maxRel < 1e-5,
            $"λ={lambda}: maxRel={maxRel} u neuronu {worstI} váhy {worstJ} " +
            $"(analytic={analytic[worstI][worstJ]})");
    }

    [Fact]
    public void L2_does_not_touch_biases()
    {
        // Bias se NESMÍ regularizovat. Test to chytne tak, že model dostane nulový
        // gradient (predikce == cíl), takže jedinou silou, která by mohla parametry
        // hnout, je právě weight decay.
        var x = new Tensor(new double[] { 1.0, 1.0 });

        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 3);
        model.Context.L2 = 0.5;

        model.GetResults(x);                       // materializace
        var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
        foreach (var n in neurons)
            n.SetParamsForTests(new double[] { 0.25, -0.75 }, 2.0);

        // cíl = přesně to, co síť vydá → gradient z loss je nulový
        var target = new Tensor((double[])model.GetResults(x).Data.Clone());

        model.Train.Fit(x, target);
        model.Train.UpdateParams();

        foreach (var n in neurons)
        {
            Assert.Equal(2.0, n.Bias);                          // bias BEZE ZMĚNY
            Assert.True(Math.Abs(n.Weights[0]) < 0.25,          // váhy staženy k nule
                $"váha 0 se nezmenšila: {n.Weights[0]}");
            Assert.True(Math.Abs(n.Weights[1]) < 0.75,
                $"váha 1 se nezmenšila: {n.Weights[1]}");
        }
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.03)]
    public void L2_is_actually_applied_by_the_library(double lambda)
    {
        // Gradient check výš ověřuje VZOREC (grad + λ·w), ale počítá si ho sám v testu —
        // takže sám o sobě nedokáže, že ho knihovna doopravdy předá optimizeru.
        // Tady se to uzavírá: se SGD je krok přesně w -= lr·gradient, takže z posunu
        // váhy jde efektivní gradient zpětně dopočítat a porovnat.
        const double lr = 0.1;

        var x = new Tensor(new double[] { 0.4, -0.6, 0.2 });
        var target = new Tensor(new double[] { 0.3, -0.1 });

        var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(lr), new MSE(), seed: 5);
        model.Layers.Add(new Dense(3, new Tanh()));
        model.Context.L2 = lambda;

        model.GetResults(x);      // materializace

        // Biasy MUSÍ být nenulové. Startují na nule, takže by chybný člen λ·bias
        // vyšel taky nulový a test by rozbitou implementaci propustil — ověřeno
        // negativní kontrolou, tahle díra tu opravdu byla.
        var allNeurons = model.Layers.Layers.SelectMany(l => ((Dense)l).Neurons).ToList();
        for (int i = 0; i < allNeurons.Count; i++)
        {
            double[] w = allNeurons[i].Weights.Select((_, j) => 0.2 + 0.1 * j).ToArray();
            allNeurons[i].SetParamsForTests(w, 0.5 + 0.25 * i);
        }

        model.Train.Fit(x, target);

        var neurons = allNeurons;

        // co OČEKÁVÁME, že optimizer dostane, a jaké váhy z toho vyjdou
        var before = neurons.Select(n => (double[])n.Weights.Clone()).ToList();
        var expected = neurons
            .Select(n => n.Weights
                .Select((w, i) => w - lr * (n.gradientsW[i] + lambda * w)).ToArray())
            .ToList();
        double[] biasBefore = neurons.Select(n => n.Bias).ToArray();
        double[] biasGrad = neurons.Select(n => n.gradientsB).ToArray();

        model.Train.UpdateParams();

        for (int i = 0; i < neurons.Count; i++)
        {
            for (int j = 0; j < neurons[i].Weights.Length; j++)
                Assert.True(Math.Abs(expected[i][j] - neurons[i].Weights[j]) < 1e-12,
                    $"λ={lambda}, neuron {i} váha {j}: čekáno {expected[i][j]:G17}, " +
                    $"dostal {neurons[i].Weights[j]:G17} (před krokem {before[i][j]:G17})");

            // bias musí jít čistě po gradientu, BEZ členu λ·b
            double expectedBias = biasBefore[i] - lr * biasGrad[i];
            Assert.True(Math.Abs(expectedBias - neurons[i].Bias) < 1e-12,
                $"λ={lambda}, neuron {i}: bias má jít po čistém gradientu — " +
                $"čekáno {expectedBias:G17}, dostal {neurons[i].Bias:G17}");
        }
    }

    // ============================================================ DROPOUT

    [Theory]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void Dropout_gradient_matches_numeric_with_frozen_mask(double rate)
    {
        // Gradient check dropoutu. Maska se ZMRAZÍ, jinak by každý forward losoval jinou
        // a numerická derivace by měřila pokaždé jinou funkci.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4 });

        var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 21);
        model.Layers.Add(new Dense(12, new Tanh(), dropout: rate));

        model.Context.IsTraining = true;         // ruční režim: chceme dropout i v GetResults
        model.Train.Fit(x, target);              // vylosuje masku

        var denseLayers = model.Layers.Layers.Cast<Dense>().ToList();
        foreach (var d in denseLayers) d.FreezeDropoutMaskForTests = true;

        // pojistka, že se opravdu něco zahodilo — jinak by test nic netestoval
        double[] mask = denseLayers[0].DropoutMaskForTests!;
        Assert.Contains(mask, m => m == 0.0);

        var neurons = denseLayers.SelectMany(l => l.Neurons).ToList();
        var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

        Func<double> lossAt = () =>
            model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);

        double maxRel = 0; int worstI = -1, worstJ = -1;
        for (int i = 0; i < neurons.Count; i++)
        for (int j = 0; j < neurons[i].Weights.Length; j++)
        {
            double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
            double rel = Numeric.RelErr(analytic[i][j], num);
            if (rel > maxRel) { maxRel = rel; worstI = i; worstJ = j; }
        }

        Assert.True(maxRel < 1e-5,
            $"rate={rate}: maxRel={maxRel} u neuronu {worstI} váhy {worstJ}");
    }

    [Fact]
    public void Dropout_is_off_during_prediction()
    {
        // Nejdůležitější vlastnost: predikce musí být DETERMINISTICKÁ. Kdyby dropout
        // zůstal zapnutý, dal by stejný vstup pokaždé jinou odpověď.
        var model = new My_DNN.MDNN(new Dense(3, new Linear()), new SGD(0.01), new MSE(), seed: 4);
        model.Layers.Add(new Dense(32, new Tanh(), dropout: 0.5));

        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });

        double[] first = (double[])model.GetResults(x).Data.Clone();
        for (int i = 0; i < 20; i++)
            Assert.Equal(first, model.GetResults(x).Data);

        Assert.False(model.Context.IsTraining, "GetResults nesmí nechat model v trénovacím režimu");
    }

    [Fact]
    public void Dropout_zeroes_roughly_the_requested_fraction_and_scales_survivors()
    {
        const double rate = 0.4;
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 99);
        model.Layers.Add(new Dense(2000, new Linear(), dropout: rate));

        var x = new Tensor(new double[] { 1.0, 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        double[] mask = ((Dense)model.Layers.Layers[0]).DropoutMaskForTests!;

        double zeroed = mask.Count(m => m == 0.0) / (double)mask.Length;
        Assert.InRange(zeroed, rate - 0.05, rate + 0.05);

        // přeživší musí být zesílení PŘESNĚ na 1/(1-p), ne přibližně
        double expectedScale = 1.0 / (1.0 - rate);
        foreach (double m in mask.Where(m => m != 0.0))
            Assert.Equal(expectedScale, m, 12);
    }

    [Fact]
    public void Dropout_global_applies_to_hidden_but_not_output_layer()
    {
        var model = new My_DNN.MDNN(new Dense(4, new Linear()), new SGD(0.01), new MSE(), seed: 8);
        model.Layers.Add(new Dense(50, new Linear()));       // bez vlastní hodnoty → zdědí globální
        model.Context.Dropout = 0.5;

        var x = new Tensor(new double[] { 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        var hidden = (Dense)model.Layers.Layers[0];
        var outputLayer = (Dense)model.Layers.Layers[1];

        Assert.NotNull(hidden.DropoutMaskForTests);
        Assert.Null(outputLayer.DropoutMaskForTests);        // výstupní vrstva NEsmí zahazovat
    }

    [Fact]
    public void Dropout_per_layer_value_overrides_global()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 8);
        model.Layers.Add(new Dense(40, new Linear(), dropout: 0.0));   // explicitní VYPNUTÍ
        model.Layers.Add(new Dense(40, new Linear()));                 // zdědí globální
        model.Context.Dropout = 0.5;

        var x = new Tensor(new double[] { 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        Assert.Null(((Dense)model.Layers.Layers[0]).DropoutMaskForTests);      // vypnuto explicitně
        Assert.NotNull(((Dense)model.Layers.Layers[1]).DropoutMaskForTests);   // zdědilo 0.5
    }

    [Fact]
    public void Dropout_dropped_neuron_gets_no_weight_gradient()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 13);
        model.Layers.Add(new Dense(30, new Tanh(), dropout: 0.5));

        var x = new Tensor(new double[] { 0.7, -0.2 });
        var target = new Tensor(new double[] { 0.1, 0.4 });

        model.Context.IsTraining = true;
        model.Train.Fit(x, target);

        var hidden = (Dense)model.Layers.Layers[0];
        double[] mask = hidden.DropoutMaskForTests!;

        for (int i = 0; i < hidden.Neurons.Count; i++)
        {
            if (mask[i] != 0.0) continue;

            Assert.All(hidden.Neurons[i].gradientsW, g => Assert.Equal(0.0, g));
            Assert.Equal(0.0, hidden.Neurons[i].gradientsB);
        }

        Assert.Contains(mask, m => m == 0.0);      // pojistka, že se něco zahodilo
    }

    [Fact]
    public void Dropout_averages_over_the_whole_batch_not_just_active_samples()
    {
        // Past, kterou by nic jiného nechytilo: neuron vypnutý ve 3 z 8 vzorků má
        // gradient nasčítaný jen z 5 vzorků, ale dělit se MUSÍ osmi. Kdyby se dělilo
        // pěti, dostal by ten neuron o 60 % nafouknutý krok — a nešlo by to poznat
        // jinak než tím, že se model učí divně.
        const double lr = 0.1;
        const int batch = 8;

        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(lr), new MSE(), seed: 17);
        model.Layers.Add(new Dense(16, new Tanh(), dropout: 0.5));

        model.Context.IsTraining = true;

        for (int s = 0; s < batch; s++)
        {
            var x = new Tensor(new double[] { 0.3 + s * 0.1, -0.2 - s * 0.05 });
            var t = new Tensor(new double[] { 0.1, -0.3 });
            model.Train.Fit(x, t);
        }

        var hidden = (Dense)model.Layers.Layers[0];
        var before = hidden.Neurons.Select(n => (double[])n.Weights.Clone()).ToList();
        var grads = hidden.Neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

        model.Train.UpdateParams();

        for (int i = 0; i < hidden.Neurons.Count; i++)
        for (int j = 0; j < hidden.Neurons[i].Weights.Length; j++)
        {
            double expected = before[i][j] - lr * (grads[i][j] / batch);
            Assert.True(Math.Abs(expected - hidden.Neurons[i].Weights[j]) < 1e-12,
                $"neuron {i} váha {j}: čekáno dělení {batch} vzorky → {expected:G17}, " +
                $"dostal {hidden.Neurons[i].Weights[j]:G17}");
        }
    }

    [Fact]
    public void Regularization_survives_save_and_load()
    {
        // Tichý zabiják stejného druhu jako kdysi ztracená aktivace u Conv: model by se
        // uložil, načetl a dotrénovával UŽ BEZ regularizace — a nic by to nehlásilo.
        string path = Path.Combine(Path.GetTempPath(), $"mdnn_reg_{Guid.NewGuid():N}.json");

        try
        {
            var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 31);
            model.Layers.Add(new Dense(5, new Tanh(), dropout: 0.25));   // vlastní hodnota
            model.Layers.Add(new Dense(5, new Tanh()));                  // dědí globální
            model.Context.L2 = 0.017;
            model.Context.Dropout = 0.4;

            model.GetResults(new Tensor(new double[] { 0.5, -0.2 }));    // materializace
            model.SaveAsJson(path);

            var loaded = My_DNN.MDNN.LoadModel(path);

            Assert.Equal(0.017, loaded.Context.L2);
            Assert.Equal(0.4, loaded.Context.Dropout);
            Assert.Equal(0.25, ((Dense)loaded.Layers.Layers[0]).DropoutRate);

            // null se MUSÍ zachovat jako null, ne spadnout na nulu — jinak by vrstva
            // přestala dědit globální hodnotu a dropout by na ní tiše zmizel
            Assert.Null(((Dense)loaded.Layers.Layers[1]).DropoutRate);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Manual_training_loop_behaves_like_Fit()
    {
        // README nabízí ruční trénink jako „Fit rozdělený na dva kroky".
        // Musí to tedy dát TOTÉŽ — a s dropoutem to chvíli neplatilo: Train.FeedForward
        // volalo GetResults, které trénovací režim nezapíná, takže ruční cesta tiše
        // běžela BEZ dropoutu.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4 });

        double[] Run(bool manual)
        {
            var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.1), new MSE(), seed: 91);
            model.Layers.Add(new Dense(12, new Tanh(), dropout: 0.5));

            if (manual)
            {
                model.Train.FeedForward(x);
                model.Train.BackPropagation(target);
            }
            else
            {
                model.Train.Fit(x, target);
            }

            model.Train.UpdateParams();

            return model.Layers.Layers
                .SelectMany(l => ((Dense)l).Neurons)
                .SelectMany(n => n.Weights)
                .ToArray();
        }

        double[] viaFit = Run(manual: false);
        double[] viaManual = Run(manual: true);

        for (int i = 0; i < viaFit.Length; i++)
            Assert.True(viaFit[i].Equals(viaManual[i]),
                $"váha {i}: Fit {viaFit[i]:G17}, ruční cesta {viaManual[i]:G17}");
    }

    [Fact]
    public void Dropout_rejects_invalid_rate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLu(), dropout: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLu(), dropout: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLu(), dropout: double.NaN));

        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.Dropout = 1.0);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.Dropout = -0.5);
    }

    [Fact]
    public void L2_rejects_negative_coefficient()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.L2 = -0.1);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.L2 = double.NaN);
        Assert.Equal(0.0, model.Context.L2);          // neplatné přiřazení nic nezměnilo
    }

    [Fact]
    public void L2_shrinks_weights_over_training()
    {
        // Chování v celku: se stejným seedem a stejnými daty musí model s L2 skončit
        // s menší normou vah než bez něj. Tohle je ten důvod, proč regularizace existuje.
        double NormAfterTraining(double lambda)
        {
            var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed: 11);
            model.Layers.Add(new Dense(8, new Tanh()));
            model.Context.L2 = lambda;

            var xs = new double[20][];
            var ys = new double[20][];
            for (int i = 0; i < 20; i++)
            {
                double v = i * 0.1 - 1.0;
                xs[i] = new double[] { v, v * v };
                ys[i] = new double[] { Math.Sin(v) };
            }

            model.Train.TrainLoop(xs, ys, 40, 4);
            return SumOfSquaredWeights(model);
        }

        double plain = NormAfterTraining(0.0);
        double regularized = NormAfterTraining(0.05);

        Assert.True(regularized < plain,
            $"L2 nezmenšilo váhy: bez regularizace Σw²={plain}, s L2 Σw²={regularized}");
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// Regularization tests (L2 weight decay, dropout).
///
/// The usual rule applies here too: no new capability without a gradient check.
/// For L2 that means verifying that the effective gradient matches the derivative of the
/// objective INCLUDING the penalty term — i.e. loss + λ/2·Σw².
/// </summary>
public class RegularizationTests
{
    // ============================================================ L2

    /// <summary>Sum of squares of all regularized parameters — i.e. weights, NOT biases.
    /// Must match what Neuron.UpdateWeightsAndBias does, otherwise we test something else.</summary>
    private static double SumOfSquaredWeights(NeuralNetwork model)
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
    [InlineData(0.0)]      // off — must match the unregularized result
    [InlineData(0.01)]
    [InlineData(0.5)]      // deliberately large so the penalty term dominates the loss
    public void L2_effective_gradient_matches_numeric(double lambda)
    {
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });
        var target = new Tensor(new double[] { 0.0, 1.0 });

        var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 7);
        model.Layers.Add(new Dense(4, new Tanh()));
        model.Context.L2 = lambda;

        // single-sample batch → gradientsW is divided by nothing but 1
        model.Train.Fit(x, target);

        var neurons = new List<Neuron>();
        foreach (var layer in model.Layers.Layers)
            neurons.AddRange(((Dense)layer).Neurons);

        // effective gradient = what actually gets passed to the optimizer
        var analytic = neurons
            .Select(n => n.Weights.Select((w, i) => n.gradientsW[i] + lambda * w).ToArray())
            .ToList();

        // objective INCLUDING the penalty — without it the numbers would never match
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
            $"λ={lambda}: maxRel={maxRel} at neuron {worstI} weight {worstJ} " +
            $"(analytic={analytic[worstI][worstJ]})");
    }

    [Fact]
    public void L2_does_not_touch_biases()
    {
        // The bias must NOT be regularized. The test catches this by giving the model a zero
        // gradient (prediction == target), so the only force that could move the parameters
        // is weight decay itself.
        var x = new Tensor(new double[] { 1.0, 1.0 });

        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 3);
        model.Context.L2 = 0.5;

        model.GetResults(x);                       // materialize the layers
        var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
        foreach (var n in neurons)
            n.SetParamsForTests(new double[] { 0.25, -0.75 }, 2.0);

        // target = exactly what the network outputs → the loss gradient is zero
        var target = new Tensor((double[])model.GetResults(x).Data.Clone());

        model.Train.Fit(x, target);
        model.Train.UpdateParams();

        foreach (var n in neurons)
        {
            Assert.Equal(2.0, n.Bias);                          // bias UNCHANGED
            Assert.True(Math.Abs(n.Weights[0]) < 0.25,          // weights pulled towards zero
                $"weight 0 did not shrink: {n.Weights[0]}");
            Assert.True(Math.Abs(n.Weights[1]) < 0.75,
                $"weight 1 did not shrink: {n.Weights[1]}");
        }
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.03)]
    public void L2_is_actually_applied_by_the_library(double lambda)
    {
        // The gradient check above verifies the FORMULA (grad + λ·w), but computes it in the
        // test itself — so on its own it does not prove the library actually passes it to the
        // optimizer. This closes the gap: with SGD the step is exactly w -= lr·gradient, so the
        // effective gradient can be recovered from the weight change and compared.
        const double lr = 0.1;

        var x = new Tensor(new double[] { 0.4, -0.6, 0.2 });
        var target = new Tensor(new double[] { 0.3, -0.1 });

        var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(lr), new MSE(), seed: 5);
        model.Layers.Add(new Dense(3, new Tanh()));
        model.Context.L2 = lambda;

        model.GetResults(x);      // materialize the layers

        // Biases MUST be non-zero. They start at zero, so a wrong λ·bias term would also
        // come out as zero and the test would pass a broken implementation — confirmed by
        // a negative control, this hole really existed.
        var allNeurons = model.Layers.Layers.SelectMany(l => ((Dense)l).Neurons).ToList();
        for (int i = 0; i < allNeurons.Count; i++)
        {
            double[] w = allNeurons[i].Weights.Select((_, j) => 0.2 + 0.1 * j).ToArray();
            allNeurons[i].SetParamsForTests(w, 0.5 + 0.25 * i);
        }

        model.Train.Fit(x, target);

        var neurons = allNeurons;

        // what we EXPECT the optimizer to receive, and the resulting weights
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
                    $"λ={lambda}, neuron {i} weight {j}: expected {expected[i][j]:G17}, " +
                    $"got {neurons[i].Weights[j]:G17} (before the step {before[i][j]:G17})");

            // the bias must follow the plain gradient, WITHOUT the λ·b term
            double expectedBias = biasBefore[i] - lr * biasGrad[i];
            Assert.True(Math.Abs(expectedBias - neurons[i].Bias) < 1e-12,
                $"λ={lambda}, neuron {i}: the bias should follow the plain gradient — " +
                $"expected {expectedBias:G17}, got {neurons[i].Bias:G17}");
        }
    }

    // ============================================================ DROPOUT

    [Theory]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void Dropout_gradient_matches_numeric_with_frozen_mask(double rate)
    {
        // Dropout gradient check. The mask is FROZEN, otherwise every forward would draw a new
        // one and the numeric derivative would measure a different function each time.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4 });

        var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE(), seed: 21);
        model.Layers.Add(new Dense(12, new Tanh(), dropout: rate));

        model.Context.IsTraining = true;         // manual mode: we want dropout in GetResults too
        model.Train.Fit(x, target);              // draws the mask

        var denseLayers = model.Layers.Layers.Cast<Dense>().ToList();
        foreach (var d in denseLayers) d.FreezeDropoutMaskForTests = true;

        // safeguard that something really was dropped — otherwise the test would test nothing
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
            $"rate={rate}: maxRel={maxRel} at neuron {worstI} weight {worstJ}");
    }

    [Fact]
    public void Dropout_is_off_during_prediction()
    {
        // The most important property: prediction must be DETERMINISTIC. If dropout stayed
        // on, the same input would give a different answer every time.
        var model = new NeuralNetwork(new Dense(3, new Linear()), new SGD(0.01), new MSE(), seed: 4);
        model.Layers.Add(new Dense(32, new Tanh(), dropout: 0.5));

        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });

        double[] first = (double[])model.GetResults(x).Data.Clone();
        for (int i = 0; i < 20; i++)
            Assert.Equal(first, model.GetResults(x).Data);

        Assert.False(model.Context.IsTraining, "GetResults must not leave the model in training mode");
    }

    [Fact]
    public void Dropout_zeroes_roughly_the_requested_fraction_and_scales_survivors()
    {
        const double rate = 0.4;
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 99);
        model.Layers.Add(new Dense(2000, new Linear(), dropout: rate));

        var x = new Tensor(new double[] { 1.0, 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        double[] mask = ((Dense)model.Layers.Layers[0]).DropoutMaskForTests!;

        double zeroed = mask.Count(m => m == 0.0) / (double)mask.Length;
        Assert.InRange(zeroed, rate - 0.05, rate + 0.05);

        // survivors must be scaled EXACTLY by 1/(1-p), not approximately
        double expectedScale = 1.0 / (1.0 - rate);
        foreach (double m in mask.Where(m => m != 0.0))
            Assert.Equal(expectedScale, m, 12);
    }

    [Fact]
    public void Dropout_global_applies_to_hidden_but_not_output_layer()
    {
        var model = new NeuralNetwork(new Dense(4, new Linear()), new SGD(0.01), new MSE(), seed: 8);
        model.Layers.Add(new Dense(50, new Linear()));       // no own value → inherits the global one
        model.Context.Dropout = 0.5;

        var x = new Tensor(new double[] { 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        var hidden = (Dense)model.Layers.Layers[0];
        var outputLayer = (Dense)model.Layers.Layers[1];

        Assert.NotNull(hidden.DropoutMaskForTests);
        Assert.Null(outputLayer.DropoutMaskForTests);        // the output layer must NOT drop
    }

    // Softmax normalizes across the whole layer, so a dropped neuron affects the other outputs
    // and its weight gradient is NOT zero. Backward skipped it anyway (SkipSample) → a silently
    // wrong gradient. Explicit dropout on a softmax layer is therefore an error; the global one skips it.
    [Fact]
    public void Dropout_on_softmax_layer_is_rejected_when_explicit()
    {
        var model = new NeuralNetwork(new Dense(3, new Softmax(), dropout: 0.5), new SGD(0.01), new CrossEntropy(), seed: 8);
        model.Layers.Add(new Dense(4, new Linear()));

        var x = new Tensor(new double[] { 1.0, 1.0 });
        var t = new Tensor(new double[] { 0.0, 1.0, 0.0 });

        var ex = Assert.Throws<NotSupportedException>(() => model.Train.Fit(x, t));
        Assert.Contains("Softmax", ex.Message);

        model.GetResults(x);   // prediction does not use dropout, so this passes
    }

    [Fact]
    public void Dropout_global_skips_hidden_softmax_layer()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 8);
        model.Layers.Add(new Dense(5, new Softmax()));   // hidden softmax layer, no own value
        model.Context.Dropout = 0.5;

        model.Train.Fit(new Tensor(new double[] { 1.0, 1.0 }), new Tensor(new double[] { 0.0, 1.0 }));

        Assert.Null(((Dense)model.Layers.Layers[0]).DropoutMaskForTests);
    }

    [Fact]
    public void Dropout_per_layer_value_overrides_global()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 8);
        model.Layers.Add(new Dense(40, new Linear(), dropout: 0.0));   // explicitly DISABLED
        model.Layers.Add(new Dense(40, new Linear()));                 // inherits the global one
        model.Context.Dropout = 0.5;

        var x = new Tensor(new double[] { 1.0, 1.0 });
        model.Context.IsTraining = true;
        model.GetResults(x);

        Assert.Null(((Dense)model.Layers.Layers[0]).DropoutMaskForTests);      // explicitly disabled
        Assert.NotNull(((Dense)model.Layers.Layers[1]).DropoutMaskForTests);   // inherited 0.5
    }

    [Fact]
    public void Dropout_dropped_neuron_gets_no_weight_gradient()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 13);
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

        Assert.Contains(mask, m => m == 0.0);      // safeguard that something was dropped
    }

    [Fact]
    public void Dropout_averages_over_the_whole_batch_not_just_active_samples()
    {
        // A trap nothing else would catch: a neuron dropped in 3 of 8 samples has its
        // gradient summed from only 5 samples, but it MUST be divided by eight. Dividing by
        // five would give that neuron a 60 % inflated step — and the only symptom would be
        // that the model learns strangely.
        const double lr = 0.1;
        const int batch = 8;

        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(lr), new MSE(), seed: 17);
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
                $"neuron {i} weight {j}: expected division by {batch} samples → {expected:G17}, " +
                $"got {hidden.Neurons[i].Weights[j]:G17}");
        }
    }

    [Fact]
    public void Regularization_survives_save_and_load()
    {
        // A silent killer of the same kind as the once-lost Conv activation: the model would be
        // saved, loaded and further trained WITHOUT regularization — and nothing would report it.
        string path = Path.Combine(Path.GetTempPath(), $"mdnn_reg_{Guid.NewGuid():N}.json");

        try
        {
            var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 31);
            model.Layers.Add(new Dense(5, new Tanh(), dropout: 0.25));   // own value
            model.Layers.Add(new Dense(5, new Tanh()));                  // inherits the global one
            model.Context.L2 = 0.017;
            model.Context.Dropout = 0.4;

            model.GetResults(new Tensor(new double[] { 0.5, -0.2 }));    // materialize the layers
            model.SaveAsJson(path);

            var loaded = NeuralNetwork.LoadModel(path);

            Assert.Equal(0.017, loaded.Context.L2);
            Assert.Equal(0.4, loaded.Context.Dropout);
            Assert.Equal(0.25, ((Dense)loaded.Layers.Layers[0]).DropoutRate);

            // null MUST stay null, not collapse to zero — otherwise the layer would stop
            // inheriting the global value and dropout would silently vanish from it
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
        // The README presents manual training as "Fit split into two steps".
        // So it must produce the SAME result — and with dropout it briefly did not:
        // Train.FeedForward called GetResults, which does not enable training mode, so the
        // manual path silently ran WITHOUT dropout.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4 });

        double[] Run(bool manual)
        {
            var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.1), new MSE(), seed: 91);
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
                $"weight {i}: Fit {viaFit[i]:G17}, manual path {viaManual[i]:G17}");
    }

    [Fact]
    public void Dropout_rejects_invalid_rate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLU(), dropout: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLU(), dropout: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(4, new ReLU(), dropout: double.NaN));

        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.Dropout = 1.0);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.Dropout = -0.5);
    }

    [Fact]
    public void L2_rejects_negative_coefficient()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.L2 = -0.1);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Context.L2 = double.NaN);
        Assert.Equal(0.0, model.Context.L2);          // the invalid assignment changed nothing
    }

    [Fact]
    public void L2_shrinks_weights_over_training()
    {
        // End-to-end behaviour: with the same seed and data, the model with L2 must end up
        // with a smaller weight norm than without it. That is why regularization exists.
        double NormAfterTraining(double lambda)
        {
            var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed: 11);
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
            $"L2 did not shrink the weights: without regularization Σw²={plain}, with L2 Σw²={regularized}");
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// <c>Backward(gradOutput) → gradInput</c>: the layer returns the correct gradient with respect
/// to its INPUT, checked numerically against input perturbation.
///
/// This is the only place where <c>Backward</c> is chained MANUALLY, outside <c>Gradient.RunBackward</c>.
/// Since <c>Layer.Backward</c> is public API, it is worth keeping.
///
/// ---------------------------------------------------------------------------------------
/// What this file REPLACES: a lesson worth remembering
/// ---------------------------------------------------------------------------------------
/// It used to be `BackwardConventionTests` with 16 "old and new path are bit-identical" tests.
/// They were written for the switch to the single <c>Backward</c> convention and did their job:
/// they checked that the new `Backward` produced BIT-identical gradients to the former trio
/// CalculateLayerGradients + SeedOutputGradient + BackPropagation.
///
/// But step 3 of that refactor (b2b02fb) DELETED the old path, and the tests turned into
/// comparing the code with itself: both `oldModel.Train.Fit(...)` and the manual chain now
/// called the same `Backward`. The tests stayed green, the name and docs still promised a
/// safety net, and nobody noticed.
///
/// Confirmed by mutation testing (2026-08-26): a bug was injected into backward and we
/// watched who caught it:
///
///   mutation                                BackwardConventionTests   rest of suite
///   Dense: dropout mask removed in backward          2 failed           2 failed
///   softmax+CE fusion disabled                       1 failed           3 failed
///   Conv: act' missing in backward                   0                 20 failed
///   RNN: extra act' in accumulation                  0                  3 failed
///   Dense: missing SkipSample for dropout            0                  1 failed
///
/// They caught NOTHING extra and missed three bugs out of five, because the mutation hit both
/// sides of the comparison equally. That is worse than zero value: a suite that looks like a
/// safety net and protects nothing.
///
/// **Lesson:** transitional "old vs. new" tests only make sense while the old path exists.
/// When a future refactor uses the same three steps (add alongside → switch → delete), those
/// tests must be deleted together with the old path.
/// </summary>
public class InputGradientTests
{
    /// <summary>∂L/∂output of the output layer; the same thing Gradient.RunBackward computes
    /// before passing it to the first Backward.</summary>
    private static double[] OutputGradient(NeuralNetwork model, Tensor output, Tensor target)
    {
        double[] d = new double[output.Data.Length];
        for (int i = 0; i < d.Length; i++)
            d[i] = model.Context.Loss.DerivativeOfLossFunction(output.Data[i], target.Data[i]);
        return d;
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]      // dropout is REQUIRED: otherwise no delta is exactly zero
    [InlineData(0.8)]      // and the dropped-neuron branch in gradInput is never exercised
    public void Backward_returns_gradient_with_respect_to_input(double rate)
    {
        // ∂L/∂input is what Backward RETURNS; previously the layer below Dense had to compute
        // it itself from the other layer's weights. Checked numerically against input perturbation.
        //
        // Why dropout too: a negative control showed that a dropout-free variant does NOT
        // catch a bug in handling zero deltas (swapping continue/break in the gradInput
        // accumulation passed). Without dropped neurons that branch simply never runs.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        var model = new NeuralNetwork(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 21);
        model.Layers.Add(new Dense(9, new Tanh(), dropout: rate));

        // Frozen mask: the numerical derivative must measure the SAME function every time.
        model.Context.IsTraining = true;
        model.GetResults(x);
        foreach (Dense d in model.Layers.Layers.Cast<Dense>())
            d.FreezeDropoutMaskForTests = true;

        if (rate > 0)
        {
            double[] mask = ((Dense)model.Layers.Layers[0]).DropoutMaskForTests!;
            Assert.Contains(mask, m => m == 0.0);   // sanity check: something is actually dropped
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

    /// <summary>`computeInputGradient: false` must return an EMPTY tensor for every layer.</summary>
    /// <remarks>
    /// This is the documented contract (Layer.cs): with false an EMPTY tensor is returned,
    /// not a meaningless one, so a mistake fails on the length instead of silently giving a wrong result.
    /// RNN was the only exception: it returned an array of zeros of the right length, which is
    /// exactly the meaningless tensor the contract is meant to prevent.</remarks>
    [Theory]
    [InlineData("dense")]
    [InlineData("conv")]
    [InlineData("maxpool")]
    [InlineData("rnn")]
    public void Backward_returns_an_empty_tensor_when_the_caller_does_not_want_the_input_gradient(string kind)
    {
        var x2d = new Tensor(new double[4, 4, 1]);
        var flat = new Tensor(new double[] { 0.5, -0.3 });

        (NeuralNetwork model, Tensor input) = kind switch
        {
            "conv" => (new NeuralNetwork(new Conv(2, 2, new Tanh(), "valid"), new SGD(0.01), new MSE(), seed: 5), x2d),
            "maxpool" => (new NeuralNetwork(new MaxPool(2), new SGD(0.01), new MSE(), seed: 5), x2d),
            "rnn" => (new NeuralNetwork(new RNN(3, new Tanh()), new SGD(0.01), new MSE(), seed: 5), flat),
            _ => (new NeuralNetwork(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 5), flat),
        };

        Tensor output = model.GetResults(input);
        Layer layer = model.Layers.Layers[0];

        Tensor gradInput = layer.Backward(
            new Tensor(new double[output.Data.Length]), computeInputGradient: false);

        Assert.Empty(gradInput.Data);
    }
}

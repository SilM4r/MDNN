using System;
using System.Linq;
using Xunit;
using MDNN;
using MDNN.Activations;

namespace MDNN.Tests;

/// <summary>
/// The dot product in <c>Neuron.FeedForward</c> sums in four independent chains (for speed) and
/// handles the remainder separately when the input count is not divisible by four.
///
/// Most other tests have 2–3 inputs, so they would only go through the remainder loop and never
/// run the main four-chain one. Here both run, with every remainder length.
/// </summary>
public class NeuronDotProductTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(784)]
    [InlineData(787)]
    public void Matches_a_plain_left_to_right_sum(int inputs)
    {
        var rnd = new Random(inputs);
        double[] w = Enumerable.Range(0, inputs).Select(_ => rnd.NextDouble() * 2 - 1).ToArray();
        double[] x = Enumerable.Range(0, inputs).Select(_ => rnd.NextDouble() * 2 - 1).ToArray();
        const double bias = 0.375;

        var neuron = new Neuron(inputs, new Linear());
        neuron.SetParamsForTests(w, bias);

        double expected = bias;
        for (int i = 0; i < inputs; i++) expected += w[i] * x[i];

        // Different summation order → may differ in the last bits, no more.
        Assert.Equal(expected, neuron.FeedForward(x), 12);
    }

    [Fact]
    public void Every_input_counts_exactly_once()
    {
        // Integers are exact in double, so the result must match EXACTLY here; an input that is
        // skipped or counted twice (a loop-bounds bug) would show immediately.
        const int inputs = 11;   // 2 full groups of four + remainder 3
        double[] w = Enumerable.Range(1, inputs).Select(i => (double)i).ToArray();
        double[] x = Enumerable.Range(1, inputs).Select(i => (double)(i * i)).ToArray();

        var neuron = new Neuron(inputs, new Linear());
        neuron.SetParamsForTests(w, 0);

        double expected = Enumerable.Range(1, inputs).Sum(i => (double)i * i * i);   // Σ i³
        Assert.Equal(expected, neuron.FeedForward(x));
    }
}

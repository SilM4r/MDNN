using System;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Activation_functions;

namespace MDNN.Tests;

/// <summary>
/// Skalární součin v <c>Neuron.FeedForward</c> sčítá ve čtyřech nezávislých řetězcích
/// (kvůli rychlosti) a zbytek, když počet vstupů není dělitelný čtyřmi, dopočítá zvlášť.
///
/// Většina ostatních testů má 2–3 vstupy, takže by prošla jen zbytkovou smyčkou a hlavní
/// čtyřřetězcovou by nikdy nespustila. Tady se projdou obě a všechny délky zbytku.
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

        // Jiné pořadí sčítání → může se lišit v posledních bitech, ne víc.
        Assert.Equal(expected, neuron.FeedForward(x), 12);
    }

    [Fact]
    public void Every_input_counts_exactly_once()
    {
        // Celá čísla jsou v double přesná, takže tady musí výsledek sedět PŘESNĚ —
        // vynechaný nebo dvakrát započtený vstup (chyba v mezích smyček) by se ukázal hned.
        const int inputs = 11;   // 2 celé čtveřice + zbytek 3
        double[] w = Enumerable.Range(1, inputs).Select(i => (double)i).ToArray();
        double[] x = Enumerable.Range(1, inputs).Select(i => (double)(i * i)).ToArray();

        var neuron = new Neuron(inputs, new Linear());
        neuron.SetParamsForTests(w, 0);

        double expected = Enumerable.Range(1, inputs).Sum(i => (double)i * i * i);   // Σ i³
        Assert.Equal(expected, neuron.FeedForward(x));
    }
}

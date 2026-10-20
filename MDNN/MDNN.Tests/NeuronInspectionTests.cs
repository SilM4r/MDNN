using System;
using System.Linq;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// The library is also a teaching aid: after a pass, every neuron exposes its weights, bias,
/// inputs, output before and after activation, activation and optimizer. Only the library may
/// write them (internal set), but they must always be readable.
/// </summary>
public class NeuronInspectionTests
{
    [Fact]
    public void Every_neuron_can_be_inspected_after_a_pass()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new Adam(0.01), new MSE(), seed: 1);
        model.Layers.Add(new Dense(3, new Tanh()));
        double[] x = { 0.5, -1.0, 2.0 };

        double[] output = model.GetResults(new Tensor(x)).Data;

        var hidden = ((NeuronLayer)model.Layers.Layers[0]).Neurons;
        var last = ((NeuronLayer)model.Layers.Layers[1]).Neurons;

        foreach (Neuron n in hidden)
        {
            Assert.Equal(x, n.Inputs);
            double raw = n.Bias + n.Weights.Zip(x, (w, v) => w * v).Sum();
            Assert.Equal(raw, n.RawOutput, 12);
            Assert.Equal(Math.Tanh(raw), n.Output, 12);
            Assert.Equal("Tanh", n.Activation.Name);
            Assert.IsType<Adam>(n.Optimizer);
        }

        // model output = outputs of the output layer's neurons
        Assert.Equal(output, last.Select(n => n.Output).ToArray());
    }
}

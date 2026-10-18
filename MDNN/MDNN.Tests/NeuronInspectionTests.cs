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
/// Knihovna je i učební pomůcka: po průchodu jde z každého neuronu vyčíst váhy, bias,
/// vstupy, výstup před aktivací i po ní, aktivaci a optimizer. Zapisovat do nich smí jen
/// knihovna (internal set), číst je musí jít vždycky.
/// </summary>
public class NeuronInspectionTests
{
    [Fact]
    public void Every_neuron_can_be_inspected_after_a_pass()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new Adam(0.01), new MSE(), seed: 1);
        model.Layers.Add(new Dense(3, new Tanh()));
        double[] x = { 0.5, -1.0, 2.0 };

        double[] output = model.GetResults(new Tensor(x)).Data;

        var hidden = ((LayerBasedOnNeurons)model.Layers.Layers[0]).Neurons;
        var last = ((LayerBasedOnNeurons)model.Layers.Layers[1]).Neurons;

        foreach (Neuron n in hidden)
        {
            Assert.Equal(x, n.Inputs);
            double raw = n.Bias + n.Weights.Zip(x, (w, v) => w * v).Sum();
            Assert.Equal(raw, n.RawOutput, 12);
            Assert.Equal(Math.Tanh(raw), n.Output, 12);
            Assert.Equal("Tanh", n.Activation_Func.Name);
            Assert.IsType<Adam>(n.Optimizer);
        }

        // výstup modelu = výstupy neuronů výstupní vrstvy
        Assert.Equal(output, last.Select(n => n.Output).ToArray());
    }
}

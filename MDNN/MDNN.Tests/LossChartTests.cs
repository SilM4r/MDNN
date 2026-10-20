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
/// The training loss chart lives in the separate MDNN.Plotting package. The core only offers the
/// <see cref="Train.LossChart"/> hook and does not need ScottPlot itself.
/// </summary>
public class LossChartTests
{
    private static (double[][] x, double[][] y) Data()
    {
        var x = Enumerable.Range(0, 20).Select(i => new[] { i / 20.0 }).ToArray();
        return (x, x.Select(r => new[] { r[0] * 2 }).ToArray());
    }

    private static NeuralNetwork Model()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed: 1);
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        model.Train.TestAfterTraining = false;
        return model;
    }

    [Fact]
    public void Core_library_does_not_depend_on_ScottPlot()
    {
        var references = typeof(NeuralNetwork).Assembly.GetReferencedAssemblies().Select(a => a.Name);
        Assert.DoesNotContain(references, name => name!.StartsWith("ScottPlot"));
    }

    [Fact]
    public void Loss_chart_hook_gets_the_history_at_the_end_of_TrainLoop()
    {
        var (x, y) = Data();
        var model = Model();
        int calls = 0; int[]? epochs = null; double[]? train = null; double[]? valid = null;
        model.Train.LossChart = (e, t, v) => { calls++; epochs = e; train = t; valid = v; };

        model.Train.TrainLoop(x, y, 4, 2);

        Assert.Equal(1, calls);
        Assert.NotEmpty(epochs!);
        Assert.Equal(epochs!.Length, train!.Length);
        Assert.Equal(epochs.Length, valid!.Length);
    }

    [Fact]
    public void Loss_chart_is_not_drawn_when_switched_off()
    {
        var (x, y) = Data();
        var model = Model();
        model.Train.ShowLossChart = false;
        bool called = false;
        model.Train.LossChart = (_, _, _) => called = true;

        model.Train.TrainLoop(x, y, 2, 2);

        Assert.False(called);
    }

    [Fact]
    public void Plotting_package_plugs_in_its_graph()
    {
        var model = Model();

        model.Train.UseLossChart();

        Assert.NotNull(model.Train.LossChart);
        Assert.Equal(nameof(GraphPlotter.SaveLossChart), model.Train.LossChart!.Method.Name);
    }
}

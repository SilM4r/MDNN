using System;
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
/// What `epochs` means: the NUMBER of epochs in this run, not the model's target epoch.
///
/// Previously it was stored directly in `_totalEpoch` and the loop ran `for (epoch = _epoch; epoch &lt; _totalEpoch)`.
/// On a fresh model that made no difference (`_epoch == 0`), on a used one it did: a second
/// `TrainLoop(5)` call on a model with 5 epochs behind it **silently did nothing**, yet the method
/// still finished, printed the final summary and drew the chart as if training had happened.
/// The README has always described it as a count ("number of training epochs").
/// </summary>
public class EpochSemanticsTests
{
    private static NeuralNetwork Quiet(int? seed = 7)
    {
        var m = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed);
        m.Train.ShowLossChart = false;
        m.Train.ShowModelInfo = false;
        m.Train.AutoSave = false;
        m.Train.TestAfterTraining = false;
        return m;
    }

    private static (double[][] x, double[][] y) Data(int n = 8)
    {
        var x = Enumerable.Range(1, n).Select(i => new[] { i / 10.0 }).ToArray();
        var y = x.Select(r => new[] { r[0] * 2 }).ToArray();
        return (x, y);
    }

    private static double Weight(NeuralNetwork m) => ((Dense)m.Layers.Layers[0]).Neurons[0].Weights[0];

    // ------------------------------------------------------------ a count, not a target

    [Fact]
    public void A_second_run_trains_the_requested_number_of_epochs_again()
    {
        var (x, y) = Data();
        var model = Quiet();

        model.Train.TrainLoop(x, y, 5, 2);
        Assert.Equal(5u, model.Train.CurrentEpoch);
        double afterFirst = Weight(model);

        model.Train.TrainLoop(5, 2);

        Assert.Equal(10u, model.Train.CurrentEpoch);
        Assert.NotEqual(afterFirst, Weight(model));
    }

    [Fact]
    public void Total_epoch_marks_the_end_of_the_current_run()
    {
        var (x, y) = Data();
        var model = Quiet();

        model.Train.TrainLoop(x, y, 5, 2);
        Assert.Equal(5u, model.Train.TotalEpoch);

        model.Train.TrainLoop(3, 2);

        // 5 done + 3 new → both the "Epoch: 8 / 8" output and Target_epoch in the saved file are right
        Assert.Equal(8u, model.Train.TotalEpoch);
        Assert.Equal(8u, model.Train.CurrentEpoch);
    }

    [Fact]
    public void Optimizer_steps_keep_adding_up_across_runs()
    {
        var (x, y) = Data();
        var model = Quiet();

        model.Train.TrainLoop(x, y, 5, 2);
        ulong afterFirst = model.Train.OptimizerSteps;

        model.Train.TrainLoop(5, 2);

        Assert.Equal(afterFirst * 2, model.Train.OptimizerSteps);
    }

    [Fact]
    public void Zero_epochs_still_means_train_nothing()
    {
        // An idiom in use: "split the dataset but don't train" (reference model in tests).
        var (x, y) = Data();
        var model = Quiet();

        model.Train.TrainLoop(x, y, 0);

        Assert.Equal(0u, model.Train.CurrentEpoch);
        Assert.Equal(0ul, model.Train.OptimizerSteps);
        Assert.NotNull(model.Train.TrainInputs);   // the dataset was still split
    }

    [Fact]
    public void SimpleTrainLoop_counts_epochs_too()
    {
        var (x, y) = Data();
        var model = Quiet();

        model.Train.SimpleTrainLoop(x, y, 4, 2);
        Assert.Equal(4u, model.Train.CurrentEpoch);
        double afterFirst = Weight(model);

        model.Train.SimpleTrainLoop(x, y, 4, 2);

        Assert.Equal(8u, model.Train.CurrentEpoch);
        Assert.NotEqual(afterFirst, Weight(model));
    }

    [Fact]
    public void A_loaded_model_trains_the_number_of_epochs_it_is_asked_for()
    {
        var (x, y) = Data();
        string path = Path.Combine(Path.GetTempPath(), $"mdnn_epochs_{Guid.NewGuid():N}");

        try
        {
            var first = Quiet();
            first.Train.TrainLoop(x, y, 5, 2);
            first.SaveAsJson(path);

            var loaded = NeuralNetwork.LoadModel(path);
            loaded.Train.ShowLossChart = false;
            loaded.Train.ShowModelInfo = false;
            loaded.Train.AutoSave = false;
            loaded.Train.TestAfterTraining = false;

            Assert.Equal(5u, loaded.Train.CurrentEpoch);

            // Previously: 3 < 5, so the loop never ran and the model was not trained further.
            loaded.Train.TrainLoop(x, y, 3, 2);

            Assert.Equal(8u, loaded.Train.CurrentEpoch);
        }
        finally
        {
            string file = path + ".json";
            if (File.Exists(file)) File.Delete(file);
        }
    }

    // ------------------------------------------------------------ report interval

    [Fact]
    public void Report_interval_is_derived_from_the_length_of_this_run()
    {
        var (x, y) = Data(20);
        var model = Quiet();
        model.Train.ConsoleReportCount = 2;   // want 2 reports per run

        model.Train.TrainLoop(x, y, 10, 5);
        Assert.Equal(5u, model.Train.ReportInterval);   // 10 epochs / 2 reports

        // The second run is again 10 epochs long, so the interval must stay 5.
        // Computed from the absolute target epoch (20) it would be 10 → only one report.
        model.Train.TrainLoop(10, 5);
        Assert.Equal(5u, model.Train.ReportInterval);
    }

    [Fact]
    public void More_requested_reports_than_epochs_still_reports_every_epoch()
    {
        var (x, y) = Data();
        var model = Quiet();
        model.Train.ConsoleReportCount = 100;

        model.Train.TrainLoop(x, y, 3, 2);

        Assert.Equal(1u, model.Train.ReportInterval);
        Assert.Equal(100u, model.Train.ConsoleReportCount);   // public setting left untouched
    }
}

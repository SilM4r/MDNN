using System;
using System.IO;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// Co znamená `numberOfEpoch`: POČET epoch tohohle běhu, ne cílová epocha modelu.
///
/// Dřív se ukládalo přímo do `_totalEpoch` a smyčka jela `for (epoch = _epoch; epoch &lt; _totalEpoch)`.
/// Na čerstvém modelu to vycházelo nastejno (`_epoch == 0`), na použitém ne: druhé volání
/// `TrainLoop(5)` na modelu, který má za sebou 5 epoch, **tiše neudělalo nic** — a metoda
/// přesto doběhla, vypsala závěrečný přehled a nakreslila graf, jako by trénink proběhl.
/// README to přitom celou dobu popisuje jako počet („number of training epochs").
/// </summary>
public class EpochSemanticsTests
{
    private static My_DNN.MDNN Quiet(int? seed = 7)
    {
        var m = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed);
        m.Train.ShowLossChartInTrainLoop = false;
        m.Train.ShowModelInfoIntrainLoop = false;
        m.Train.AutoSaveInTrainLoop = false;
        m.Train.TestNeuralNetworkAfterTraining = false;
        return m;
    }

    private static (double[][] x, double[][] y) Data(int n = 8)
    {
        var x = Enumerable.Range(1, n).Select(i => new[] { i / 10.0 }).ToArray();
        var y = x.Select(r => new[] { r[0] * 2 }).ToArray();
        return (x, y);
    }

    private static double Weight(My_DNN.MDNN m) => ((Dense)m.Layers.Layers[0]).Neurons[0].Weights[0];

    // ------------------------------------------------------------ počet, ne cíl

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

        // 5 hotových + 3 nové → výpis „Epoch: 8 / 8" i Target_epoch v uloženém souboru sedí
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
        // Použitý idiom: „rozděl mi dataset, ale netrénuj" (referenční model v testech).
        var (x, y) = Data();
        var model = Quiet();

        model.Train.TrainLoop(x, y, 0);

        Assert.Equal(0u, model.Train.CurrentEpoch);
        Assert.Equal(0ul, model.Train.OptimizerSteps);
        Assert.NotNull(model.Train.TrainDataInputs);   // dataset se přesto rozdělil
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

            var loaded = My_DNN.MDNN.LoadModel(path);
            loaded.Train.ShowLossChartInTrainLoop = false;
            loaded.Train.ShowModelInfoIntrainLoop = false;
            loaded.Train.AutoSaveInTrainLoop = false;
            loaded.Train.TestNeuralNetworkAfterTraining = false;

            Assert.Equal(5u, loaded.Train.CurrentEpoch);

            // Dřív: 3 < 5, takže se smyčka vůbec neprotočila a model se nedotrénoval.
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
        model.Train.NumberOfShowEpochInConsole = 2;   // chci 2 reporty za běh

        model.Train.TrainLoop(x, y, 10, 5);
        Assert.Equal(5u, model.Train.ReportInterval);   // 10 epoch / 2 reporty

        // Druhý běh je zase 10 epoch dlouhý, takže interval musí zůstat 5.
        // Kdyby se počítal z absolutní cílové epochy (20), vyšlo by 10 → jen jeden report.
        model.Train.TrainLoop(10, 5);
        Assert.Equal(5u, model.Train.ReportInterval);
    }

    [Fact]
    public void More_requested_reports_than_epochs_still_reports_every_epoch()
    {
        var (x, y) = Data();
        var model = Quiet();
        model.Train.NumberOfShowEpochInConsole = 100;

        model.Train.TrainLoop(x, y, 3, 2);

        Assert.Equal(1u, model.Train.ReportInterval);
        Assert.Equal(100u, model.Train.NumberOfShowEpochInConsole);   // veřejné nastavení nedotčené
    }
}

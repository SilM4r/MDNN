using System;
using System.Collections.Generic;
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
/// Data-paralelní trénink (`Train.Workers` &gt; 1): každé vlákno má repliku modelu, spočítá
/// svůj kus dávky a gradienty se sečtou do originálu.
///
/// Hlavní kontrakt: paralelní běh musí dát TOTÉŽ co sekvenční — až na zaokrouhlení, protože
/// sčítání double není asociativní a gradient dávky se sčítá v jiném pořadí. Kdyby si
/// repliky sdílely nějaký pracovní buffer, rozdíl by nebyl v posledních cifrách, ale
/// v celých číslech (dvě vlákna by si přepisovala aktivace), takže tolerance 1e-9 tu
/// slouží jako ostrá hranice mezi „jiné pořadí sčítání" a „chyba".
/// </summary>
public class DataParallelTests
{
    private static T Quiet<T>(T model) where T : My_DNN.MDNN
    {
        model.Train.ShowLossChartInTrainLoop = false;
        model.Train.ShowModelInfoIntrainLoop = false;
        model.Train.AutoSaveInTrainLoop = false;
        model.Train.TestNeuralNetworkAfterTraining = false;
        return model;
    }

    /// <summary>Všechny parametry modelu za sebou: váhy a biasy neuronů, kernely a biasy Conv.</summary>
    private static double[] Parameters(My_DNN.MDNN model)
    {
        var all = new List<double>();
        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is LayerBasedOnNeurons byNeurons)
            {
                foreach (Neuron n in byNeurons.Neurons)
                {
                    all.AddRange(n.Weights);
                    all.Add(n.Bias);
                }
            }
            else if (layer is Conv conv)
            {
                foreach (var f in conv.Kernel)
                foreach (var row in f)
                foreach (var col in row)
                    all.AddRange(col);
                all.AddRange(conv.Biases);
            }
        }
        return all.ToArray();
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double scale = Math.Max(1.0, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / scale);
        }
        Assert.True(worst < tolerance, $"největší relativní rozdíl {worst:G3} (tolerance {tolerance:G3})");
    }

    private static (double[][] x, double[][] y) Regression(int n = 10)
    {
        var x = new double[n][];
        var y = new double[n][];
        for (int i = 0; i < n; i++)
        {
            double v = i * 0.2 - 1.0;
            x[i] = new[] { v, v * v, Math.Cos(v) };
            y[i] = new[] { Math.Sin(v), v * 0.5 };
        }
        return (x, y);
    }

    private static My_DNN.MDNN Mlp(Optimizer optimizer, int seed = 5)
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), optimizer, new MSE(), seed);
        model.Layers.Add(new Dense(6, new Tanh()));
        return Quiet(model);
    }

    // ------------------------------------------------------------ nastavení

    [Fact]
    public void Workers_default_to_one_and_reject_zero()
    {
        var model = Mlp(new SGD(0.1));

        Assert.Equal(1, model.Train.Workers);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Train.Workers = 0);
    }

    // ------------------------------------------------------------ paralelně = sekvenčně

    [Theory]
    [InlineData(2, 5)]    // dávka se dělí rovnoměrně jen náhodou: 5 vzorků na 2 vlákna = 2 + 3
    [InlineData(3, 5)]    // 1 + 2 + 2
    [InlineData(4, 2)]    // víc vláken než vzorků v dávce → použijí se jen 2
    public void Dense_training_matches_sequential(int workers, int batch)
    {
        var (x, y) = Regression();
        var sequential = Mlp(new Adam(0.05));
        var parallel = Mlp(new Adam(0.05));
        parallel.Train.Workers = workers;

        sequential.Train.SimpleTrainLoop(x, y, 4, (uint)batch);
        parallel.Train.SimpleTrainLoop(x, y, 4, (uint)batch);

        AssertClose(Parameters(sequential), Parameters(parallel));
        Assert.Equal(sequential.Train.OptimizerSteps, parallel.Train.OptimizerSteps);
    }

    [Fact]
    public void Reported_loss_matches_sequential()
    {
        var (x, y) = Regression();
        var sequential = Mlp(new SGD(0.1));
        var parallel = Mlp(new SGD(0.1));
        parallel.Train.Workers = 3;

        // Report po epoše akumulátor vynuluje. S jedním reportem na dva epochy proběhne
        // jen po první, takže po druhé epoše zůstane její loss k porovnání.
        sequential.Train.NumberOfShowEpochInConsole = 1;
        parallel.Train.NumberOfShowEpochInConsole = 1;

        sequential.Train.SimpleTrainLoop(x, y, 2, 5);
        parallel.Train.SimpleTrainLoop(x, y, 2, 5);

        // Průměr loss se počítá z akumulátorů replik — kdyby se nesečetl počet vzorků,
        // vyšel by průměr jen z jedné repliky.
        double expected = sequential.Context.Loss.GetAverageLossPerIteration();
        double actual = parallel.Context.Loss.GetAverageLossPerIteration();
        Assert.True(expected > 0);
        Assert.Equal(expected, actual, 12);
    }

    // [vzorek, výška, šířka, kanály] — dva kanály, ať se replikují i vícekanálové kernely.
    private static double[,,,] Images(int n)
    {
        var img = new double[n, 6, 6, 2];
        for (int s = 0; s < n; s++)
        for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
        for (int c = 0; c < 2; c++)
            img[s, i, j, c] = Math.Sin(s + i * 0.7 + j * 0.3 + c);
        return img;
    }

    private static double[,] Labels(int n)
    {
        var y = new double[n, 2];
        for (int s = 0; s < n; s++) y[s, s % 2] = 1;
        return y;
    }

    private static My_DNN.MDNN Cnn()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 3);
        model.Layers.Add(new Conv(3, 3, new ReLu(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.Layers.Add(new Dense(4, new Tanh()));
        return Quiet(model);
    }

    [Fact]
    public void Cnn_training_matches_sequential()
    {
        var sequential = Cnn();
        var parallel = Cnn();
        parallel.Train.Workers = 3;

        sequential.Train.TrainLoop(Images(12), Labels(12), 3, 4);
        parallel.Train.TrainLoop(Images(12), Labels(12), 3, 4);

        AssertClose(Parameters(sequential), Parameters(parallel));
    }

    private static My_DNN.MDNN Rnn()
    {
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed: 9);
        model.Layers.Add(new RNN(4, new Tanh()));
        return Quiet(model);
    }

    [Fact]
    public void Rnn_sequence_training_matches_sequential()
    {
        // 10 sekvencí × 5 kroků × 2 rysy → 1 výstup na krok. Replika bere CELOU sekvenci,
        // takže skrytý stav a RTRL citlivosti se mezi sekvencemi nemíchají.
        var x = new double[10, 5, 2];
        var y = new double[10, 5, 1];
        for (int s = 0; s < 10; s++)
        for (int t = 0; t < 5; t++)
        {
            x[s, t, 0] = Math.Sin(s + t * 0.4);
            x[s, t, 1] = Math.Cos(s * 0.3 - t);
            y[s, t, 0] = 0.5 * x[s, t, 0] - 0.2 * x[s, t, 1];
        }

        var sequential = Rnn();
        var parallel = Rnn();
        parallel.Train.Workers = 3;

        sequential.Train.TrainLoop(x, y, 3, 4, isSequence: true);
        parallel.Train.TrainLoop(x, y, 3, 4, isSequence: true);

        AssertClose(Parameters(sequential), Parameters(parallel));
    }

    [Fact]
    public void Layers_added_between_runs_are_picked_up()
    {
        // Repliky se staví každou epochu znovu. Kdyby se držely déle, po Add by ukazovaly
        // na váhy, které model už nepoužívá.
        var (x, y) = Regression();
        var sequential = Mlp(new SGD(0.1));
        var parallel = Mlp(new SGD(0.1));
        parallel.Train.Workers = 2;

        foreach (var model in new[] { sequential, parallel })
        {
            model.Train.SimpleTrainLoop(x, y, 2, 4);
            model.Layers.Add(new Dense(3, new Tanh()));
            model.Train.SimpleTrainLoop(x, y, 2, 4);
        }

        AssertClose(Parameters(sequential), Parameters(parallel));
    }

    // ------------------------------------------------------------ determinismus

    private static double[] TrainWithDropout(int workers)
    {
        var (x, y) = Regression(12);
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
        model.Layers.Add(new Dense(8, new ReLu(), dropout: 0.3));
        Quiet(model).Train.Workers = workers;

        model.Train.SimpleTrainLoop(x, y, 5, 6);
        return Parameters(model);
    }

    [Fact]
    public void Same_seed_and_workers_give_bit_identical_result_even_with_dropout()
    {
        // Každá replika má vlastní generátor odvozený ze seedu modelu, takže masky
        // nezávisí na tom, které vlákno doběhne dřív.
        Assert.Equal(TrainWithDropout(3), TrainWithDropout(3));
    }

    [Fact]
    public void Dropout_in_replicas_actually_drops()
    {
        // Kdyby replika dropout ignorovala, paralelní běh s dropoutem by se rovnal
        // paralelnímu běhu bez něj.
        var (x, y) = Regression(12);
        double[] Run(double dropout)
        {
            var model = new My_DNN.MDNN(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
            model.Layers.Add(new Dense(8, new ReLu(), dropout: dropout));
            Quiet(model).Train.Workers = 3;
            model.Train.SimpleTrainLoop(x, y, 3, 6);
            return Parameters(model);
        }

        Assert.NotEqual(Run(0.0), Run(0.3));
    }

    [Fact]
    public void Replica_context_carries_the_training_settings()
    {
        // Přímá kontrola: test níž (Global_dropout_reaches_the_replicas) by prošel i tehdy,
        // kdyby replika globální dropout ignorovala — originál s dropoutem čerpá seedy
        // replik ze svého generátoru, takže se změní pořadí dat a výsledek se liší i tak.
        var original = new NetworkContext { Loss = new CrossEntropy(), L2 = 0.01, Dropout = 0.4, SequenceTrain = true };
        original.Loss.CalculateLoss(new[] { 0.5 }, new[] { 1.0 });

        NetworkContext replica = original.CreateReplica(randomSeed: 7);

        Assert.Equal(0.4, replica.Dropout);
        Assert.Equal(0.01, replica.L2);
        Assert.True(replica.SequenceTrain);
        Assert.IsType<CrossEntropy>(replica.Loss);
        Assert.NotSame(original.Loss, replica.Loss);
        Assert.Equal(0, replica.Loss.GetAverageLossPerIteration());   // vlastní, prázdný akumulátor
        Assert.Equal(new Random(7).Next(), replica.Random.Next());
    }

    [Fact]
    public void Global_dropout_reaches_the_replicas()
    {
        // Globální Context.Dropout je jiná cesta než dropout zadaný na vrstvě: replika ho
        // musí převzít v kontextu, ne s vrstvou.
        var (x, y) = Regression(12);
        double[] Run(double dropout)
        {
            var model = new My_DNN.MDNN(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
            model.Layers.Add(new Dense(8, new ReLu()));
            model.Context.Dropout = dropout;
            Quiet(model).Train.Workers = 3;
            model.Train.SimpleTrainLoop(x, y, 3, 6);
            return Parameters(model);
        }

        Assert.NotEqual(Run(0.0), Run(0.3));
    }

    // ------------------------------------------------------------ chyby

    /// <summary>Vlastní vrstva (identita) — README je povoluje, repliky ale neumí.</summary>
    private sealed class Identity : LayerWithUntrainedParameters
    {
        private int[] shape = { 0 };
        private Tensor? last;
        public override string Name => "Identity";
        public override int[] Input_size_and_shape => shape;
        public override int[] Output_size_and_shape => shape;
        public override Tensor? Layer_output => last;
        public override Tensor? Layer_raw_output => last;
        public override Tensor FeedForward(Tensor input_values) => last = input_values;
        public override void WireShapes(int? number_of_elements = null, int[]? number_of_input = null)
        {
            if (number_of_input != null) shape = number_of_input;
        }
        public override void MaterializeParameters() { }
        public override bool IsMaterialized => true;
        public override Tensor Backward(Tensor gradOutput, bool preActivation = false, bool computeInputGradient = true)
            => gradOutput;
    }

    [Fact]
    public void Custom_layer_without_replica_support_is_rejected()
    {
        var (x, y) = Regression();
        var model = Mlp(new SGD(0.1));
        model.Layers.Insert(0, new Identity());

        model.Train.SimpleTrainLoop(x, y, 1, 4);   // sekvenčně to jde
        model.Train.Workers = 2;

        var ex = Assert.Throws<NotSupportedException>(() => model.Train.SimpleTrainLoop(x, y, 1, 4));
        Assert.Contains("Identity", ex.Message);
    }

    [Fact]
    public void Error_in_a_worker_surfaces_as_the_original_exception()
    {
        // Sekvenčně dá nesedící počet cílů ArgumentException. Paralelně musí přijít
        // TATÁŽ výjimka, ne AggregateException z Parallel.For.
        var (x, _) = Regression();
        var wrongTargets = x.Select(_ => new[] { 1.0, 2.0, 3.0 }).ToArray();   // model má 2 výstupy
        var model = Mlp(new SGD(0.1));
        model.Train.Workers = 3;

        Assert.Throws<ArgumentException>(() => model.Train.SimpleTrainLoop(x, wrongTargets, 1, 5));
    }

    [Fact]
    public void Rnn_outside_sequence_training_is_rejected()
    {
        // Mimo sekvenční režim teče skrytý stav RNN z jednoho vzorku do dalšího v pořadí
        // dávky. Paralelně by si ho každá replika nesla jinak — tiše jiný výsledek.
        var (x, y) = Regression();
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 2);
        model.Layers.Add(new RNN(3, new Tanh()));
        Quiet(model).Train.Workers = 2;

        Assert.Throws<NotSupportedException>(() => model.Train.SimpleTrainLoop(x, y, 1, 4));
    }

    [Fact]
    public void Gpu_with_workers_is_rejected()
    {
        var (x, y) = Regression();
        var model = Mlp(new SGD(0.1));
        model.Context.CalculationViaGpu = true;
        model.Train.Workers = 2;

        Assert.Throws<NotSupportedException>(() => model.Train.SimpleTrainLoop(x, y, 1, 4));
    }
}

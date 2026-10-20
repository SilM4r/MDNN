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
/// Data-parallel training (`Train.Workers` &gt; 1): each thread has a replica of the model,
/// computes its share of the batch, and the gradients are summed into the original.
///
/// Main contract: a parallel run must give the SAME result as a sequential one — up to
/// rounding, because double addition is not associative and the batch gradient is summed in
/// a different order. If the replicas shared some working buffer, the difference would not be
/// in the last digits but in whole numbers (two threads overwriting each other's activations),
/// so the 1e-9 tolerance serves as a sharp line between "different summation order" and "bug".
/// </summary>
public class DataParallelTests
{
    private static T Quiet<T>(T model) where T : NeuralNetwork
    {
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        model.Train.TestAfterTraining = false;
        return model;
    }

    /// <summary>All model parameters in sequence: neuron weights and biases, Conv kernels and biases.</summary>
    private static double[] Parameters(NeuralNetwork model)
    {
        var all = new List<double>();
        foreach (Layer layer in model.Layers.Layers)
        {
            if (layer is NeuronLayer byNeurons)
            {
                foreach (Neuron n in byNeurons.Neurons)
                {
                    all.AddRange(n.Weights);
                    all.Add(n.Bias);
                }
            }
            else if (layer is Conv conv)
            {
                foreach (var f in conv.Kernels)
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
        Assert.True(worst < tolerance, $"largest relative difference {worst:G3} (tolerance {tolerance:G3})");
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

    private static NeuralNetwork Mlp(Optimizer optimizer, int seed = 5)
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), optimizer, new MSE(), seed);
        model.Layers.Add(new Dense(6, new Tanh()));
        return Quiet(model);
    }

    // ------------------------------------------------------------ settings

    [Fact]
    public void Workers_default_to_one_and_reject_zero()
    {
        var model = Mlp(new SGD(0.1));

        Assert.Equal(1, model.Train.Workers);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Train.Workers = 0);
    }

    // ------------------------------------------------------------ parallel = sequential

    [Theory]
    [InlineData(2, 5)]    // the batch does not split evenly: 5 samples over 2 threads = 2 + 3
    [InlineData(3, 5)]    // 1 + 2 + 2
    [InlineData(4, 2)]    // more threads than samples in the batch → only 2 are used
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

        // A report after an epoch resets the accumulator. With one report for two epochs it runs
        // only after the first, so the second epoch's loss remains to be compared.
        sequential.Train.ConsoleReportCount = 1;
        parallel.Train.ConsoleReportCount = 1;

        sequential.Train.SimpleTrainLoop(x, y, 2, 5);
        parallel.Train.SimpleTrainLoop(x, y, 2, 5);

        // The average loss is computed from the replicas' accumulators — if the sample counts
        // were not summed, the average would come from a single replica only.
        double expected = sequential.Context.Loss.GetAverageLossPerIteration();
        double actual = parallel.Context.Loss.GetAverageLossPerIteration();
        Assert.True(expected > 0);
        Assert.Equal(expected, actual, 12);
    }

    // [sample, height, width, channels] — two channels so multi-channel kernels get replicated too.
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

    private static NeuralNetwork Cnn()
    {
        var model = new NeuralNetwork(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 3);
        model.Layers.Add(new Conv(3, 3, new ReLU(), "valid"));
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

    private static NeuralNetwork Rnn()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE(), seed: 9);
        model.Layers.Add(new RNN(4, new Tanh()));
        return Quiet(model);
    }

    [Fact]
    public void Rnn_sequence_training_matches_sequential()
    {
        // 10 sequences × 5 steps × 2 features → 1 output per step. A replica takes the WHOLE
        // sequence, so the hidden state and RTRL sensitivities do not mix between sequences.
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
        // Replicas are rebuilt every epoch. If they were kept longer, after Add they would
        // point to weights the model no longer uses.
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

    // ------------------------------------------------------------ evaluation (valid, test)
    //
    // Validation and test are forward-only; with Workers > 1 only the outputs are computed
    // in parallel — loss and score are summed from them sequentially in sample order. The
    // result must therefore be bit-identical to Workers = 1, not just "close".

    private static NeuralNetwork Classifier()
    {
        var model = new NeuralNetwork(new Dense(3, new Softmax()), new Adam(0.02), new CrossEntropy(), seed: 13);
        model.Layers.Add(new Dense(6, new Tanh()));
        Quiet(model).Train.ConsoleReportCount = 100;   // report every epoch → validation every time
        return model;
    }

    private static (double[][] x, double[][] y) Classes(int n = 30)
    {
        var x = new double[n][];
        var y = new double[n][];
        for (int i = 0; i < n; i++)
        {
            double a = Math.Sin(i * 1.7), b = Math.Cos(i * 0.9);
            x[i] = new[] { a, b };
            y[i] = new double[3];
            y[i][a > 0.3 ? 0 : b > 0 ? 1 : 2] = 1;
        }
        return (x, y);
    }

    [Fact]
    public void Validation_is_bit_identical_to_sequential()
    {
        // With batch size 1 even training is bit-identical (one replica computes the sample and
        // 0 + x is exact), so we can compare EXACTLY: if parallel validation gave a different
        // valid loss, a different best model would be picked and the weights would differ too.
        var (x, y) = Classes();
        var sequential = Classifier();
        var parallel = Classifier();
        parallel.Train.Workers = 3;

        sequential.Train.TrainLoop(x, y, 8, 1);
        parallel.Train.TrainLoop(x, y, 8, 1);

        Assert.Equal(sequential.Train.LowestValidLoss, parallel.Train.LowestValidLoss);
        Assert.Equal(sequential.Train.BestEpoch, parallel.Train.BestEpoch);
        Assert.Equal(Parameters(sequential), Parameters(parallel));
    }

    [Fact]
    public void Test_scores_are_identical_to_sequential()
    {
        var (x, y) = Classes();
        var model = Classifier();
        model.Train.SimpleTrainLoop(x, y, 5, 4);

        Tensor inputs = new Tensor(x);
        Tensor targets = new Tensor(y);

        var expected = model.Train.Evaluate(inputs, targets, showInConsole: false);
        model.Train.Workers = 4;
        var actual = model.Train.Evaluate(inputs, targets, showInConsole: false);

        Assert.Equal(expected, actual);
        Assert.Equal(30, actual.Item2);
    }

    [Fact]
    public void Sequence_test_scores_are_identical_to_sequential()
    {
        // A whole sequence goes through one replica and the hidden state is reset at its start.
        var x = new double[9, 4, 2];
        var y = new double[9, 4, 1];
        for (int s = 0; s < 9; s++)
        for (int t = 0; t < 4; t++)
        {
            x[s, t, 0] = Math.Sin(s + t);
            x[s, t, 1] = Math.Cos(s - t);
            y[s, t, 0] = x[s, t, 0] > 0 ? 1 : 0;
        }

        var model = Rnn();
        model.Train.TrainLoop(x, y, 2, 1, isSequence: true);

        Tensor inputs = new Tensor(x);
        Tensor targets = new Tensor(y);

        var expected = model.Train.Evaluate(inputs, targets, showInConsole: false);
        model.Train.Workers = 3;
        var actual = model.Train.Evaluate(inputs, targets, showInConsole: false);

        Assert.Equal(expected, actual);
        Assert.Equal(9 * 4, actual.Item2);
    }

    [Fact]
    public void Evaluation_does_not_draw_from_the_models_random_generator()
    {
        // Training replicas draw their seeds from the model's generator when dropout is on.
        // Evaluation replicas must not: validation runs between epochs and an advanced generator
        // would shuffle the next epoch differently than with Workers = 1.
        var (x, y) = Classes();
        NeuralNetwork WithDropout()
        {
            var m = new NeuralNetwork(new Dense(3, new Softmax()), new Adam(0.02), new CrossEntropy(), seed: 13);
            m.Layers.Add(new Dense(6, new Tanh(), dropout: 0.3));
            m.GetResults(new Tensor(x[0]));
            return m;
        }

        var untouched = WithDropout();
        var evaluated = WithDropout();
        evaluated.Train.Workers = 3;
        evaluated.Train.Evaluate(new Tensor(x), new Tensor(y), showInConsole: false);

        Assert.Equal(untouched.Context.Random.Next(), evaluated.Context.Random.Next());
    }

    // ------------------------------------------------------------ determinism

    private static double[] TrainWithDropout(int workers)
    {
        var (x, y) = Regression(12);
        var model = new NeuralNetwork(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
        model.Layers.Add(new Dense(8, new ReLU(), dropout: 0.3));
        Quiet(model).Train.Workers = workers;

        model.Train.SimpleTrainLoop(x, y, 5, 6);
        return Parameters(model);
    }

    [Fact]
    public void Same_seed_and_workers_give_bit_identical_result_even_with_dropout()
    {
        // Each replica has its own generator derived from the model's seed, so the masks
        // do not depend on which thread finishes first.
        Assert.Equal(TrainWithDropout(3), TrainWithDropout(3));
    }

    [Fact]
    public void Dropout_in_replicas_actually_drops()
    {
        // If a replica ignored dropout, a parallel run with dropout would equal
        // a parallel run without it.
        var (x, y) = Regression(12);
        double[] Run(double dropout)
        {
            var model = new NeuralNetwork(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
            model.Layers.Add(new Dense(8, new ReLU(), dropout: dropout));
            Quiet(model).Train.Workers = 3;
            model.Train.SimpleTrainLoop(x, y, 3, 6);
            return Parameters(model);
        }

        Assert.NotEqual(Run(0.0), Run(0.3));
    }

    [Fact]
    public void Replica_context_carries_the_training_settings()
    {
        // Direct check: the test below (Global_dropout_reaches_the_replicas) would pass even
        // if a replica ignored global dropout — the original with dropout draws replica seeds
        // from its generator, so the data order changes and the result differs anyway.
        var original = new NetworkContext { Loss = new CrossEntropy(), L2 = 0.01, Dropout = 0.4, SequenceTrain = true };
        original.Loss.CalculateLoss(new[] { 0.5 }, new[] { 1.0 });

        NetworkContext replica = original.CreateReplica(randomSeed: 7);

        Assert.Equal(0.4, replica.Dropout);
        Assert.Equal(0.01, replica.L2);
        Assert.True(replica.SequenceTrain);
        Assert.IsType<CrossEntropy>(replica.Loss);
        Assert.NotSame(original.Loss, replica.Loss);
        Assert.Equal(0, replica.Loss.GetAverageLossPerIteration());   // its own, empty accumulator
        Assert.Equal(new Random(7).Next(), replica.Random.Next());
    }

    [Fact]
    public void Global_dropout_reaches_the_replicas()
    {
        // Global Context.Dropout is a different path than dropout set on a layer: the replica
        // must take it over via the context, not with the layer.
        var (x, y) = Regression(12);
        double[] Run(double dropout)
        {
            var model = new NeuralNetwork(new Dense(2, new Linear()), new Adam(0.05), new MSE(), seed: 21);
            model.Layers.Add(new Dense(8, new ReLU()));
            model.Context.Dropout = dropout;
            Quiet(model).Train.Workers = 3;
            model.Train.SimpleTrainLoop(x, y, 3, 6);
            return Parameters(model);
        }

        Assert.NotEqual(Run(0.0), Run(0.3));
    }

    // ------------------------------------------------------------ errors

    /// <summary>A custom layer (identity) — the README allows them, but replicas don't support them.</summary>
    private sealed class Identity : ParameterlessLayer
    {
        private int[] shape = { 0 };
        private Tensor? last;
        public override string Name => "Identity";
        public override int[] InputShape => shape;
        public override int[] OutputShape => shape;
        public override Tensor? Output => last;
        public override Tensor? RawOutput => last;
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

        model.Train.SimpleTrainLoop(x, y, 1, 4);   // works sequentially
        model.Train.Workers = 2;

        var ex = Assert.Throws<NotSupportedException>(() => model.Train.SimpleTrainLoop(x, y, 1, 4));
        Assert.Contains("Identity", ex.Message);
    }

    [Fact]
    public void Error_in_a_worker_surfaces_as_the_original_exception()
    {
        // Sequentially, a mismatched target count gives ArgumentException. In parallel the
        // SAME exception must surface, not an AggregateException from Parallel.For.
        var (x, _) = Regression();
        var wrongTargets = x.Select(_ => new[] { 1.0, 2.0, 3.0 }).ToArray();   // the model has 2 outputs
        var model = Mlp(new SGD(0.1));
        model.Train.Workers = 3;

        Assert.Throws<ArgumentException>(() => model.Train.SimpleTrainLoop(x, wrongTargets, 1, 5));
    }

    [Fact]
    public void Rnn_outside_sequence_training_is_rejected()
    {
        // Outside sequence mode the RNN hidden state flows from one sample to the next in batch
        // order. In parallel each replica would carry it differently — a silently different result.
        var (x, y) = Regression();
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 2);
        model.Layers.Add(new RNN(3, new Tanh()));
        Quiet(model).Train.Workers = 2;

        Assert.Throws<NotSupportedException>(() => model.Train.SimpleTrainLoop(x, y, 1, 4));
    }
}

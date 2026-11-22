using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// Every dataset of a TrainLoop run — training, validation, test, and the data given to
/// Evaluate — is shaped the same way (sequences [n, steps] → [n, steps, 1], samples [n] → [n, 1]),
/// and the sequence mode belongs to the run.
///
/// Previously only the TRAINING data was reshaped, in place: an explicit validation set of
/// sequences [n, steps] crashed at the first validation, the reshape changed the shape of the
/// user's own tensor, and once a model had been trained on sequences it stayed in sequence mode.
/// </summary>
public class DatasetModeTests : GradientCheckTestBase
{
    private static NeuralNetwork Quiet(NeuralNetwork model, bool testAfterTraining = false)
    {
        model.Train.ShowLossChart = false;
        model.Train.ShowModelInfo = false;
        model.Train.AutoSave = false;
        model.Train.TestAfterTraining = testAfterTraining;
        model.Train.ConsoleReportCount = 1;
        return model;
    }

    private static void Silent(Action action)
    {
        var original = Console.Out;
        Console.SetOut(TextWriter.Null);
        try { action(); }
        finally { Console.SetOut(original); }
    }

    private static NeuralNetwork Recurrent(bool testAfterTraining = false)
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), seed: 1), testAfterTraining);
        model.Layers.Add(new RNN(3, new Tanh()));
        return model;
    }

    private static double[,] Sequences(int count, int steps, int seed)
    {
        var random = new Random(seed);
        var data = new double[count, steps];
        for (int i = 0; i < count; i++)
            for (int j = 0; j < steps; j++)
                data[i, j] = random.NextDouble();
        return data;
    }

    // ------------------------------------------------------------ explicit validation / test sets

    [Fact]
    public void Two_dimensional_sequences_train_with_an_explicit_validation_set()
    {
        var model = Recurrent();
        model.Train.SetDatasets(
            new LabeledData(Sequences(10, 5, 1), Sequences(10, 5, 2)),
            new LabeledData(Sequences(4, 5, 3), Sequences(4, 5, 4)));

        // previously: "The number of indices must be less than the number of tensor dimensions"
        // at the first validation — only the training set was reshaped to [n, steps, 1]
        Silent(() => model.Train.TrainLoop(2, 1, isSequence: true));

        Assert.Equal(new[] { 4, 5, 1 }, model.Train.ValidInputs!.Shape);
    }

    [Fact]
    public void Two_dimensional_sequences_are_tested_on_an_explicit_test_set()
    {
        var model = Recurrent(testAfterTraining: true);
        model.Train.SetDatasets(
            new LabeledData(Sequences(10, 5, 1), Sequences(10, 5, 2)),
            new LabeledData(Sequences(4, 5, 3), Sequences(4, 5, 4)),
            new LabeledData(Sequences(3, 5, 5), Sequences(3, 5, 6)));

        Silent(() => model.Train.TrainLoop(2, 1, isSequence: true));   // EvaluateAll over all three

        Assert.Equal(new[] { 3, 5, 1 }, model.Train.TestInputs!.Shape);
    }

    [Fact]
    public void One_dimensional_samples_train_with_an_explicit_validation_set()
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), seed: 1));
        double[] x = Enumerable.Range(0, 12).Select(i => i / 12.0).ToArray();
        double[] vx = { 0.05, 0.55, 0.95 };

        model.Train.SetDatasets(
            new LabeledData(x, x.Select(v => 2 * v).ToArray()),
            new LabeledData(vx, vx.Select(v => 2 * v).ToArray()));

        Silent(() => model.Train.TrainLoop(2, 4));

        Assert.Equal(new[] { 3, 1 }, model.Train.ValidInputs!.Shape);
    }

    [Fact]
    public void Two_dimensional_targets_of_sequences_with_features_are_shaped_too()
    {
        // inputs [n, steps, features], targets [n, steps]: previously the targets were reshaped only
        // when the INPUTS were 2D, so this crashed on GetSubTensor of the targets
        var random = new Random(7);
        var x = new double[8, 4, 2];
        for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 2; k++) x[i, j, k] = random.NextDouble();

        var model = Recurrent();
        Silent(() => model.Train.TrainLoop(x, Sequences(8, 4, 8), 2, 1, isSequence: true));
    }

    [Fact]
    public void Training_does_not_change_the_shape_of_the_users_tensors()
    {
        var trainInputs = new Tensor(Sequences(10, 5, 1));
        var trainTargets = new Tensor(Sequences(10, 5, 2));
        var model = Recurrent();
        model.Train.SetDatasets(
            new LabeledData(trainInputs, trainTargets),
            new LabeledData(Sequences(4, 5, 3), Sequences(4, 5, 4)));

        Silent(() => model.Train.TrainLoop(1, 1, isSequence: true));

        // previously: Reshape in place, the user's tensors came back as [10, 5, 1]
        Assert.Equal(new[] { 10, 5 }, trainInputs.Shape);
        Assert.Equal(new[] { 10, 5 }, trainTargets.Shape);
    }

    [Fact]
    public void Evaluate_takes_two_dimensional_sequences()
    {
        var model = Recurrent();
        model.Context.SequenceTrain = true;

        (int _, int total) = model.Train.Evaluate(new Tensor(Sequences(2, 3, 1)), new Tensor(Sequences(2, 3, 2)), showInConsole: false);

        Assert.Equal(6, total);   // 2 sequences × 3 steps
    }

    [Fact]
    public void One_target_per_sequence_is_reported_clearly()
    {
        var model = Recurrent();

        // previously: a bare ArgumentOutOfRangeException from a slice
        var ex = Assert.Throws<ArgumentException>(() =>
            Silent(() => model.Train.TrainLoop(Sequences(6, 4, 1), new double[6], 1, 1, isSequence: true)));
        Assert.Contains("one target per step", ex.Message);
    }

    [Fact]
    public void A_different_number_of_steps_in_inputs_and_targets_is_reported()
    {
        var model = Recurrent();

        var ex = Assert.Throws<ArgumentException>(() =>
            Silent(() => model.Train.TrainLoop(Sequences(6, 4, 1), Sequences(6, 3, 2), 1, 1, isSequence: true)));
        Assert.Contains("steps", ex.Message);
    }

    // ------------------------------------------------------------ the sequence mode belongs to the run

    [Fact]
    public void A_sequence_run_does_not_keep_the_model_in_sequence_mode()
    {
        var model = Quiet(new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.01), seed: 1));
        Silent(() => model.Train.TrainLoop(Sequences(12, 5, 1), Sequences(12, 5, 2), 1, 1, isSequence: true));
        Assert.True(model.Context.SequenceTrain);

        // the same input width (1), but ordinary samples now
        double[][] x = Enumerable.Range(0, 20).Select(i => new[] { i / 20.0 }).ToArray();
        double[][] y = x.Select(v => new[] { 2 * v[0] }).ToArray();

        // previously: SequenceTrain stayed true and the 20 samples were treated as 20 sequences
        // of one step each ([20, 1] → [20, 1, 1])
        Silent(() => model.Train.TrainLoop(x, y, 1, 1));

        Assert.False(model.Context.SequenceTrain);
        Assert.Equal(2, model.Train.ValidInputs!.Shape.Length);   // [samples, 1], not [sequences, steps, 1]
    }
}

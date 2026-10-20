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
/// Shape wiring: the first pass walks the chain left to right, each layer gets the previous
/// layer's output shape, and from then on the shape is FIXED.
///
/// These tests guard both halves of that sentence:
///   1. wiring must handle every input shape the forward pass handles
///      (previously it did not: `WireShapes` handled rank 1 and 3, `FeedForward` also rank 2)
///   2. after wiring, a different shape must FAIL, not be silently computed from what was
///      left in the buffer from the previous sample
///
/// Before the fix the second half showed like this: a layer wired for 4×4 received a 3×3
/// input of zeros and returned numbers from the previous sample. No exception.
/// </summary>
public class ShapeWiringTests
{
    private static NeuralNetwork ModelWith(params Layer[] hidden)
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1);
        foreach (Layer l in hidden) model.Layers.Add(l);
        return model;
    }

    private static double[,,] Filled(int h, int w, int c, double value)
    {
        var img = new double[h, w, c];
        for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        for (int k = 0; k < c; k++)
            img[i, j, k] = value;
        return img;
    }

    // ------------------------------------------------------------ 1) wiring the whole chain

    [Fact]
    public void First_forward_wires_the_whole_chain_left_to_right()
    {
        var model = ModelWith(new Conv(2, 3, new Tanh(), "valid"), new MaxPool(2), new Dense(6, new Tanh()));

        Assert.All(model.Layers.Layers, l => Assert.True(l.InputShape[0] <= 0));

        model.GetResults(new Tensor(Filled(8, 8, 1, 0.5)));

        var layers = model.Layers.Layers;
        Assert.Equal(new[] { 8, 8, 1 }, layers[0].InputShape);
        Assert.Equal(new[] { 6, 6, 2 }, layers[0].OutputShape);
        Assert.Equal(new[] { 6, 6, 2 }, layers[1].InputShape);
        Assert.Equal(new[] { 3, 3, 2 }, layers[1].OutputShape);
        Assert.Equal(new[] { 18 }, layers[2].InputShape);   // 3*3*2 flattened for Dense
        Assert.Equal(new[] { 6 }, layers[3].InputShape);
        Assert.All(layers, l => Assert.True(l.IsMaterialized));
    }

    // ------------------------------------------------------------ 2) wiring handles what forward handles

    [Theory]
    [InlineData(3)]   // [4,4,1] — channel dimension given
    [InlineData(2)]   // [4,4]   — 2D image without channels: THIS used to crash
    [InlineData(1)]   // [16]    — flat vector
    public void Conv_as_first_layer_accepts_every_rank_the_forward_accepts(int rank)
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        Tensor input = rank switch
        {
            3 => new Tensor(new double[4, 4, 1]),
            2 => new Tensor(new double[4, 4]),
            _ => new Tensor(new double[16]),
        };

        Tensor output = model.GetResults(input);

        // valid + kernel 2 over 4×4 → 3×3×1 → Dense(2)
        Assert.Equal(2, output.Data.Length);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].InputShape);
    }

    [Fact]
    public void MaxPool_as_first_layer_accepts_a_2D_input_too()
    {
        var model = ModelWith(new MaxPool(2));

        Tensor output = model.GetResults(new Tensor(new double[4, 4]));

        Assert.Equal(2, output.Data.Length);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].InputShape);
    }

    [Fact]
    public void Unsupported_rank_is_rejected_with_a_message_that_says_what_is_wrong()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        var ex = Assert.ThrowsAny<Exception>(() => model.GetResults(new Tensor(new double[2, 4, 4, 1])));

        // The point of the test: the message must not be empty. Previously this was a bare
        // `throw new Exception()` with no text, and the user had no way to find out what happened.
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
        Assert.Contains("Conv", ex.Message);
    }

    // ------------------------------------------------------------ 3) after wiring the shape is fixed

    [Fact]
    public void Conv_rejects_a_smaller_input_instead_of_reusing_the_stale_buffer()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        // the first sample fills the input buffer with hundreds
        model.GetResults(new Tensor(Filled(4, 4, 1, 100.0)));

        // the second sample is smaller; before the fix only its corner was overwritten and the
        // rest was computed from the leftover hundreds, silently, with no exception
        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        Assert.Contains("4×4×1", ex.Message);
        Assert.Contains("3×3×1", ex.Message);
    }

    [Fact]
    public void MaxPool_rejects_a_smaller_input_instead_of_reusing_the_stale_buffer()
    {
        var model = ModelWith(new MaxPool(2));

        model.GetResults(new Tensor(Filled(4, 4, 1, 100.0)));

        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        Assert.Contains("4×4×1", ex.Message);
    }

    [Fact]
    public void Conv_rejects_an_input_with_a_different_channel_count()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        model.GetResults(new Tensor(Filled(4, 4, 2, 1.0)));

        Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(4, 4, 1, 1.0))));
    }

    [Fact]
    public void Conv_rejects_a_bigger_input_with_a_clear_message_not_IndexOutOfRange()
    {
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        model.GetResults(new Tensor(Filled(4, 4, 1, 1.0)));

        // Previously: IndexOutOfRangeException from deep inside the copy loop, no mention of shapes.
        var ex = Assert.Throws<ArgumentException>(
            () => model.GetResults(new Tensor(Filled(5, 5, 1, 1.0))));

        Assert.Contains("4×4×1", ex.Message);
        Assert.Contains("5×5×1", ex.Message);
    }

    [Fact]
    public void The_right_shape_keeps_working_after_a_rejected_one()
    {
        // A rejected input must not disturb the layer: shapes stay, the model keeps working.
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        double[] before = model.GetResults(new Tensor(Filled(4, 4, 1, 0.25))).Data;

        Assert.ThrowsAny<Exception>(() => model.GetResults(new Tensor(Filled(3, 3, 1, 0.0))));

        double[] after = model.GetResults(new Tensor(Filled(4, 4, 1, 0.25))).Data;

        Assert.Equal(before, after);
        Assert.Equal(new[] { 4, 4, 1 }, model.Layers.Layers[0].InputShape);
    }

    // ------------------------------------------------------------ 4) forward does not touch the caller's input

    [Fact]
    public void Conv_does_not_reshape_the_tensor_it_was_given()
    {
        // The layer normalizes the shape for ITSELF; the tensor belongs to the caller and must not
        // change under them. Previously `FeedForward` called TensorValues.Reshape(...), so after a
        // pass through the model the user's tensor suddenly had a different Shape than they gave it.
        var model = ModelWith(new Conv(1, 2, new Linear(), "valid"));

        var input = new Tensor(new double[16]);
        int[] shapeBefore = (int[])input.Shape.Clone();

        model.GetResults(input);

        Assert.Equal(shapeBefore, input.Shape);
    }

    [Fact]
    public void MaxPool_does_not_reshape_the_tensor_it_was_given()
    {
        var model = ModelWith(new MaxPool(2));

        var input = new Tensor(new double[4, 4]);
        int[] shapeBefore = (int[])input.Shape.Clone();

        model.GetResults(input);

        Assert.Equal(shapeBefore, input.Shape);
    }

    [Fact]
    public void The_same_tensor_can_be_fed_twice_and_gives_the_same_answer()
    {
        // The most practical symptom: whoever keeps an input and sends it again must get the same result.
        //
        // NOTE: this is a GUARD, not proof; it passes before the fix too. The overwritten shape
        // ([9] → [3,3,1]) normalizes to itself the second time, so the mutation is not visible
        // from outside. The proof is the pair of tests above, which check the shape directly.
        var model = ModelWith(new Conv(1, 2, new Tanh(), "valid"));

        var input = new Tensor(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });

        double[] first = model.GetResults(input).Data;
        double[] second = model.GetResults(input).Data;

        Assert.Equal(first, second);
    }

    // ------------------------------------------------------------ 5) training goes through

    [Fact]
    public void Training_a_CNN_over_2D_samples_runs()
    {
        // Dataset [samples, height, width] → per sample [height, width], i.e. rank 2.
        // Exactly the case that used to fail with an empty exception during wiring.
        var model = ModelWith(new Conv(1, 3, new Tanh(), "valid"));
        model.Train.ShowLossChart = false;
        model.Train.AutoSave = false;
        model.Train.ShowModelInfo = false;
        model.Train.TestAfterTraining = false;

        var x = new double[20, 8, 8];
        var y = new double[20, 2];
        for (int n = 0; n < 20; n++)
        {
            for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                x[n, i, j] = (n + i * 8 + j) * 0.001;
            y[n, n % 2] = 1;
        }

        var original = Console.Out;
        Console.SetOut(TextWriter.Null);
        try { model.Train.TrainLoop(x, y, 2, 4); }
        finally { Console.SetOut(original); }

        Assert.Equal(2u, model.Train.CurrentEpoch);
        Assert.Equal(new[] { 8, 8, 1 }, model.Layers.Layers[0].InputShape);
    }
}

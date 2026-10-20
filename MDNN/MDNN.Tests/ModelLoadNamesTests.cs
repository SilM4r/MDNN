using System;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// When a model is loaded, the loss and activations are created by the NAME stored in the file.
///
/// Previously both factories silently turned anything unknown into a default: the loss into MSE,
/// the activation into Linear. The model then loaded without error and computed a different
/// function than it was trained with. This hit custom losses/activations and above all old files
/// with a Conv layer, which saved the activation name via ToString() (the full type name).
/// </summary>
public class ModelLoadNamesTests
{
    public static TheoryData<ActivationFunction> BuiltInActivations()
        => new() { new Linear(), new Sigmoid(), new Tanh(), new ReLU(), new LeakyReLU(), new Softmax() };

    [Theory]
    [MemberData(nameof(BuiltInActivations))]
    public void Every_built_in_activation_round_trips_by_name(ActivationFunction activation)
    {
        Assert.IsType(activation.GetType(), ActivationFunction.FromName(activation.Name));
    }

    [Theory]
    [MemberData(nameof(BuiltInActivations))]
    public void Full_type_name_from_old_conv_files_is_recognised(ActivationFunction activation)
    {
        // This is how Conv saved the name via ToString(): "MDNN.Activations.ReLU"
        string legacy = activation.GetType().FullName!;
        Assert.IsType(activation.GetType(), ActivationFunction.FromName(legacy));
    }

    [Theory]
    [InlineData("Swish")]
    [InlineData("")]
    [InlineData(null)]
    public void Unknown_activation_is_an_error_not_linear(string? name)
    {
        Assert.Throws<ArgumentException>(() => ActivationFunction.FromName(name));
    }

    [Fact]
    public void Built_in_losses_round_trip_by_name()
    {
        Assert.IsType<MSE>(Loss.FromName(new MSE().Name));
        Assert.IsType<CrossEntropy>(Loss.FromName(new CrossEntropy().Name));
    }

    [Theory]
    [InlineData("Huber")]
    [InlineData("")]
    [InlineData(null)]
    public void Unknown_loss_is_an_error_not_mse(string? name)
    {
        Assert.Throws<ArgumentException>(() => Loss.FromName(name));
    }

    /// <summary>Custom loss; the README allows deriving from Loss.</summary>
    private sealed class HalfMse : Loss
    {
        public override string Name => "HalfMSE";
        public override double LossFunction(double value, double target_value)
            => 0.5 * (value - target_value) * (value - target_value);
        public override double DerivativeOfLossFunction(double value, double target_value)
            => value - target_value;
    }

    [Fact]
    public void Loading_a_model_with_a_custom_loss_fails_loudly()
    {
        var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), new HalfMse(), seed: 1);
        model.GetResults(new Tensor(new double[] { 1.0, 2.0 }));
        string json = model.SaveAsJsonString();

        // Previously it loaded with MSE and further training would use a different objective.
        var target = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        var ex = Assert.Throws<ArgumentException>(() => target.LoadWeightsFromString(json));
        Assert.Contains("HalfMSE", ex.Message);
    }

    [Fact]
    public void Old_conv_file_with_type_name_loads_with_its_real_activation()
    {
        var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        model.Layers.Add(new Conv(2, 2, new ReLU(), "valid"));
        model.GetResults(new Tensor(new double[4, 4, 1]));

        // Old flat format (v0, no envelope or checksum) with the activation name from ToString().
        JsonNode modelNode = JsonNode.Parse(model.SaveAsJsonString())!["Model"]!;
        JsonNode conv = modelNode["Layers"]!.AsArray()[0]!;
        Assert.Equal("ReLU", conv["Name_of_activation_function"]!.GetValue<string>());
        conv["Name_of_activation_function"] = typeof(ReLU).FullName;

        var loaded = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        loaded.LoadWeightsFromString(modelNode.ToJsonString());

        // Previously Linear: a trained CNN behaved like a linear network after loading.
        Assert.IsType<ReLU>(loaded.Layers.Layers[0].Activation);
    }
}

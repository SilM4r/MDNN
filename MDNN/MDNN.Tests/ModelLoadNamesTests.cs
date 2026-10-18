using System;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Loss a aktivace se při načtení modelu staví podle JMÉNA ze souboru.
///
/// Dřív obě továrny všechno neznámé tiše převedly na výchozí hodnotu: loss na MSE,
/// aktivaci na Linear. Model se pak načetl bez chyby a počítal jinou funkcí, než se
/// trénoval. Týkalo se to vlastních loss/aktivací a hlavně starých souborů s Conv
/// vrstvou, která jméno aktivace ukládala přes ToString() (celé jméno typu).
/// </summary>
public class ModelLoadNamesTests
{
    public static TheoryData<Activation_func> BuiltInActivations()
        => new() { new Linear(), new Sigmoid(), new Tanh(), new ReLu(), new Leak_ReLu(), new Softmax() };

    [Theory]
    [MemberData(nameof(BuiltInActivations))]
    public void Every_built_in_activation_round_trips_by_name(Activation_func activation)
    {
        Assert.IsType(activation.GetType(), Activation_func.inicialization_activation_func(activation.Name));
    }

    [Theory]
    [MemberData(nameof(BuiltInActivations))]
    public void Full_type_name_from_old_conv_files_is_recognised(Activation_func activation)
    {
        // Takhle jméno ukládala Conv přes ToString(): "My_DNN.Activation_functions.ReLu"
        string legacy = activation.GetType().FullName!;
        Assert.IsType(activation.GetType(), Activation_func.inicialization_activation_func(legacy));
    }

    [Theory]
    [InlineData("Swish")]
    [InlineData("")]
    [InlineData(null)]
    public void Unknown_activation_is_an_error_not_linear(string? name)
    {
        Assert.Throws<ArgumentException>(() => Activation_func.inicialization_activation_func(name));
    }

    [Fact]
    public void Built_in_losses_round_trip_by_name()
    {
        Assert.IsType<MSE>(Loss.inicialization_Loss_func(new MSE().Name));
        Assert.IsType<CrossEntropy>(Loss.inicialization_Loss_func(new CrossEntropy().Name));
    }

    [Theory]
    [InlineData("Huber")]
    [InlineData("")]
    [InlineData(null)]
    public void Unknown_loss_is_an_error_not_mse(string? name)
    {
        Assert.Throws<ArgumentException>(() => Loss.inicialization_Loss_func(name));
    }

    /// <summary>Vlastní loss — README dědění z Loss povoluje.</summary>
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
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.1), new HalfMse(), seed: 1);
        model.GetResults(new Tensor(new double[] { 1.0, 2.0 }));
        string json = model.SaveAsJsonString();

        // Dřív se načetl s MSE a dotrénoval by se jinou účelovou funkcí.
        var target = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        var ex = Assert.Throws<ArgumentException>(() => target.LoadWeightsFromString(json));
        Assert.Contains("HalfMSE", ex.Message);
    }

    [Fact]
    public void Old_conv_file_with_type_name_loads_with_its_real_activation()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        model.Layers.Add(new Conv(2, 2, new ReLu(), "valid"));
        model.GetResults(new Tensor(new double[4, 4, 1]));

        // Starý plochý formát (v0, bez obálky a součtu) se jménem aktivace přes ToString().
        JsonNode modelNode = JsonNode.Parse(model.SaveAsJsonString())!["Model"]!;
        JsonNode conv = modelNode["Layers"]!.AsArray()[0]!;
        Assert.Equal("ReLu", conv["Name_of_activation_function"]!.GetValue<string>());
        conv["Name_of_activation_function"] = typeof(ReLu).FullName;

        var loaded = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        loaded.LoadWeightsFromString(modelNode.ToJsonString());

        // Dřív Linear — natrénovaná CNN se po načtení chovala jako lineární síť.
        Assert.IsType<ReLu>(loaded.Layers.Layers[0].Activation_Func);
    }
}

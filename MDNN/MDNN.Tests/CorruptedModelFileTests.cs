using System;
using System.Text.Json.Nodes;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Soubor s modelem, ve kterém chybí povinné pole, musí skončit srozumitelnou
/// <see cref="ModelFileCorruptedException"/> s názvem pole.
///
/// Dřív se s nullable poli exportních tříd pracovalo, jako by tam byla vždycky, takže
/// poškozený soubor spadl na NullReferenceException hluboko v konstruktoru vrstvy —
/// bez jediného slova o tom, co chybí. Kontrolní součet tohle nechytí u starého formátu
/// (v0, bez obálky), ani když soubor někdo upraví a součet přepočítá.
/// </summary>
public class CorruptedModelFileTests
{
    // Starý plochý formát (v0): holý JSON modelu, bez obálky a bez kontrolního součtu —
    // jen tak jde pole odebrat, aniž by to první chytil nesouhlasící součet.
    private static JsonNode SavedModel()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 1);
        model.Layers.Add(new Conv(2, 2, new ReLu(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.GetResults(new Tensor(new double[6, 6, 1]));
        return JsonNode.Parse(model.SaveAsJsonString())!["Model"]!.DeepClone();
    }

    private static void AssertCorrupted(JsonNode model, string expectedInMessage)
    {
        var target = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        var ex = Assert.Throws<ModelFileCorruptedException>(() => target.LoadWeightsFromString(model.ToJsonString()));
        Assert.Contains(expectedInMessage, ex.Message);
    }

    [Theory]
    [InlineData(0, "Kernel")]
    [InlineData(0, "Biases")]
    [InlineData(0, "InputShape")]
    [InlineData(1, "InputsShape")]
    [InlineData(1, "PoolSize")]
    [InlineData(2, "Neurons")]
    public void Missing_layer_field_is_reported_by_name(int layer, string field)
    {
        JsonNode model = SavedModel();
        model["Layers"]![layer]!.AsObject().Remove(field);

        AssertCorrupted(model, field);
    }

    [Theory]
    [InlineData("Optimizer")]
    [InlineData("Layers")]
    public void Missing_model_section_is_reported_by_name(string section)
    {
        JsonNode model = SavedModel();
        model.AsObject().Remove(section);

        AssertCorrupted(model, section);
    }

    [Fact]
    public void Null_layer_entry_is_reported()
    {
        JsonNode model = SavedModel();
        model["Layers"]![1] = null;

        AssertCorrupted(model, "prázdnou");
    }
}

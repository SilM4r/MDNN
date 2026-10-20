using System;
using System.Text.Json.Nodes;
using Xunit;
using MDNN;
using MDNN.Layers;
using MDNN.Activations;
using MDNN.Optimizers;
using MDNN.Losses;

namespace MDNN.Tests;

/// <summary>
/// A model file with a required field missing must fail with a clear
/// <see cref="ModelFileCorruptedException"/> that names the field.
///
/// Previously the nullable fields of the export classes were used as if always present, so a
/// corrupted file crashed with a NullReferenceException deep inside a layer constructor, with no
/// hint of what was missing. The checksum does not catch this for the old format (v0, no
/// envelope), nor when someone edits the file and recomputes the checksum.
/// </summary>
public class CorruptedModelFileTests
{
    // Old flat format (v0): bare model JSON, no envelope and no checksum. Only this way can a
    // field be removed without a checksum mismatch catching it first.
    private static JsonNode SavedModel()
    {
        var model = new NeuralNetwork(new Dense(2, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 1);
        model.Layers.Add(new Conv(2, 2, new ReLU(), "valid"));
        model.Layers.Add(new MaxPool(2));
        model.GetResults(new Tensor(new double[6, 6, 1]));
        return JsonNode.Parse(model.SaveAsJsonString())!["Model"]!.DeepClone();
    }

    private static void AssertCorrupted(JsonNode model, string expectedInMessage)
    {
        var target = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
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

        AssertCorrupted(model, "empty entry");
    }
}

using System.Text.Json;
using System.Text.Json.Serialization;

namespace My_DNN.Save_neural_network
{
    public class LayerConverter : JsonConverter<BaseExportLayer>
    {
        public override BaseExportLayer Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            using (var jsonDoc = JsonDocument.ParseValue(ref reader))
            {
                var root = jsonDoc.RootElement;
                var layerType = root.GetProperty("LayerType").GetString();

                return layerType switch
                {
                    "Dense" => JsonSerializer.Deserialize<ExportDenseLayer>(root.GetRawText(), options),
                    "Conv" => JsonSerializer.Deserialize<ExportCNNLayer>(root.GetRawText(), options),
                    "RNN" => JsonSerializer.Deserialize<ExportRnnLayer>(root.GetRawText(), options),
                    "MaxPool" => JsonSerializer.Deserialize<ExportMaxPoolLayer>(root.GetRawText(), options),
                    _ => throw new NotSupportedException($"Unknown layer type: {layerType}")
                };
            }
        }

        public override void Write(Utf8JsonWriter writer, BaseExportLayer value, JsonSerializerOptions options)
        {
            JsonSerializer.Serialize(writer, value, value.GetType(), options);
        }
    }

    public class LayerListConverter : JsonConverter<List<BaseExportLayer>>
    {
        public override List<BaseExportLayer> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            using (var jsonDoc = JsonDocument.ParseValue(ref reader))
            {
                var layers = new List<BaseExportLayer>();
                foreach (var element in jsonDoc.RootElement.EnumerateArray())
                {
                    layers.Add(JsonSerializer.Deserialize<BaseExportLayer>(element.GetRawText(), options));
                }
                return layers;
            }
        }

        public override void Write(Utf8JsonWriter writer, List<BaseExportLayer> value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            foreach (var layer in value)
            {
                JsonSerializer.Serialize(writer, layer, layer.GetType(), options);
            }
            writer.WriteEndArray();
        }
    }
}

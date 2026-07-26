using My_DNN.Layers;
using My_DNN.Layers.classes;
using System.Text.Json;
using System.Text.Json.Serialization;
namespace My_DNN.Save_neural_network
{
    public class NetworkSaveLoadManager
    {
        public string Note { get; set; }
        public double Valid_Loss { get; set; }
        public string Schema { get; set; }
        public uint Current_epoch { get; set; }
        public uint Target_epoch { get; set; }
        public uint Mini_batch { get; set; }

        public string Loss_functions { get; set; }

        public ExportOptimizer Optimizer { get; set; }

        [JsonConverter(typeof(LayerListConverter))]
        public List<BaseExportLayer> Layers { get; set; }

    
        public NetworkSaveLoadManager(MDNN model)
        {
            Schema = model.Schema;
            Current_epoch = model.Train.CurrentEpoch;
            Target_epoch = model.Train.TotalEpoch;
            Mini_batch = model.Train.MiniBatch;
            Loss_functions = model.Loss.Name;
            Valid_Loss = model.Loss.GetAverageLossPerIteration();
            Note = model.Note;
            Optimizer = new ExportOptimizer(model.Optimizer);
            Layers = new List<BaseExportLayer>();

            foreach (Layer layer in model.Layers.Layers)
            {
                if (layer is Dense)
                {
                    Layers.Add(new ExportDenseLayer((Dense)layer));
                }
                else if (layer is RNN)
                {
                    Layers.Add(new ExportRnnLayer((RNN)layer));
                }
                else if (layer is Conv)
                {
                    Layers.Add(new ExportCNNLayer((Conv)layer));
                }
                else if (layer is MaxPool)
                {
                    Layers.Add(new ExportMaxPoolLayer((MaxPool)layer));
                }
                else
                {
                    throw new Exception("not implemetet yet");
                }
                    
            }
        }

        [JsonConstructor]
        public NetworkSaveLoadManager() 
        {
        }

        // jádro serializace (bez souboru) — reuse pro disk i pro in-memory snapshot
        public string SaveToString()
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
            };
            return System.Text.Json.JsonSerializer.Serialize(this, options);
        }

        public void Save(string fileName)
        {
            File.WriteAllText(@$"{fileName}.json", SaveToString());
        }

        public static NetworkSaveLoadManager LoadFromString(string json)
        {
            JsonSerializerOptions options = new JsonSerializerOptions
            {
                Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
            };

            NetworkSaveLoadManager? model = System.Text.Json.JsonSerializer.Deserialize<NetworkSaveLoadManager>(json, options);

            if (model != null)
            {
                return model;
            }

            else
            {
                throw new ArgumentException("Bad format of file");
            }
        }

        public static NetworkSaveLoadManager Load(string fullPath)
        {
            return LoadFromString(File.ReadAllText(fullPath));
        }


    }
}

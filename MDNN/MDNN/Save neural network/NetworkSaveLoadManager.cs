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
            Current_epoch = model.Train.Current_epoch;
            Target_epoch = model.Train.Total_epoch;
            Mini_batch = model.Train.Mini_batch;
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

        public void Save(string fileName)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
            };
            string json = System.Text.Json.JsonSerializer.Serialize(this, options);
            
            File.WriteAllText(@$"{fileName}.json", json);
        }

        public static NetworkSaveLoadManager Load(string fullPath)
        {
            NetworkSaveLoadManager? model;

            string json = File.ReadAllText(fullPath);
            JsonSerializerOptions options = new JsonSerializerOptions
            {
                Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
            };

            model = System.Text.Json.JsonSerializer.Deserialize<NetworkSaveLoadManager>(json, options);

            if (model != null)
            {
                return model;
            }

            else
            {
                throw new ArgumentException("Bad format of file");
            }
        }


    }
}

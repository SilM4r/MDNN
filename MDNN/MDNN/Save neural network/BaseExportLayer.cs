using My_DNN.Layers;
using System.Text.Json.Serialization;

namespace My_DNN.Save_neural_network
{
    public class BaseExportLayer
    {
        public string? LayerType 
        {
            get;
            set; 
        }

        // Povinné pole vrstvy ze souboru. Pole jsou nullable, protože v JSON chybět MŮŽOU —
        // dřív se s nimi ale pracovalo, jako by tam byla vždycky, takže poškozený soubor
        // (třeba bez "Kernel") spadl na NullReferenceException hluboko v konstruktoru vrstvy.
        internal static T Required<T>(T? value, string layerType, string field) where T : class
            => value ?? throw new ModelFileCorruptedException(
                $"Layer {layerType} in the model file is missing the field '{field}'.");

        internal static T Required<T>(T? value, string layerType, string field) where T : struct
            => value ?? throw new ModelFileCorruptedException(
                $"Layer {layerType} in the model file is missing the field '{field}'.");
    }

    public class ExportDenseLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }

        // Dropout nastavený PŘÍMO na téhle vrstvě. null = vrstva dědí globální hodnotu,
        // takže se null musí i uložit — kdyby se místo něj zapsala nula, načtený model by
        // měl dropout natvrdo vypnutý a globální nastavení by na něj přestalo platit.
        // (Globální hodnota sama patří k modelu, ne k vrstvě, a ukládá se zvlášť.)
        public double? DropoutRate { get; set; }

        public List<ExportNeuron>? Neurons { get; set; }

        [JsonConstructor]
        public ExportDenseLayer()
        {
            LayerType = "Dense";
        }

        public ExportDenseLayer(Dense layer, bool includeOptimizerState = true)
        {
            LayerType = "Dense";

            Name_of_activation_function = layer.Activation_Func.Name;
            DropoutRate = layer.DropoutRate;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron, includeOptimizerState));
            }
        }
    }

    // Conv vrstva
    public class ExportCNNLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }
        public string? Padding { get; set; }
        public double[][][][]? Kernel { get; set; }
        public double[]? Biases { get; set; }
        public int[]? InputShape { get; set; }

        // Stav optimizeru vrstvy (Conv má jeden optimizer pro kernely i biasy) — viz
        // ExportNeuron.OptimizerState.
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Dictionary<string, double[]>? OptimizerState { get; set; }

        [JsonConstructor]
        public ExportCNNLayer()
        {
            LayerType = "Conv";
        }

        public ExportCNNLayer(Conv layer, bool includeOptimizerState = true)
        {
            LayerType = "Conv";
            Kernel = layer.Kernel;
            Biases = layer.Biases;
            // .Name, ne .ToString() — ToString() dá název typu ("My_DNN.Activation_functions.ReLu"),
            // který inicialization_activation_func dřív neznal a spadl do default → Linear.
            // Natrénovaná CNN se tak po načtení tiše chovala jako lineární síť. Soubory
            // uložené ještě přes ToString() teď továrna rozpozná (zkrátí jméno typu).
            Name_of_activation_function = layer.Activation_Func.Name;
            Padding = layer.Padding;
            InputShape = layer.Input_size_and_shape;
            OptimizerState = includeOptimizerState ? layer.OptimizerForSave.ExportState() : null;
        }
    }

    public class ExportMaxPoolLayer : BaseExportLayer
    {
        public int[]? InputsShape { get; set; }
        public int? PoolSize { get; set; }

        [JsonConstructor]
        public ExportMaxPoolLayer()
        {
            LayerType = "MaxPool";
        }

        public ExportMaxPoolLayer(MaxPool layer)
        {
            LayerType = "MaxPool";
            InputsShape = layer.Input_size_and_shape;
            PoolSize = layer.PoolSize;
        }
    }

    // RNN vrstva
    public class ExportRnnLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }
        public List<ExportNeuron>? Neurons { get; set; }

        [JsonConstructor]
        public ExportRnnLayer()
        {
            LayerType = "RNN";
        }

        public ExportRnnLayer(RNN layer, bool includeOptimizerState = true)
        {
            LayerType = "RNN";

            Name_of_activation_function = layer.Activation_Func.Name;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron, includeOptimizerState));
            }
        }
    }
}

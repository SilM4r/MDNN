using MDNN.Layers;
using System.Text.Json.Serialization;

namespace MDNN.Serialization
{
    // Property names of the Export* classes are the JSON keys of the model file format —
    // do NOT rename them, or previously saved models will stop loading.
    internal class BaseExportLayer
    {
        public string? LayerType 
        {
            get;
            set; 
        }

        // A required layer field from the file. Fields are nullable because they CAN be missing in
        // the JSON — previously they were used as if always present, so a damaged file (e.g. without
        // "Kernel") crashed with a NullReferenceException deep in the layer constructor.
        internal static T Required<T>(T? value, string layerType, string field) where T : class
            => value ?? throw new ModelFileCorruptedException(
                $"Layer {layerType} in the model file is missing the field '{field}'.");

        internal static T Required<T>(T? value, string layerType, string field) where T : struct
            => value ?? throw new ModelFileCorruptedException(
                $"Layer {layerType} in the model file is missing the field '{field}'.");
    }

    internal class ExportDenseLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }

        // Dropout set DIRECTLY on this layer. null = the layer inherits the global value, so null
        // must be saved as well — writing zero instead would hard-disable dropout in the loaded
        // model and the global setting would no longer apply to it.
        // (The global value belongs to the model, not the layer, and is saved separately.)
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

            Name_of_activation_function = layer.Activation.Name;
            DropoutRate = layer.DropoutRate;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron, includeOptimizerState));
            }
        }
    }

    // Conv layer
    internal class ExportCNNLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }
        public string? Padding { get; set; }
        public double[][][][]? Kernel { get; set; }
        public double[]? Biases { get; set; }
        public int[]? InputShape { get; set; }

        // Optimizer state of the layer (Conv has one optimizer for both kernels and biases) — see
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
            Kernel = layer.Kernels;
            Biases = layer.Biases;
            // .Name, not .ToString() — ToString() gives the type name ("MDNN.Activations.ReLU"),
            // which FromName previously did not recognize and fell back to Linear, so a trained CNN
            // silently behaved as a linear network after loading. Files saved via ToString() are now
            // recognized by the factory (it shortens the type name).
            Name_of_activation_function = layer.Activation.Name;
            Padding = layer.Padding;
            InputShape = layer.InputShape;
            OptimizerState = includeOptimizerState ? layer.OptimizerForSave.ExportState() : null;
        }
    }

    internal class ExportMaxPoolLayer : BaseExportLayer
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
            InputsShape = layer.InputShape;
            PoolSize = layer.PoolSize;
        }
    }

    // RNN layer
    internal class ExportRnnLayer : BaseExportLayer
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

            Name_of_activation_function = layer.Activation.Name;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron, includeOptimizerState));
            }
        }
    }
}

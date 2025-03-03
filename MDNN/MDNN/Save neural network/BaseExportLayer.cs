using My_DNN.Layers;
using My_DNN.Layers.classes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace My_DNN.Save_neural_network
{
    public class BaseExportLayer
    {
        public string? LayerType 
        {
            get;
            set; 
        }

    }

    public class ExportDenseLayer : BaseExportLayer
    {
        public string? Name_of_activation_function { get; set; }
        public List<ExportNeuron>? Neurons { get; set; }

        [JsonConstructor]
        public ExportDenseLayer()
        {
            LayerType = "Dense";
        }

        public ExportDenseLayer(Dense layer)
        {
            LayerType = "Dense";

            Name_of_activation_function = layer.Activation_Func.Name;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron));
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

        [JsonConstructor]
        public ExportCNNLayer()
        {
            LayerType = "Conv";
        }

        public ExportCNNLayer(Conv layer)
        {
            LayerType = "Conv";
            Kernel = layer.Kernel;
            Biases = layer.Biases;
            Name_of_activation_function = layer.Activation_Func.ToString();
            Padding = layer.Padding;
            InputShape = layer.Input_size_and_shape;
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

        public ExportRnnLayer(RNN layer)
        {
            LayerType = "RNN";

            Name_of_activation_function = layer.Activation_Func.Name;

            Neurons = new List<ExportNeuron>();

            foreach (Neuron neuron in layer.Neurons)
            {
                this.Neurons.Add(new ExportNeuron(neuron));
            }
        }
    }
}

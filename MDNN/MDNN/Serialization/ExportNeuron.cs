using System.Text.Json.Serialization;

namespace MDNN.Serialization
{
    // Property names are JSON keys of the model file format — do not rename them.
    internal class ExportNeuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        // Optimizer state of this neuron (for Adam the moments m, v and step t) — see
        // Optimizer.ExportState. null = not saved (SaveAsJson(..., includeOptimizerState:
        // false), SGD, older files); the loaded neuron then starts with empty state.
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Dictionary<string, double[]>? OptimizerState { get; set; }

        [JsonConstructor]
        public ExportNeuron(double[] weights, double bias) 
        {
            Weights = weights;
            Bias = bias;
        }

        public ExportNeuron(Neuron neuron, bool includeOptimizerState = true)
        {
            Weights = neuron.Weights;
            Bias = neuron.Bias;
            OptimizerState = includeOptimizerState ? neuron.Optimizer.ExportState() : null;
        }
    }
}

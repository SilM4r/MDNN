using System.Text.Json.Serialization;

namespace My_DNN.Save_neural_network
{
    public class ExportNeuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        // Stav optimizeru tohoto neuronu (u Adamu momenty m, v a krok t) — viz
        // Optimizer.ExportState. null = neukládá se (SaveAsJson(..., includeOptimizerState:
        // false), SGD, starší soubory); načtený neuron pak začne s prázdným stavem.
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
            OptimizerState = includeOptimizerState ? neuron.optimizer.ExportState() : null;
        }
    }
}

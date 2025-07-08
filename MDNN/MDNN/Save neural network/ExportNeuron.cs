using System.Text.Json.Serialization;

namespace My_DNN.Save_neural_network
{
    public class ExportNeuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        [JsonConstructor]
        public ExportNeuron(double[] weights, double bias) 
        {
            Weights = weights;
            Bias = bias;
        }

        public ExportNeuron(Neuron neuron)
        {
            Weights = neuron.Weights;
            Bias = neuron.Bias;
        }
    }
}

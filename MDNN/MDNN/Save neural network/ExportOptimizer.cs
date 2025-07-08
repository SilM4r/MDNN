using My_DNN.Optimizers;
using System.Text.Json.Serialization;

namespace My_DNN.Save_neural_network
{
    public class ExportOptimizer
    {
        public string Name { get; set; }
        public double[] Hyperparameters { get; set; }

        public ExportOptimizer(Optimizer optimizer) 
        {
            Name = optimizer.Name;
            Hyperparameters = optimizer.Hyperparameters;
        }
        [JsonConstructor]
        public ExportOptimizer(string Name, double[] Hyperparameters)
        {
            this.Name = Name;
            this.Hyperparameters = Hyperparameters;
        }
    }
}

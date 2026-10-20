using MDNN.Optimizers;
using System.Text.Json.Serialization;

namespace MDNN.Serialization
{
    // Property names are JSON keys of the model file format — do not rename them.
    internal class ExportOptimizer
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

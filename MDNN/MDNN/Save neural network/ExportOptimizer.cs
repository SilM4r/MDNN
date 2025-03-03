using mdnn.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace mdnn.Save_neural_network
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

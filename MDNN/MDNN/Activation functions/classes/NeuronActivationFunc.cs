using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mdnn.Activation_functions.classes
{
    public abstract class NeuronActivationFunc: Activation_func
    {
        public override bool Apply_to_layer => false;
        public abstract override string Name { get; }
        public abstract override double Apply(double value);
        public abstract override double Derivative(double value);


    }
}

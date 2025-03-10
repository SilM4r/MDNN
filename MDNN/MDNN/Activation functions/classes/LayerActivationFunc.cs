using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN.Activation_functions
{
    public abstract class LayerActivationFunc : Activation_func
    {
        public override bool Apply_to_layer => true;

        public abstract double[] ApplyToLayer(double[] values);
        public abstract double[] DerivativeForLayer(double[] values);
    }
}

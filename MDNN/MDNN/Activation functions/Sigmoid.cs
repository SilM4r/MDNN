
using mdnn.Activation_functions.classes;

namespace mdnn.Activation_functions
{
    public class Sigmoid : NeuronActivationFunc
    {

        public override string Name
        {
            get { return "Sigmoid"; }
        }
        public override double Apply(double value)
        {
            return 1.0f / (1.0f + (double)Math.Exp(-value));
        }

        public override double Derivative(double value)
        {
            return Apply(value) * (1 - Apply(value));
        }
    }
}

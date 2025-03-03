

using mdnn.Activation_functions.classes;

namespace mdnn.Activation_functions
{
    public class Linear : NeuronActivationFunc
    {
        // public override string Name => "Linear";
        public override string Name
        {
            get { return "Linear"; }
        }

        public override double Apply(double value)
        {
            return value;
        }

        public override double Derivative(double value)
        {
            return value;
        }
    }
}



namespace My_DNN.Activation_functions
{
    public class Linear : ClassicActivationFunc
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

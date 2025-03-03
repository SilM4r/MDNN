

namespace My_DNN.Activation_functions
{
    public class ReLu : NeuronActivationFunc
    {
        public override string Name
        {
            get { return "ReLu"; }
        }

        public override double Apply(double value)
        {
            return Math.Max(0, value);
        }

        public override double Derivative(double value)
        {
            if (value < 0)
                return 0;

            else return 1; 
        }
    }
}

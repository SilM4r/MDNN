
namespace My_DNN.Activation_functions
{
    public class Tanh : ClassicActivationFunc
    {
        public override string Name
        {
            get { return "Tanh"; }
        }

        public override double Apply(double value)
        {
            return Math.Tanh(value);
        }

        public override double Derivative(double value)
        {
            return 1 - Math.Pow(Apply(value), 2);
        }
    }
}

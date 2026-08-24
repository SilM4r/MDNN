
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
            // Math.Pow je obecná mocninná funkce (větvení + vnitřní log/exp); na druhou
            // mocninu stačí jedno násobení. Běží to na každý neuron při každém backwardu.
            double s = Apply(value);
            return 1 - s * s;
        }
    }
}

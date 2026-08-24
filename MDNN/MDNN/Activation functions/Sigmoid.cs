
namespace My_DNN.Activation_functions
{
    public class Sigmoid : ClassicActivationFunc
    {

        public override string Name
        {
            get { return "Sigmoid"; }
        }
        public override double Apply(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public override double Derivative(double value)
        {
            // s(1-s) potřebuje sigmoidu jednou, ne dvakrát — dřív se tu Math.Exp
            // počítal 2×. Výsledek je bitově stejný, jen o polovinu levnější.
            double s = Apply(value);
            return s * (1 - s);
        }
    }
}

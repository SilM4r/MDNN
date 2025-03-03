

namespace My_DNN.Loss_functions
{
    public class CrossEntropy: Loss
    {
        public override string Name => "Cross Entropy";
        public override double LossFunction(double value, double target_value)
        {
            if (target_value == 1)
            {
                return -Math.Log(value + 1e-15);
            }

            else
            {
                return -Math.Log(1 - value + 1e-15);
            }

            //return Math.Pow(value - target_value, 2);
        }

        public override double DerivativeOfLossFunction(double value, double target_value)
        {

            //return (target_value == 1) ? -1 / (value + 1e-15) : 1 / (1 - value + 1e-15);


            // Note: pokud je výstupní aktivační funkce SoftMax;
            return value - target_value;

        }
    }
}

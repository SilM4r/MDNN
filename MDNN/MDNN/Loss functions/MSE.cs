using My_DNN.Loss_functions;

namespace My_DNN
{
    public class MSE: Loss
    {
        public override string Name => "MSE";
        public override double LossFunction(double value, double target_value) 
        {
            // násobení místo Math.Pow — viz Tanh.Derivative; volá se na každý výstup
            double diff = value - target_value;
            return diff * diff;
        }

        public override double DerivativeOfLossFunction(double value, double target_value)
        {
            return 2*(value - target_value);
        }

    }
}

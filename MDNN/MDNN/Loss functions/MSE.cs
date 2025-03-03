using mdnn.Loss_functions;


namespace mdnn
{
    public class MSE: Loss
    {
        public override string Name => "MSE";
        public override double LossFunction(double value, double target_value) 
        {
            return Math.Pow(value - target_value, 2);
            //return  Math.Abs(value - target_value);
        }

        public override double DerivativeOfLossFunction(double value, double target_value)
        {
            return 2*(value - target_value);
        }

    }
}

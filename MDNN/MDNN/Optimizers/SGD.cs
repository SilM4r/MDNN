

namespace mdnn.Optimizers
{
    public class SGD : Optimizer
    {
        private double L;

        public double[] Parameters = new double[1];

        public override string Name => "SGD";
        public override double[] Hyperparameters => Parameters;

        public SGD(double learning_rate) 
        {
            L = learning_rate;
            Parameters[0] = L;
        }

        public override double Update(double w, double gradient)
        {
            w = w - L * gradient;

            return w;
        }
    }
}

namespace My_DNN.Optimizers
{
    public class Adam : Optimizer
    {
        private double b1;
        private double b2;
        private double L;
        private double m;
        private double v;

        private double[] Parameters = new double[2];

        public override double[] Hyperparameters => Parameters;
        public override string Name => "Adam";

        public Adam(double L, double beta1 = 0.9, double beta2 = 0.999)
        {
            this.L = L;
            b1 = beta1;
            b2 = beta2;

            Parameters = new double[] { this.L, b1, b2 };

        }

        public override double Update(double w, double gradient)
        {

            m = b1 * m + (1 - b1) * gradient;
            v = b2 * v + (1 - b2) * Math.Pow(gradient, 2);
            w = w - L * (m / Rms(v));
            return w;
        }
    }
}

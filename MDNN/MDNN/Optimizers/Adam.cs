namespace My_DNN.Optimizers
{
    public class Adam : Optimizer
    {
        private double b1;
        private double b2;
        private double L;
        private double[] m = new double[0];
        private double[] v = new double[0];
        private int[] t = new int[0];

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

        public override double Update(double w, double gradient, int i)
        {
            if (m.Length <= i) 
            { 
                Array.Resize(ref m, i+1); 
                Array.Resize(ref v, i+1); 
                Array.Resize(ref t, i+1); 
            }
            t[i]++;
            m[i] = b1 * m[i] + (1 - b1) * gradient;
            v[i] = b2 * v[i] + (1 - b2) * gradient * gradient;
            double mHat = m[i] / (1 - Math.Pow(b1, t[i]));   
            double vHat = v[i] / (1 - Math.Pow(b2, t[i]));
            return w - L * mHat / (Math.Sqrt(vHat) + 1e-8);
        }
    }
}

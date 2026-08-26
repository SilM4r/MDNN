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
                // Zdvojnásobení kapacity, ne růst po jedné. Update se volá pro každou váhu
                // zvlášť a index roste 0,1,2,…, takže exaktní Array.Resize znamenal jedno
                // přealokování a překopírování NA KAŽDOU VÁHU — kvadraticky. Změřeno na
                // 784 vahách: první dávka 0.2174 ms proti 0.0087 ms těch dalších, tedy
                // 0.21 ms režie navíc na každý neuron (~107 ms jednorázově u Dense(512)).
                // Přebytečná kapacita je vynulovaná, takže se chová jako nedotčený parametr.
                int capacity = Math.Max(i + 1, m.Length * 2);

                Array.Resize(ref m, capacity);
                Array.Resize(ref v, capacity);
                Array.Resize(ref t, capacity);
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

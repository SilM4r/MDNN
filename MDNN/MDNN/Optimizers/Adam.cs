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

        // Přesná velikost, žádná rezerva navíc: přebytečná kapacita (dřív zdvojnásobování,
        // 785 → 1024 na neuron) stála ~3 % v každém UpdateParams kvůli větší paměti.
        public override void EnsureCapacity(int parameterCount)
        {
            if (m.Length >= parameterCount) return;

            Array.Resize(ref m, parameterCount);
            Array.Resize(ref v, parameterCount);
            Array.Resize(ref t, parameterCount);
        }

        public override double Update(double w, double gradient, int i)
        {
            // Pojistka pro volání bez EnsureCapacity (optimizer použitý samostatně).
            // Vrstvy si kapacitu připraví dopředu, takže v tréninku se sem nevstoupí.
            if (m.Length <= i) EnsureCapacity(i + 1);
            t[i]++;
            m[i] = b1 * m[i] + (1 - b1) * gradient;
            v[i] = b2 * v[i] + (1 - b2) * gradient * gradient;
            double mHat = m[i] / (1 - Math.Pow(b1, t[i]));   
            double vHat = v[i] / (1 - Math.Pow(b2, t[i]));
            return w - L * mHat / (Math.Sqrt(vHat) + 1e-8);
        }
    }
}

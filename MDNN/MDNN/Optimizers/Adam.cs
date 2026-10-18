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

        public override Optimizer CreateFresh() => new Adam(L, b1, b2);

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
            double mHat = m[i] / BiasCorrection1(t[i]);
            double vHat = v[i] / BiasCorrection2(t[i]);
            return w - L * mHat / (Math.Sqrt(vHat) + 1e-8);
        }

        // Korekce zkreslení 1 − β^t. Dřív se počítala přes Math.Pow DVAKRÁT NA KAŽDOU VÁHU,
        // a to byla zhruba třetina času UpdateParams. Přitom t mají všechny váhy jednoho
        // neuronu stejné (Update je volá v jedné smyčce za sebou), takže stačí si pamatovat
        // poslední t a jeho hodnotu — Math.Pow se pak počítá jednou na neuron a krok.
        //
        // Hodnota je přesně tatáž, jen se nepočítá znovu. Proto ne průběžné násobení β^t:
        // to by bylo ještě levnější, ale posunulo by zaokrouhlení a výsledky by se proti
        // dřívějšku (i proti PyTorchi) rozešly v posledních cifrách.
        private int cachedT1 = -1, cachedT2 = -1;
        private double correction1, correction2;

        private double BiasCorrection1(int step)
        {
            if (step != cachedT1)
            {
                cachedT1 = step;
                correction1 = 1 - Math.Pow(b1, step);
            }
            return correction1;
        }

        private double BiasCorrection2(int step)
        {
            if (step != cachedT2)
            {
                cachedT2 = step;
                correction2 = 1 - Math.Pow(b2, step);
            }
            return correction2;
        }
    }
}

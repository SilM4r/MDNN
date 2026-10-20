namespace MDNN.Optimizers
{
    /// <summary>
    /// The Adam optimizer (adaptive moment estimation) with bias correction.
    /// </summary>
    public class Adam : Optimizer
    {
        private readonly double b1;
        private readonly double b2;
        private readonly double learningRate;
        private double[] m = new double[0];
        private double[] v = new double[0];
        private int[] t = new int[0];

        /// <inheritdoc/>
        public override double[] Hyperparameters => new[] { learningRate, b1, b2 };
        /// <inheritdoc/>
        public override string Name => "Adam";

        /// <summary>
        /// Creates an Adam optimizer.
        /// </summary>
        /// <param name="learningRate">Step size.</param>
        /// <param name="beta1">Decay rate of the first-moment estimate.</param>
        /// <param name="beta2">Decay rate of the second-moment estimate.</param>
        public Adam(double learningRate, double beta1 = 0.9, double beta2 = 0.999)
        {
            this.learningRate = learningRate;
            b1 = beta1;
            b2 = beta2;
        }

        /// <inheritdoc/>
        public override Optimizer CreateFresh() => new Adam(learningRate, b1, b2);

        // t is an integer but stored as double in the file — exact up to 2^53.
        /// <inheritdoc/>
        public override Dictionary<string, double[]> ExportState() => new()
        {
            ["m"] = (double[])m.Clone(),
            ["v"] = (double[])v.Clone(),
            ["t"] = t.Select(step => (double)step).ToArray(),
        };

        /// <inheritdoc/>
        public override void ImportState(Dictionary<string, double[]> state)
        {
            double[] newM = RequireState(state, "m", null);
            double[] newV = RequireState(state, "v", newM.Length);
            double[] newT = RequireState(state, "t", newM.Length);

            m = newM;
            v = newV;
            t = newT.Select(step => (int)step).ToArray();
            cachedT1 = cachedT2 = -1;   // bias correction is recomputed for the loaded t
        }

        // Exact size, no spare room: excess capacity (previously doubling, 785 → 1024 per neuron)
        // cost ~3 % in every UpdateParams due to the larger memory footprint.
        /// <inheritdoc/>
        public override void EnsureCapacity(int parameterCount)
        {
            if (m.Length >= parameterCount) return;

            Array.Resize(ref m, parameterCount);
            Array.Resize(ref v, parameterCount);
            Array.Resize(ref t, parameterCount);
        }

        /// <inheritdoc/>
        public override double Update(double w, double gradient, int paramIndex)
        {
            // Safeguard for calls without EnsureCapacity (optimizer used standalone).
            // Layers prepare capacity ahead of time, so training never enters here.
            if (m.Length <= paramIndex) EnsureCapacity(paramIndex + 1);
            t[paramIndex]++;
            m[paramIndex] = b1 * m[paramIndex] + (1 - b1) * gradient;
            v[paramIndex] = b2 * v[paramIndex] + (1 - b2) * gradient * gradient;
            double mHat = m[paramIndex] / BiasCorrection1(t[paramIndex]);
            double vHat = v[paramIndex] / BiasCorrection2(t[paramIndex]);
            return w - learningRate * mHat / (Math.Sqrt(vHat) + 1e-8);
        }

        // Bias correction 1 − β^t. Previously computed via Math.Pow TWICE PER WEIGHT, which was about
        // a third of UpdateParams time. All weights of one neuron share the same t (Update is called
        // for them in one loop), so remembering the last t and its value is enough — Math.Pow then
        // runs once per neuron and step.
        //
        // The value is exactly the same, just not recomputed. Hence not a running product of β^t:
        // cheaper still, but it would shift rounding and results would differ in the last digits
        // from before (and from PyTorch).
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

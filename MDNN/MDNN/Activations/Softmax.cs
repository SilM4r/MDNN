
namespace MDNN.Activations
{
    /// <summary>
    /// Softmax over the whole layer output (numerically stabilized by subtracting the maximum).
    /// </summary>
    public class Softmax : LayerActivationFunction
    {
        /// <inheritdoc/>
        public override string Name
        {
            get { return "Softmax"; }
        }

        /// <summary>
        /// Returns <paramref name="value"/> unchanged; softmax is applied layer-wide via <see cref="ApplyToLayer"/>.
        /// </summary>
        public override double Apply(double value)
        {
            return value;
        }

        // Softmax has no per-element derivative — output s_i depends on ALL inputs.
        // Previously this returned the Jacobian diagonal via a stateful counter advanced on every
        // call: wrong math, dependence on call order and a race in a parallel backward pass.
        // Backward for softmax goes exclusively through BackwardForLayer.
        /// <summary>
        /// Not supported: softmax has no per-element derivative. Use <see cref="BackwardForLayer"/>.
        /// </summary>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override double Derivative(double value)
        {
            throw new NotSupportedException(
                "Softmax has no per-element derivative (each output depends on all inputs). " +
                "Use BackwardForLayer(rawOutput, gradFromAbove).");
        }

        // ∂L/∂z_i = Σ_j g_j · J[j,i],  where J[j,i] = s_j·(δ_ji − s_i)
        //         = s_i · (g_i − Σ_j g_j·s_j)
        // The Jacobian is never materialized — a single dot product suffices, so it is
        // O(n) instead of O(n²) with no matrix allocation.
        /// <inheritdoc/>
        public override double[] BackwardForLayer(double[] rawOutput, double[] gradFromAbove)
        {
            if (rawOutput.Length != gradFromAbove.Length)
            {
                throw new ArgumentException(
                    $"rawOutput has {rawOutput.Length} elements, the gradient from above {gradFromAbove.Length}.");
            }

            double[] s = ApplyToLayer(rawOutput);

            double dot = 0;
            for (int j = 0; j < s.Length; j++)
            {
                dot += gradFromAbove[j] * s[j];
            }

            double[] result = new double[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                result[i] = s[i] * (gradFromAbove[i] - dot);
            }

            return result;
        }

        /// <inheritdoc/>
        public override double[] ApplyToLayer(double[] values)
        {
            int length = values.Length;
            double[] result = new double[length];
            double expSum = 0.0;


            double max = values.Max();

            for (int i = 0; i < length; i++)
            {
                result[i] = Math.Exp(values[i] - max); 
                expSum += result[i];
            }
            for (int i = 0; i < length; i++)
            {
                result[i] /= expSum; 
            }
            return result;
        }

    }
}
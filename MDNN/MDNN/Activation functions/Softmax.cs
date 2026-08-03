
namespace My_DNN.Activation_functions
{
    public class Softmax : LayerActivationFunc
    {
        public override string Name
        {
            get { return "Softmax"; }
        }

        public override double Apply(double value)
        {
            return value;
        }

        // Softmax nemá per-prvkovou derivaci — výstup s_i závisí na VŠECH vstupech.
        // Dřív se tady vracela diagonála Jacobiánu přes stavový čítač, který se posouval
        // s každým voláním: špatná matematika, závislost na pořadí volání a race při
        // paralelním backwardu. Backward pro softmax jde výhradně přes BackwardForLayer.
        public override double Derivative(double value)
        {
            throw new NotSupportedException(
                "Softmax nemá per-prvkovou derivaci (výstup závisí na všech vstupech). " +
                "Použij BackwardForLayer(rawOutput, gradFromAbove).");
        }

        // ∂L/∂z_i = Σ_j g_j · J[j,i],  kde J[j,i] = s_j·(δ_ji − s_i)
        //         = s_i · (g_i − Σ_j g_j·s_j)
        // Jacobián se nikdy nematerializuje — stačí jeden skalární součin, takže je to
        // O(n) místo O(n²) a bez alokace matice.
        public override double[] BackwardForLayer(double[] rawOutput, double[] gradFromAbove)
        {
            if (rawOutput.Length != gradFromAbove.Length)
            {
                throw new ArgumentException(
                    $"rawOutput má {rawOutput.Length} prvků, gradient shora {gradFromAbove.Length}.");
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
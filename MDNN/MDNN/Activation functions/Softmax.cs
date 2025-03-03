
using mdnn.Activation_functions.classes;

namespace mdnn.Activation_functions
{
    public class Softmax : LayerActivationFunc
    {

        private double[,] jacobian;
        private int a = -1;

        public override string Name
        {
            get { return "Softmax"; }
        }

        public override double Apply(double value)
        {
            return value;
        }

        public override double Derivative(double value)
        {
            a++;

            return jacobian[a,a];  
        }

        public override double[] ApplyToLayer(double[] values)
        {
            int length = values.Length;
            double[] result = new double[length];
            double expSum = 0.0;

            // Numerická stabilita - posunutí hodnot
            double max = values.Max();

            for (int i = 0; i < length; i++)
            {
                result[i] = Math.Exp(values[i] - max); // Posun hodnot
                expSum += result[i];
            }

            for (int i = 0; i < length; i++)
            {
                result[i] /= expSum; // Normalizace
            }

            return result;
        }

        public override double[] DerivativeForLayer(double[] values)
        {
            int length = values.Length;
            double[] derivatives = new double[length];
            a = -1;


            double[] softmaxValues = ApplyToLayer(values);

            jacobian = new double[length,length];


            // Vytvoření Jacobian matice
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    if (i == j)
                    {
                        jacobian[i, j] = softmaxValues[i] * (1.0 - softmaxValues[i]);
                    }
                    else
                    {
                        jacobian[i, j] = -softmaxValues[i] * softmaxValues[j];
                    }
                }
            }

            // Výpočet derivace pro každý neuron (diagonální prvky Jacobian matice)
            for (int i = 0; i < length; i++)
            {
                derivatives[i] = jacobian[i, i];
            }

            return derivatives;
        }
    }



    /*
    public class Softmax : LayerActivationFunc
    {
        double expSum = 0;
        List<double> expValues = new List<double>();

        public override string Name
        {
            get { return "Softmax"; }
        }

        public override double Apply(double value)
        {
            double expValue = Math.Exp(value);
            expSum += expValue;
            expValues.Add(expValue);

            return value;
        }

        public override double Derivative(double value)
        {
            double expValue = Math.Exp(value);
            //value = (expValue * expSum - Math.Pow(expValue,2)) / (Math.Pow(expSum, 2));

            expSum = 0;
            return value;
        }

        public override double[] ApplyToLayer(double[] values)
        {
            double[] result = new double[values.Length];

            for(int i = 0; i < values.Length; i++)
            {
                // result[i] =  Math.Exp(value[i]) / expSum;
                result[i] = expValues[i] / expSum;
            }

            expSum = 0;
            expValues = new List<double>();

            return result.ToArray();
        }
        public override double[] DerivativeForLayer(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                double expValue = Math.Exp(values[i]);
                expSum += expValue;
            }

            return values;
        }
    }
    */
}

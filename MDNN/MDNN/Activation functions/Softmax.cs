﻿
namespace My_DNN.Activation_functions
{
    public class Softmax : LayerActivationFunc
    {
        private double expSum = 0;
        private List<double> expValues = new List<double>();

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
            return jacobian[a, a];
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

        public override double[] DerivativeForLayer(double[] values)
        {
            int length = values.Length;
            double[] derivatives = new double[length];
            a = -1;

            double[] softmaxValues = ApplyToLayer(values);

            jacobian = new double[length, length];
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
            for (int i = 0; i < length; i++)
            {
                derivatives[i] = jacobian[i, i];
            }

            return derivatives;
        }
    }
}
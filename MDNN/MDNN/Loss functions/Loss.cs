

namespace mdnn.Loss_functions
{
    public abstract class Loss
    {
        public double[]? Losses
        {
            get { return losses; }
        }

        private int iteration = 0;
        private double totalLossSum = 0;

        private double[]? losses;

        public abstract string Name { get; }

        public abstract double LossFunction(double value, double target_value);

        public abstract double DerivativeOfLossFunction(double value, double target_value);

        public void CalculateLoss(double[] value, double[] target_value)
        {
            int n = value.Length;

            losses = new double[n];

            for (int i = 0; i < n; i++)
            {
                double lossValue = LossFunction(value[i], target_value[i]);
                losses[i] = lossValue;
                totalLossSum += lossValue;
            }

            iteration++;

        }

        public double CalculateAndGetLoss(double[] value, double[] target_value)
        {
            int n = value.Length;

            double lossSum = 0;

            for (int i = 0; i < n; i++)
            {
                lossSum += LossFunction(value[i], target_value[i]);
            }

            return lossSum;
        }


        public double GetResetAverageLossPerIteration()
        {
            if (iteration == 0)
            {
                return 0;
            }

            double output = totalLossSum / iteration;
            iteration = 0;
            totalLossSum = 0;

            return output;
        }

        public double GetAverageLossPerIteration()
        {  
            if (iteration == 0)
            {
                return 0;
            }

            return totalLossSum / iteration;
        }

        public void ResetAverageLossPerIteration()
        {
            iteration = 0;
            totalLossSum = 0;
        }


        public static Loss inicialization_Loss_func(string name_of_loss_function)
        {
            switch (name_of_loss_function)
            {
                case "Cross Entropy":
                    return new CrossEntropy();
                case "MSE":
                    return new MSE();
                default: return new MSE();
            }
        }

        public static Loss MSE()
        {
            return new MSE();
        }
        public static Loss CrossEntropy()
        {
            return new CrossEntropy();
        }
    }
}

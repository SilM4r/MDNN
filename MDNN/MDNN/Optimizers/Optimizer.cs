﻿using My_DNN.Save_neural_network;


namespace My_DNN.Optimizers
{
    public abstract class Optimizer
    {
        public abstract string Name { get; }
        public abstract double[] Hyperparameters { get; }
        public abstract double Update(double w, double gradient);

        protected static double Rms(double val)
        {
            double sqrt = Math.Sqrt(val);
            if (sqrt == 0)
            {
                sqrt += double.Epsilon;
            }
            return sqrt;
        }

        public static Optimizer Refactor_optimizer(ExportOptimizer Optimizer)
        {
            return GetOptimizer(Optimizer.Name, Optimizer.Hyperparameters);
        }

        public static Optimizer Clone_optimizer(Optimizer Optimizer)
        {
            return GetOptimizer(Optimizer.Name, Optimizer.Hyperparameters);
        }

        private static Optimizer GetOptimizer(string name, double[] hyperparameters)
        {
            switch (name)
            {
                case "SGD":
                    return new SGD(hyperparameters[0]);
                case "Adam":
                    return new Adam(hyperparameters[0], hyperparameters[1], hyperparameters[2]);
                case "Momentum":
                    return new Momentum(hyperparameters[0], hyperparameters[1]);
                default: throw new ArgumentException($"this optimizer ({name}) does not exist");
            }
        }
    }
}

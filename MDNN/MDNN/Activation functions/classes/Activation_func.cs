using My_DNN.Activation_functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN
{
    public abstract class Activation_func
    {
        public abstract string Name { get; }
        public abstract double Apply(double value);
        public abstract double Derivative(double value);
        public abstract bool Apply_to_layer { get; }

        public static Activation_func inicialization_activation_func(string name_of_activation_function)
        {
            switch (name_of_activation_function.ToLower())
            {
                case "sigmoid":
                    return new Sigmoid();
                case "tanh":
                    return new Tanh();
                case "relu":
                    return new ReLu();
                case "leak_relu":
                    return new Leak_ReLu();
                case "softmax":
                    return new Softmax();
                default: return new Linear();
            }
        }

    }
}

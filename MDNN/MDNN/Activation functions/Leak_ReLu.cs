﻿
namespace My_DNN.Activation_functions
{
    public class Leak_ReLu : ClassicActivationFunc
    {

        public override string Name
        {
            get { return "Leak_ReLu"; }
        }

        public override double Apply(double value)
        {
            if (value < 0)
                return value * 0.01;
            else
                return value;
        }

        public override double Derivative(double value)
        {
            if (value < 0)
                return 0.01;
            else return 1;
        }
    }
}

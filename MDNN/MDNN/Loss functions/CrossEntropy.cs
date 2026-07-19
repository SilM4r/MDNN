namespace My_DNN.Loss_functions
{
    public class CrossEntropy: Loss
    {
        public override string Name => "Cross Entropy";
        // Kategorická cross-entropy: L = -Σ t_i · ln(s_i)
        public override double LossFunction(double value, double targetValue)
        {
            return -targetValue * Math.Log(value + 1e-15);
        }

        // Fúzovaný gradient softmax + kategorická CE: dL/dz = s - t.
        // Platí jen pro softmax výstup; backprop díky FusedWithSoftmax bere
        // tuto hodnotu jako hotovou deltu a NEnásobí ji softmax derivací.
        public override double DerivativeOfLossFunction(double value, double targetValue)
        {
            return value - targetValue;
        }

        public override bool RequiresSoftmax => true;
    }
}

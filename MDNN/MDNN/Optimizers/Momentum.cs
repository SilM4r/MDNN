namespace My_DNN.Optimizers
{
    public class Momentum : Optimizer
    {
        private double L;
        private double a;

        private double[] velocity = new double[0];

        public double[] Parameters = new double[2];

        public override double[] Hyperparameters => Parameters;
        public override string Name => "Momentum";

        public Momentum(double L, double a) 
        {
            this.L = L;
            this.a = a;

            Parameters = new double[] {L,a};
        }

        // Přesná velikost — viz Adam.EnsureCapacity.
        public override void EnsureCapacity(int parameterCount)
        {
            if (velocity.Length < parameterCount) Array.Resize(ref velocity, parameterCount);
        }

        public override double Update(double w, double gradient, int i)
        {
            // Pojistka pro volání bez EnsureCapacity — viz Adam.Update.
            if (velocity.Length <= i) EnsureCapacity(i + 1);
            velocity[i] = a * velocity[i] + L * gradient;
            return w - velocity[i];
        }
    }
}

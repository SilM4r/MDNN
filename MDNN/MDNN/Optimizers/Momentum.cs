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

        public override double Update(double w, double gradient, int i)
        {
            if (velocity.Length <= i) Array.Resize(ref velocity, i + 1);  
            velocity[i] = a * velocity[i] + L * gradient;
            return w - velocity[i];
        }
    }
}

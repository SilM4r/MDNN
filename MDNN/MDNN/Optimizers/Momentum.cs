namespace My_DNN.Optimizers
{
    public class Momentum : Optimizer
    {
        private double L;
        private double a;

        private double velocity;

        public double[] Parameters = new double[2];

        public override double[] Hyperparameters => Parameters;
        public override string Name => "Momentum";

        public Momentum(double L, double a) 
        {
            this.L = L;
            this.a = a;

            Parameters = new double[] {L,a};
        }

        public override double Update(double w, double gradient)
        {
            //V(t)=γV(t−1)+α.∇J(θ)

            velocity = a * velocity + L * gradient;

            w = w - velocity;

            return w;
        }
    }
}

using mdnn.Activation_functions;
using mdnn.Save_neural_network;
using mdnn.Optimizers;
using mdnn.Activation_functions.classes;

namespace mdnn
{

    public class Neuron
    {
        public double[] Weights
        {
            get { return weights; }
        }
        public double Bias
        {
            get { return bias; }
        }

        
        public double output;
        public double raw_output;
        public double[] inputs;

        public Activation_func activation_func;
        public Optimizer optimizer;

        private double[] weights;
        private double bias;
        
        private double[] gradientsW;
        private double gradientsB;

        private int mini_batch_size = 0;

        public Neuron(int Number_of_input, Activation_func activation_function)
        {
           

            weights = new double[Number_of_input];
            inputs = new double[Number_of_input];
            gradientsW = new double[Number_of_input];

            output = 0;

            for (int i = 0; i < Number_of_input; i++)
            {
                Weights[i] = GeneralNeuralNetworkSettings.rnd.NextDouble() / Number_of_input;
            }

            bias = 0;

            inicializationGradients();
            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            activation_func = activation_function;
        }

        public Neuron(ExportNeuron exportNeuron, Activation_func activation_function)
        {

            weights = exportNeuron.Weights;
            inputs = new double[Weights.Length];
            gradientsW = new double[Weights.Length];

            bias = exportNeuron.Bias;

            output = 0;

            inicializationGradients();
            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            activation_func = activation_function;
        }


        public double feedForward(double[] values)
        {
            inputs = values;

            raw_output = 0;
            for (int i = 0; i < values.Count(); i++)
            {
                raw_output += values[i] * Weights[i];
            }

            raw_output += Bias;
            output = activation_func.Apply(raw_output);

            return output;
        }

        public void backPropagation(double e, double L)
        {

            for (int i = 0; i < Weights.Count(); i++)
            {
                Weights[i] -= L * e * inputs[i];
            }
            bias -= L * e * 1;
        }
        public void Calculate_gradients_of_W_B(double e)
        {

            for (int i = 0; i < Weights.Count(); i++)
            {
                gradientsW[i] += e * inputs[i];
            }
            gradientsB += e ;

            mini_batch_size++;
        }

        public void Update_weights_bias()
        {

            for (int i = 0; i < Weights.Count(); i++)
            {
                Weights[i] = optimizer.Update(Weights[i], gradientsW[i] / mini_batch_size);
            }
            bias = optimizer.Update(bias, gradientsB / mini_batch_size);

            mini_batch_size = 0;

            inicializationGradients();
        }

        public void Mutate_params(int chance_of_mutation, int percent_mutation)
        {
            for (int i = 0; i < Weights.Count(); i++)
            {
                int randomValueBetween0And100 = GeneralNeuralNetworkSettings.rnd.Next(101);
                if (randomValueBetween0And100 < chance_of_mutation)
                {
                    Random random = new Random();
                    // Rozmezí je -percentage/2 až +percentage/2
                    double minPercentage = -percent_mutation / 100.0 / 2.0;
                    double maxPercentage = percent_mutation / 100.0 / 2.0;

                    // Vygeneruj náhodnou hodnotu mezi minPercentage a maxPercentage
                    double randomPercentage = minPercentage + (random.NextDouble() * (maxPercentage - minPercentage));

                    // Změň hodnotu x o náhodné procento
                    Weights[i] += Weights[i] * randomPercentage;
                }
            }
        }

        private void inicializationGradients()
        {
            for (int i = 0; i < gradientsW.Count(); i++)
            {
                gradientsW[i] = 0;
            }
            gradientsB = 0;
        }
    }
}
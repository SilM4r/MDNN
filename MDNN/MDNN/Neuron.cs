using My_DNN.Save_neural_network;
using My_DNN.Optimizers;

namespace My_DNN
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
        
        internal double[] gradientsW;
        private double gradientsB;

        private int mini_batch_size = 0;

        // `random` = zdroj náhody pro inicializaci vah. Předává ho vrstva ze svého
        // NetworkContextu (kvůli reprodukovatelnosti přes seed); null = samostatný neuron
        // bez modelu, spadne se na sdílený globální generátor jako dřív.
        public Neuron(int Number_of_input, Activation_func activation_function, Random? random = null)
        {
            Random rnd = random ?? GeneralNeuralNetworkSettings.rnd;

            weights = new double[Number_of_input];
            inputs = new double[Number_of_input];
            gradientsW = new double[Number_of_input];

            output = 0;

            // limit je pro všechny váhy stejný → počítat ho ve smyčce bylo zbytečné
            double limit = Number_of_input > 0 ? Math.Sqrt(6.0 / Number_of_input) : 0;

            for (int i = 0; i < Number_of_input; i++)
            {
                Weights[i] = (rnd.NextDouble() * 2 - 1) * limit;
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

        // RTRL: gradient parametru = okamžitý gradient shora (gImm) × citlivost tohoto parametru.
        // sensW[i] = ∂h/∂Weights[i], sensB = ∂h/∂bias (obojí drží a posouvá RNN vrstva).
        public void AccumulateGradient(double gImm, double[] sensW, double sensB)
        {
            for (int i = 0; i < Weights.Count(); i++)
            {
                gradientsW[i] += gImm * sensW[i];
            }
            gradientsB += gImm * sensB;

            mini_batch_size++;
        }

        public void Update_weights_bias()
        {
            // Prázdná dávka je legitimní stav (UpdateParams bez předchozího backpropu).
            // Bez tohohle se dělilo nulou → 0/0 = NaN → tiše celý model NaN.
            if (mini_batch_size == 0)
            {
                return;
            }

            for (int i = 0; i < Weights.Count(); i++)
            {
                Weights[i] = optimizer.Update(Weights[i], gradientsW[i] / mini_batch_size,i);
            }
            bias = optimizer.Update(bias, gradientsB / mini_batch_size,Weights.Count());

            mini_batch_size = 0;

            inicializationGradients();
        }

        // `randomSource`: stejná logika jako v konstruktoru — když ho volající nedodá,
        // použije se globální generátor. (Neuron sám Context nevidí, takže si ho musí
        // podat ten, kdo mutaci spouští.)
        public void Mutate_params(int chance_of_mutation, int percent_mutation, Random? randomSource = null)
        {
            Random random = randomSource ?? GeneralNeuralNetworkSettings.rnd;

            for (int i = 0; i < Weights.Count(); i++)
            {
                int randomValueBetween0And100 = random.Next(101);
                if (randomValueBetween0And100 < chance_of_mutation)
                {
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
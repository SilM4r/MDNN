using My_DNN.Activation_functions;
using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;
using My_DNN.Optimizers;

namespace My_DNN.Layers
{

    public class Dense : LayerBasedOnNeurons
    {
        public override List<Neuron> Neurons => neurons;
        public override string Name => "Dense";
        public override Tensor Layer_output => new Tensor(output);
        public override Tensor Layer_raw_output => new Tensor(raw_output);
        public override int[] Input_size_and_shape => input_size;
        public override int[] Output_size_and_shape => new int[] { output.Length };
        public override Activation_func Activation_Func
        {
            get { return activation_func; }
            set
            {
                activation_func = value;
                foreach (Neuron neuron in neurons)
                {
                    neuron.activation_func = activation_func;
                }
            }
        }

        private int[] input_size;

        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;
        private Activation_func activation_func;

        public Dense(int number_of_neuron, Activation_func? activation_func = null)
        {
            if (number_of_neuron <= 0)
            {
                throw new ArgumentException("The number of neurons in a layer must be greater than 0");
            }

            // aktivace: default když nezadaná; správný input size + neurony dopočítá LayerAdjustment (při připojení k modelu)
            this.activation_func = activation_func ?? GeneralNeuralNetworkSettings.default_hidden_layers_activation_func;
            input_size = new int[] { 0 };

            output = new double[number_of_neuron];
            raw_output = new double[number_of_neuron];

            neurons = new List<Neuron>();

            for (int i = 0; i < number_of_neuron; i++)
            {
                neurons.Add(new Neuron(input_size[0], this.activation_func));   // placeholder (0 vah), přepíše LayerAdjustment
            }
        }
        public Dense(ExportDenseLayer layer)
        {
            activation_func = Activation_func.inicialization_activation_func(layer.Name_of_activation_function);
            output = new double[layer.Neurons.Count()];
            raw_output = new double[layer.Neurons.Count()];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in layer.Neurons)
            {
                neurons.Add(new Neuron(neuron, activation_func));
            }

            input_size = new int[] { neurons[0].Weights.Length };

        }
        public override void LayerAdjustment(int? number_of_elements = null, int[]? number_of_input = null)
        {

            if (number_of_input != null)
            {
                if (number_of_input.Length == 1)
                {
                    input_size = number_of_input;
                }

                else
                {
                    input_size = new int[] { 1 };
                    foreach (int input in number_of_input)
                    {
                        input_size[0] *= input;
                    }
                }
            }

            if (number_of_elements != null)
            {
                output = new double[(int)number_of_elements];
                raw_output = new double[(int)number_of_elements];
            }

            // Přestavuj JEN když se tvar opravdu mění. Dřív se `neurons` vždycky zahodily a
            // postavily znovu s náhodnou inicializací — takže i volání, které nic nemění
            // (druhé SetInputSizeForFirstLayer se stejným tvarem, nebo Add() vrstvy se
            // stejným počtem neuronů), tiše smazalo natrénované váhy.
            //
            // Když se tvar MĚNÍ (jiný počet neuronů nebo jiná velikost vstupu), je přestavba
            // nevyhnutelná — mění se délka vektoru vah a staré hodnoty nemají kam sednout.
            if (!NeuronsMatchShape(output.Length, input_size[0]))
            {
                neurons = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    neurons.Add(new Neuron(input_size[0], activation_func, Context?.Random));
                }
            }

            // per-model optimizer: každý neuron dostane klon optimizeru z Contextu (nezávislost modelů).
            // Děje se i při zachovaných vahách — LayerAdjustment znamená přepojení architektury
            // a stav optimizeru (momenty Adamu) se k nové topologii stejně nevztahuje.
            if (Context != null)
                foreach (Neuron n in neurons)
                    n.optimizer = Optimizer.Clone_optimizer(Context.Optimizer);
        }

        // Sedí už postavené neurony přesně na požadovaný tvar?
        private bool NeuronsMatchShape(int neuronCount, int inputSize)
        {
            return neurons.Count == neuronCount
                   && neurons.Count > 0
                   && neurons[0].Weights.Length == inputSize;
        }
        public override Tensor FeedForward(Tensor input_values)
        {

            double[] values = input_values.Data;

            if (Context != null && Context.CalculationViaGpu)
            {
                return FeedForwardViaGpu(values);
            }

            for (int i = 0; i < neurons.Count(); i++)
            {
                output[i] = neurons[i].feedForward(values);
                raw_output[i] = neurons[i].raw_output;
            }

            if (activation_func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = activation_func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                output = layerActivationFunc.ApplyToLayer(raw_output);

                for (int i = 0; i < neurons.Count(); i++)
                {
                    neurons[i].output = output[i];
                }
            }

            return new Tensor(output);
        }
        public override async Task<Tensor> FeedForwardAsync(Tensor input_values)
        {
            double[] values = input_values.Data;

            if (Context != null && Context.CalculationViaGpu)
            {
                return FeedForwardViaGpu(values);
            }

            Task[] feedTasks = new Task[neurons.Count()];
            for (int i = 0; i < neurons.Count(); i++)
            {
                int index = i; // zachycení lokální kopie indexu
                feedTasks[index] = Task.Run(() =>
                {
                    output[index] = neurons[index].feedForward(values);
                    raw_output[index] = neurons[index].raw_output;
                });
            }
            await Task.WhenAll(feedTasks);

            if (activation_func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = activation_func as LayerActivationFunc;
                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                output = layerActivationFunc.ApplyToLayer(raw_output);

                // Aktualizujeme výstupy neuronů paralelně
                Task[] updateTasks = new Task[neurons.Count()];
                for (int i = 0; i < neurons.Count(); i++)
                {
                    int index = i;
                    updateTasks[index] = Task.Run(() =>
                    {
                        neurons[index].output = output[index];
                    });
                }
                await Task.WhenAll(updateTasks);
            }

            return new Tensor(output);
        }
        public override Tensor CalculateLayerGradients(Tensor nextLayerE, Layer nextLayer)
        {
            // Krok 1: gradient shora vůči VÝSTUPU této vrstvy (∂L/∂output), bez derivace aktivace.
            double[] gradFromAbove = GradientFromAbove(nextLayerE.Data, nextLayer);

            // Krok 2: přes derivaci aktivace na ∂L/∂raw_output.
            return new Tensor(ApplyActivationBackward(gradFromAbove));
        }

        // ∂L/∂output_j. Když je další vrstva neuronová, projde se přes její váhy;
        // jinak (Conv/MaxPool) přichází gradient rovnou po prvcích.
        private double[] GradientFromAbove(double[] next_layer_e, Layer nextLayer)
        {
            double[] gradFromAbove = new double[Neurons.Count()];

            LayerBasedOnNeurons? nextlayer = nextLayer as LayerBasedOnNeurons;

            if (nextlayer != null)
            {
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    double de = 0;
                    for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                    {
                        de += next_layer_e[k] * nextlayer.Neurons[k].Weights[j];
                    }
                    gradFromAbove[j] = de;
                }
            }
            else
            {
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    gradFromAbove[j] = next_layer_e[j];
                }
            }

            return gradFromAbove;
        }

        // Aktivace působící na celou vrstvu (softmax) potřebuje CELÝ vektor gradientu —
        // per-prvkové násobení Derivative() by použilo jen diagonálu Jacobiánu a bylo by špatně.
        private double[] ApplyActivationBackward(double[] gradFromAbove)
        {
            if (activation_func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = activation_func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                return layerActivationFunc.BackwardForLayer(raw_output, gradFromAbove);
            }

            double[] e = new double[Neurons.Count()];
            for (int j = 0; j < Neurons.Count(); j++)
            {
                Neuron neuron = Neurons[j];
                e[j] = gradFromAbove[j] * neuron.activation_func.Derivative(neuron.raw_output);
            }
            return e;
        }
        public override async Task<Tensor> CalculateLayerGradientsAsync(Tensor nextLayerE, Layer nextLayer)
        {

            if (nextLayerE.Shape.Length > 1)
            {
                throw new NotImplementedException();
            }

            double[] next_layer_e = nextLayerE.Data;
            double[] gradFromAbove = new double[Neurons.Count()];

            LayerBasedOnNeurons? nextlayer = nextLayer as LayerBasedOnNeurons;
            if (nextlayer != null)
            {
                Task[] tasks = new Task[Neurons.Count()];
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    int index = j;
                    tasks[index] = Task.Run(() =>
                    {
                        double de = 0;
                        for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                        {
                            de += next_layer_e[k] * nextlayer.Neurons[k].Weights[index];
                        }
                        gradFromAbove[index] = de;
                    });
                }
                await Task.WhenAll(tasks);
            }
            else
            {
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    gradFromAbove[j] = next_layer_e[j];
                }
            }

            // stejná konvence jako v synchronní verzi (vrstvová aktivace dostane celý vektor)
            return new Tensor(ApplyActivationBackward(gradFromAbove));
        }
        public override void BackPropagation(Tensor TensorE)
        {
            double[] e = TensorE.Data;

            for (int i = 0; i < neurons.Count(); i++)
            {
                neurons[i].Calculate_gradients_of_W_B(e[i]);
            }
        }
        public override async Task BackPropagationAsync(Tensor TensorE)
        {
            double[] e = TensorE.Data;

            Task[] tasks = new Task[Neurons.Count()];
            for (int i = 0; i < neurons.Count(); i++)
            {
                int index = i;
                tasks[index] = Task.Run(() =>
                {
                    neurons[index].Calculate_gradients_of_W_B(e[index]);
                });
            }
            await Task.WhenAll(tasks);
        }
        public override void UpdateParams()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.Update_weights_bias();
            }
        }
        public override async Task UpdateParamsAsync()
        {
            int index = -1;
            Task[] tasks = new Task[Neurons.Count()];
            foreach (Neuron neuron in neurons)
            {
                index++;
                tasks[index] = Task.Run(() =>
                {
                    neuron.Update_weights_bias();
                });
            }
            await Task.WhenAll(tasks);
        }
        private Tensor FeedForwardViaGpu(double[] values)
        {
            int quantity = neurons.Count();
            float[] weights = new float[quantity * input_size[0]];
            float[] bias = new float[quantity];

            for (int j = 0; j < quantity; j++)
            {
                for (int i = 0; i < input_size[0]; i++)
                {
                    weights[j * input_size[0] + i] = (float)neurons[j].Weights[i];
                }

                bias[j] = (float)neurons[j].Bias;
                neurons[j].inputs = values;
            }

            float[] gpuOutput = new float[quantity];
            float[] gpuvalues = values.Select(d => (float)d).ToArray(); ;

            GPUManager.GPUCalculation(gpuvalues, weights, bias, gpuOutput, input_size[0], quantity);

            raw_output = gpuOutput.Select(d => (double)d).ToArray();

            if (activation_func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = activation_func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                output = layerActivationFunc.ApplyToLayer(raw_output);
            }

            for (int j = 0; j < neurons.Count(); j++)
            {

                neurons[j].raw_output = raw_output[j];
                if (!activation_func.Apply_to_layer)
                {
                    output[j] = activation_func.Apply(raw_output[j]);
                }
                neurons[j].output = output[j];
            }

            return new Tensor(output);
        }
    }
}

using My_DNN.Activation_functions;
using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;

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

            if (activation_func == null)
            {
                if (LayerManager.number_of_penultimate_output_in_Layer[0] == -1)
                {
                    activation_func = GeneralNeuralNetworkSettings.default_output_activation_func;
                }
                else
                {
                    activation_func = GeneralNeuralNetworkSettings.default_hidden_layers_activation_func;
                }

            }
            this.activation_func = activation_func;

            if (LayerManager.number_of_penultimate_output_in_Layer[0] == -1)
            {
                input_size = new int[] { 0 };
            }
            else
            {
                input_size = LayerManager.number_of_penultimate_output_in_Layer;
            }

            if (input_size.Length > 1)
            {
                int[] size = new int[] { 1 };
                foreach (int input in input_size)
                {
                    size[0] *= input;
                }

                input_size = size;
            }

            output = new double[number_of_neuron];
            raw_output = new double[number_of_neuron];

            neurons = new List<Neuron>();

            for (int i = 0; i < number_of_neuron; i++)
            {
                neurons.Add(new Neuron(input_size[0], activation_func));
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

            neurons = new List<Neuron>();

            for (int i = 0; i < output.Length; i++)
            {
                neurons.Add(new Neuron(input_size[0], activation_func));
            }
        }
        public override Tensor FeedForward(Tensor input_values)
        {

            double[] values = input_values.Data;

            if (GeneralNeuralNetworkSettings.calculationViaGpu)
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

            if (GeneralNeuralNetworkSettings.calculationViaGpu)
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

            double[] next_layer_e = nextLayerE.Data;

            double de = 0;
            double[] e = new double[Neurons.Count()];

            LayerBasedOnNeurons? nextlayer = nextLayer as LayerBasedOnNeurons;

            if (nextlayer != null)
            {
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                    {
                        de += next_layer_e[k] * nextlayer.Neurons[k].Weights[j];
                    }

                    Neuron neuron = Neurons[j];

                    de = de * neuron.activation_func.Derivative(neuron.raw_output);
                    e[j] = de;
                    de = 0;
                }
            }
            else
            {
                for(int j = 0;j < Neurons.Count(); j++)
                {
                    Neuron neuron = Neurons[j];
                    e[j] = next_layer_e[j] * neuron.activation_func.Derivative(neuron.raw_output);
                }
            }
            return new Tensor(e);
        }
        public override async Task<Tensor> CalculateLayerGradientsAsync(Tensor nextLayerE, Layer nextLayer)
        {

            if (nextLayerE.Shape.Length > 1)
            {
                throw new NotImplementedException();
            }

            double[] next_layer_e = nextLayerE.Data;
            double[] e = new double[Neurons.Count()];

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
                        Neuron neuron = Neurons[index];
                        de = de * neuron.activation_func.Derivative(neuron.raw_output);
                        e[index] = de;
                    });
                }
                await Task.WhenAll(tasks);
            }
            else
            {
                Task[] tasks = new Task[Neurons.Count()];
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    int index = j;
                    tasks[index] = Task.Run(() =>
                    {
                        Neuron neuron = Neurons[index];
                        e[index] = next_layer_e[index] * neuron.activation_func.Derivative(neuron.raw_output);
                    });
                }
                await Task.WhenAll(tasks);
            }

            return new Tensor(e);
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

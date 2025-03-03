using mdnn.Activation_functions.classes;
using mdnn.Layers.classes;
using mdnn.Save_neural_network;


namespace mdnn.Layers
{
    public class RNN : LayerBasedOnNeurons
    {
        public override List<Neuron> Neurons => neurons;
        public override string Name => "Rnn";

        public override Tensor Layer_output => new Tensor(output);
        public override int[] Input_size_and_shape => input_size;
        public override int[] Output_size_and_shape => new int[] { output.Length };
        public override Tensor Layer_raw_output => new Tensor(raw_output);
        public override Activation_func Activation_Func
        {
            get { return activation_func; }
            set { activation_func = value; }
        }

        private int[] input_size;

        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;
        private Activation_func activation_func;

        private double[] old_layer_e;


        public RNN(ExportRnnLayer layer)
        {
            activation_func = Activation_func.inicialization_activation_func(layer.Name_of_activation_function);
            output = new double[layer.Neurons.Count()];
            old_layer_e = new double[layer.Neurons.Count()];
            raw_output = new double[layer.Neurons.Count()];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in layer.Neurons)
            {
                neurons.Add(new Neuron(neuron, activation_func));
            }

            input_size = new int[] { neurons[0].Weights.Length - 1 };

        }

        public RNN(int number_of_neuron, Activation_func? activation_func = null)
        {
            if (number_of_neuron <= 0)
            {
                throw new ArgumentException("The number of neurons in a layer must be greater than 0");
            }

            if (activation_func == null)
            {
                if (LayerManager.number_of_penultimate_output_in_Layer[0] == 0)
                {
                    activation_func = GeneralNeuralNetworkSettings.output_activation_func;
                }
                else
                {
                    activation_func = GeneralNeuralNetworkSettings.hidden_layers_activation_func;
                }

            }

            input_size = LayerManager.number_of_penultimate_output_in_Layer;

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
            old_layer_e = new double [number_of_neuron];

            this.activation_func = activation_func;
            neurons = new List<Neuron>();

            for (int i = 0; i < number_of_neuron; i++)
            {
                neurons.Add(new Neuron(input_size[0] + 1, activation_func));
            }

        }

        public void ResetSequence()
        {
            old_layer_e = new double[Output_size_and_shape[0]];
            output = new double[Output_size_and_shape[0]];
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
                neurons.Add(new Neuron(input_size[0] + 1, activation_func));
            }
        }

        public override Tensor FeedForward(Tensor TensorValues)
        {

            double[] values = TensorValues.Data;

            double[] newValues = new double[values.Length + 1];


            for (int i = 0; i < values.Length; i++)
            {
                newValues[i] = values[i];
            }

            for (int i = 0; i < neurons.Count(); i++)
            {

                newValues[values.Length] = output[i];

                output[i] = neurons[i].feedForward(newValues);
                raw_output[i] = neurons[i].raw_output;
            }

            if (activation_func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = activation_func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                output = layerActivationFunc.ApplyToLayer(output);

                for (int i = 0; i < neurons.Count(); i++)
                {
                    neurons[i].output = output[i];
                }
            }

            return new Tensor(output);
        }

        public override void BackPropagation(Tensor tenosorDe)
        {
            double[] e = tenosorDe.Data;
            for (int i = 0; i < neurons.Count(); i++)
            {
                neurons[i].Calculate_gradients_of_W_B(e[i]);
            }
        }

        public override void UpdateParams()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.Update_weights_bias();
            }
        }

        public override Tensor CalculateLayerGradients(Tensor nextLayerE, Layer next_layer)
        {

            if (nextLayerE.Shape.Length > 1)
            {
                throw new NotImplementedException();
            }

            double[] next_layer_e = nextLayerE.Data;

            double de = 0;
            double[] e = new double[Neurons.Count()];

            LayerBasedOnNeurons? nextlayer = next_layer as LayerBasedOnNeurons;

            if (nextlayer != null)
            {

                for (int j = 0; j < Neurons.Count(); j++)
                {
                    for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                    {
                        de += next_layer_e[k] * nextlayer.Neurons[k].Weights[j];
                    }

                    Neuron neuron = Neurons[j];

                    de += old_layer_e[j] * neuron.Weights[neuron.Weights.Count() - 1];

                    de = de * neuron.activation_func.Derivative(neuron.raw_output);
                    e[j] = de;

                    old_layer_e[j] = de;
                    de = 0;
                }
            }
            else
            {
                for (int j = 0; j < Neurons.Count(); j++)
                {
                    Neuron neuron = Neurons[j];
                    e[j] = next_layer_e[j] * neuron.activation_func.Derivative(neuron.raw_output);
                }
            }

            return new Tensor(e);
        }

        
    }
}

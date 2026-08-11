using My_DNN.Activation_functions;
using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;
using My_DNN.Optimizers;

namespace My_DNN.Layers
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
            set
            {
                activation_func = value;
                CheckIsActivationFuncIsNotApplyToLayer();
            }
        }

        private int[] input_size;

        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;
        private Activation_func activation_func;

        // RTRL: citlivosti ∂h_i/∂param, akumulované DOPŘEDU v čase
        private double[][] sW;   // sW[neuron][váha]  (poslední váha = rekurentní)
        private double[] sB;     // sB[neuron]        (bias)
        private double[] gImm;   // okamžitý gradient shora vůči h_i (mezi CalcGrad → BackProp)


        public RNN(ExportRnnLayer layer)
        {
            activation_func = Activation_func.inicialization_activation_func(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
            output = new double[layer.Neurons.Count()];
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

            // aktivace: default když nezadaná; input size + neurony dopočítá LayerAdjustment (při připojení k modelu)
            this.activation_func = activation_func ?? GeneralNeuralNetworkSettings.default_hidden_layers_activation_func;
            CheckIsActivationFuncIsNotApplyToLayer();
            input_size = new int[] { 0 };

            output = new double[number_of_neuron];
            raw_output = new double[number_of_neuron];

            neurons = new List<Neuron>();

            for (int i = 0; i < number_of_neuron; i++)
            {
                neurons.Add(new Neuron(input_size[0] + 1, this.activation_func));   // placeholder, přepíše LayerAdjustment
            }

        }

        // Vrstvová aktivace (softmax) na RNN byla rozbitá: ApplyToLayer se aplikovalo na
        // `output`, tedy na hodnoty, které UŽ prošly per-neuronovou aktivací (dvojitá
        // aktivace), zatímco raw_output i RTRL citlivosti zůstaly z předsoftmaxové cesty.
        // Než to dostane vlastní návrh, radši explicitní chyba než tichý nesmysl.
        // Stejná politika jako u Conv (CheckIsActivationFuncIsNotApplyToLayer).
        private void CheckIsActivationFuncIsNotApplyToLayer()
        {
            if (activation_func.Apply_to_layer)
            {
                throw new ArgumentException(
                    "Na RNN vrstvu zatím nelze použít aktivaci působící na celou vrstvu (např. Softmax). " +
                    "Použij např. Tanh() nebo ReLu(); softmax dej na navazující Dense vrstvu.");
            }
        }

        public void ResetSequence()
        {
            // nová sekvence → vynuluj skrytý stav i RTRL citlivosti (realokace = vynulování)
            output = new double[Output_size_and_shape[0]];
            AllocateSensitivities();
        }

        // alokuje (a tím vynuluje) citlivosti podle aktuálních neuronů
        private void AllocateSensitivities()
        {
            sW = new double[neurons.Count][];
            for (int i = 0; i < neurons.Count; i++)
                sW[i] = new double[neurons[i].Weights.Length];
            sB = new double[neurons.Count];
            gImm = new double[neurons.Count];
        }

        // pojistka: citlivosti musí sedět na počet neuronů I na počet vah
        // (neurony se přebudují až ve Fit → SetInputSizeForFirstLayer, po ResetSequence)
        private void EnsureSensitivities()
        {
            if (sW == null || sW.Length != neurons.Count
                || (neurons.Count > 0 && sW[0].Length != neurons[0].Weights.Length))
                AllocateSensitivities();
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

            // Stejná idempotence jako u Dense — bez ní i volání, které nic nemění, smazalo
            // natrénované váhy. Pozor: RNN neuron má o jednu váhu navíc (rekurentní vstup).
            if (!NeuronsMatchShape(output.Length, input_size[0] + 1))
            {
                neurons = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    neurons.Add(new Neuron(input_size[0] + 1, activation_func));
                }
            }

            // per-model optimizer (nezávislost modelů)
            if (Context != null)
                foreach (Neuron n in neurons)
                    n.optimizer = Optimizer.Clone_optimizer(Context.Optimizer);
        }

        private bool NeuronsMatchShape(int neuronCount, int weightsPerNeuron)
        {
            return neurons.Count == neuronCount
                   && neurons.Count > 0
                   && neurons[0].Weights.Length == weightsPerNeuron;
        }

        public override Tensor FeedForward(Tensor TensorValues)
        {

            double[] values = TensorValues.Data;

            double[] newValues = new double[values.Length + 1];


            for (int i = 0; i < values.Length; i++)
            {
                newValues[i] = values[i];
            }

            EnsureSensitivities();

            for (int i = 0; i < neurons.Count(); i++)
            {
                newValues[values.Length] = output[i];   // rekurentní vstup = h_i(t-1) = předchozí výstup

                output[i] = neurons[i].feedForward(newValues);
                raw_output[i] = neurons[i].raw_output;   // z_i(t)

                // RTRL: posuň citlivosti o krok dopředu
                double actD = neurons[i].activation_func.Derivative(raw_output[i]);   // act'(z_i)
                double r = neurons[i].Weights[newValues.Length - 1];                  // rekurentní váha
                for (int k = 0; k < newValues.Length; k++)                            // ∂z/∂w_k = newValues[k]
                    sW[i][k] = actD * (newValues[k] + r * sW[i][k]);
                sB[i] = actD * (1.0 + r * sB[i]);                                     // ∂z/∂bias = 1
            }

            return new Tensor(output);
        }

        // RNN jako VÝSTUPNÍ vrstva: gImm se jinak nemá kde vzít (CalculateLayerGradients
        // se pro poslední vrstvu nevolá). BEZ derivace aktivace — RTRL citlivosti sW/sB
        // ji už obsahují (viz FeedForward: sW[i][k] = actD · (...)), takže by se aplikovala
        // dvakrát. Stejná konvence jako gImm[j] v CalculateLayerGradients.
        public override void SeedOutputGradient(double[] dLossDOutput)
        {
            EnsureSensitivities();

            if (dLossDOutput.Length != neurons.Count)
            {
                throw new ArgumentException(
                    $"Gradient shora má {dLossDOutput.Length} prvků, vrstva {neurons.Count} neuronů.");
            }

            for (int i = 0; i < neurons.Count; i++)
            {
                gImm[i] = dLossDOutput[i];
            }
        }

        public override void BackPropagation(Tensor tenosorDe)
        {
            // RTRL: gradient parametru = g_imm(t) · citlivost(t), sečteno přes čas.
            // Argument se nepoužije — pracujeme s uloženým gImm + citlivostmi (jako Conv s dOutput).
            for (int i = 0; i < neurons.Count(); i++)
            {
                neurons[i].AccumulateGradient(gImm[i], sW[i], sB[i]);
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
            double[] e = new double[Neurons.Count()];

            LayerBasedOnNeurons? nextlayer = next_layer as LayerBasedOnNeurons;

            for (int j = 0; j < Neurons.Count(); j++)
            {
                // g_imm_j = OKAMŽITÝ gradient shora vůči h_j (dL_t/dh_j; bez rekurence, bez act')
                double gImmJ;
                if (nextlayer != null)
                {
                    gImmJ = 0;
                    for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                        gImmJ += next_layer_e[k] * nextlayer.Neurons[k].Weights[j];
                }
                else
                {
                    gImmJ = next_layer_e[j];
                }

                gImm[j] = gImmJ;   // ulož pro BackPropagation (RTRL: dParam += g_imm · citlivost)

                // vrať OKAMŽITOU deltu (g_imm · act') pro předchozí vrstvu
                e[j] = gImmJ * Neurons[j].activation_func.Derivative(Neurons[j].raw_output);
            }

            return new Tensor(e);
        }

        
    }
}

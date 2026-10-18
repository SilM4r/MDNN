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

        // Snímek, ne okno do pracovního pole — viz Tensor.FromVector.
        public override Tensor Layer_output => Tensor.FromVector(output);
        public override int[] Input_size_and_shape => input_size;
        public override int[] Output_size_and_shape => new int[] { output.Length };
        public override Tensor Layer_raw_output => Tensor.FromVector(raw_output);
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

        public override void WireShapes(int? number_of_elements = null, int[]? number_of_input = null)
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

        }

        // Pozor: RNN neuron má o jednu váhu navíc (rekurentní vstup h(t-1)).
        public override bool IsMaterialized =>
            input_size[0] > 0 && NeuronsMatchShape(output.Length, input_size[0] + 1);

        public override void MaterializeParameters()
        {
            if (input_size[0] <= 0)
            {
                return;
            }

            if (!NeuronsMatchShape(output.Length, input_size[0] + 1))
            {
                // do `neurons` až hotové — viz Dense.MaterializeParameters
                var rebuilt = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    rebuilt.Add(new Neuron(input_size[0] + 1, activation_func, Context?.Random));
                }

                // per-model optimizer jen pro čerstvé neurony — viz Dense.MaterializeParameters
                if (Context != null)
                {
                    foreach (Neuron n in rebuilt)
                    {
                        n.optimizer = Optimizer.Clone_optimizer(Context.Optimizer);
                    }
                }

                neurons = rebuilt;
            }
        }

        // Replika pro paralelní trénink — viz Layer.CreateReplica. Kromě výstupů má vlastní
        // i skrytý stav a RTRL citlivosti: replika zpracovává CELÉ sekvence, takže stav
        // mezi kroky sekvence zůstává v jedné replice a nemíchá se s jinou sekvencí.
        internal override Layer CreateReplica()
        {
            RNN replica = (RNN)MemberwiseClone();
            replica.output = new double[output.Length];
            replica.raw_output = new double[raw_output.Length];
            replica.neurons = neurons.Select(n => n.CreateReplica()).ToList();
            replica.AllocateSensitivities();
            return replica;
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

            for (int i = 0; i < neurons.Count; i++)
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

            return Tensor.FromVector(output);
        }

        // ---------------------------------------------------------------- konvence B

        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "RNN nesmí být výstupní vrstvou se softmaxem (guard v Gradient), " +
                    "takže fúze softmax+CE se jí netýká.");

            EnsureSensitivities();

            double[] gradOut = gradOutput.Data;

            if (gradOut.Length != neurons.Count)
                throw new ArgumentException(
                    $"Gradient shora má {gradOut.Length} prvků, vrstva {neurons.Count} neuronů.");

            // TADY je vidět, proč konvence B RNN sedí LÍP než dnešek: `gImm` je definičně
            // ∂L/∂h_j BEZ derivace aktivace — a to je přesně to, co Backward dostává.
            // Ve staré cestě existovalo jako SKRYTÉ POLE jen proto, že si ho musely předat
            // dvě metody (CalculateLayerGradients → BackPropagation). Tady je to lokálka.
            //
            // Pozn.: rekurentní váha se do gradientu vstupu nepromítá — příspěvek zpět
            // v čase RTRL nezachycuje. Je to známé omezení, ne opomenutí (viz README).
            int inputCount = neurons[0].Weights.Length - 1;   // poslední váha je rekurentní

            // Když volající vstupní gradient nechce, ani ho nealokujeme — a hlavně
            // vrátíme PRÁZDNÝ tenzor, ne pole nul. Kontrakt v Layer.cs to říká výslovně:
            // omyl má spadnout na délce, ne až na tichém špatném výsledku. RNN se jako
            // jediná ze čtyř vrstev odchylovala.
            double[] gradInput = computeInputGradient ? new double[inputCount] : Array.Empty<double>();

            for (int j = 0; j < neurons.Count; j++)
            {
                double gImmJ = gradOut[j];

                // RTRL: gradient parametru = g_imm(t) · citlivost(t), sečteno přes čas.
                // Citlivosti sW/sB derivaci aktivace UŽ obsahují (advancují se ve forwardu),
                // proto se tady act' NEpřidává — na rozdíl od Dense.
                neurons[j].AccumulateGradient(gImmJ, sW[j], sB[j]);

                // Do vrstvy pod sebou ale act' patří: ∂L/∂z_j = g_imm_j · act'(z_j).
                if (!computeInputGradient) continue;

                double delta = gImmJ * neurons[j].activation_func.Derivative(neurons[j].raw_output);
                if (delta == 0.0) continue;

                double[] w = neurons[j].Weights;
                for (int k = 0; k < inputCount; k++) gradInput[k] += delta * w[k];
            }

            return computeInputGradient ? new Tensor(gradInput) : Tensor.Empty;
        }

        public override void UpdateParams()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.Update_weights_bias(Context?.L2 ?? 0);
            }
        }

    }
}

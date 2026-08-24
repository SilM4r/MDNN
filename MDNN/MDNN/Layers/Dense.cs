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

        // ---------------------------------------------------------------- dropout

        private double? dropout;

        // Maska z posledního forwardu: 0 = neuron byl vypnutý, 1/(1-p) = přežil,
        // 1 = dropout se neuplatnil (predikce nebo rate 0). Backward MUSÍ použít TUTÉŽ
        // masku, proto se drží mezi forwardem a backwardem, negeneruje se znovu.
        private double[]? dropoutMask;

        /// <summary>
        /// Dropout pro TUTO vrstvu. <c>null</c> (výchozí) = použij globální
        /// <see cref="NetworkContext.Dropout"/>. Vlastní hodnota globální přebije —
        /// včetně nuly, kterou se dropout na téhle vrstvě vypne.
        /// </summary>
        public double? DropoutRate
        {
            get => dropout;
            set
            {
                if (value != null) ValidateDropout(value.Value);
                dropout = value;
            }
        }

        private static void ValidateDropout(double value)
        {
            if (double.IsNaN(value) || value < 0 || value >= 1)
                throw new ArgumentOutOfRangeException(nameof(DropoutRate), value,
                    "Dropout musí být v intervalu <0, 1). 0 = vypnuto; 1 by zahodilo " +
                    "všechny neurony a škálovací faktor 1/(1-p) by dělil nulou.");
        }

        /// <summary>Kolik se doopravdy zahazuje: vlastní hodnota, jinak globální —
        /// a na výstupní vrstvě se globální ZÁMĚRNĚ ignoruje (viz NetworkContext.Dropout).</summary>
        private double EffectiveDropout
        {
            get
            {
                if (dropout != null) return dropout.Value;
                if (IsOutputLayer) return 0.0;
                return Context?.Dropout ?? 0.0;
            }
        }

        /// <param name="dropout">Podíl neuronů zahozených při tréninku, 0 ≤ p &lt; 1.
        /// null = zdědit globální <see cref="NetworkContext.Dropout"/>.</param>
        public Dense(int number_of_neuron, Activation_func? activation_func = null, double? dropout = null)
        {
            if (number_of_neuron <= 0)
            {
                throw new ArgumentException("The number of neurons in a layer must be greater than 0");
            }

            if (dropout != null) ValidateDropout(dropout.Value);
            this.dropout = dropout;

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

            // Starší soubory tenhle klíč nemají → null → vrstva se vrátí k dědění globální
            // hodnoty, což je i výchozí chování nového modelu. Zpětně kompatibilní.
            if (layer.DropoutRate != null) ValidateDropout(layer.DropoutRate.Value);
            dropout = layer.DropoutRate;

            output = new double[layer.Neurons.Count()];
            raw_output = new double[layer.Neurons.Count()];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in layer.Neurons)
            {
                neurons.Add(new Neuron(neuron, activation_func));
            }

            input_size = new int[] { neurons[0].Weights.Length };

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

        public override bool IsMaterialized =>
            input_size[0] > 0 && NeuronsMatchShape(output.Length, input_size[0]);

        public override void MaterializeParameters()
        {
            if (input_size[0] <= 0)
            {
                return;   // tvar vstupu ještě neznáme, není z čeho stavět
            }

            // Přestavuj JEN když se tvar opravdu mění. Jinak by i volání, které nic nemění,
            // tiše smazalo natrénované váhy. Když se tvar MĚNÍ, je přestavba nevyhnutelná —
            // mění se délka vektoru vah a staré hodnoty nemají kam sednout.
            if (!NeuronsMatchShape(output.Length, input_size[0]))
            {
                neurons = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    neurons.Add(new Neuron(input_size[0], activation_func, Context?.Random));
                }
            }

            // per-model optimizer: každý neuron dostane klon optimizeru z Contextu (nezávislost
            // modelů). Děje se i při zachovaných vahách — přepojení architektury znamená, že se
            // stav optimizeru (momenty Adamu) k nové topologii stejně nevztahuje.
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

            // Masku je potřeba znát PŘED výpočtem, aby se vypnuté neurony daly rovnou
            // přeskočit — u p=0.5 to ušetří polovinu skalárních součinů této vrstvy.
            PrepareDropoutMask();

            // Softmax počítá přes CELÝ vektor raw_output, takže se z něj nesmí nic
            // vynechat — jinak by se normalizovalo přes zastaralá čísla. Tam se tedy
            // spočítá všechno a maska se uplatní až na hotový výstup.
            bool canSkip = dropoutMask != null && !activation_func.Apply_to_layer;

            for (int i = 0; i < neurons.Count(); i++)
            {
                if (canSkip && dropoutMask![i] == 0.0)
                {
                    // Vypnutý neuron se NEPOČÍTÁ. Nuly se nastavují explicitně, ať v polích
                    // nezůstanou hodnoty z minulého vzorku.
                    output[i] = 0;
                    raw_output[i] = 0;
                    neurons[i].output = 0;
                    neurons[i].raw_output = 0;
                    continue;
                }

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

            ApplyDropoutMask();

            return new Tensor(output);
        }

        // Vylosuje masku pro tento vzorek. null = dropout se neuplatní vůbec (vypnutý,
        // nebo běží predikce) — pak se nikde nenásobí a nealokuje.
        // Testovací můstek. Gradient check volá forward opakovaně a porovnává, jak se mění
        // loss — to má smysl jen tehdy, když je to POKAŽDÉ TÁŽ funkce. S losovanou maskou
        // by se pod rukama měnila a kontrola by neměřila nic. Zmrazením se z dropoutu
        // stane obyčejné násobení konstantním vektorem, které se ověřit dá.
        internal bool FreezeDropoutMaskForTests { get; set; }

        internal double[]? DropoutMaskForTests => dropoutMask;

        private void PrepareDropoutMask()
        {
            double rate = EffectiveDropout;
            bool training = Context?.IsTraining ?? false;

            if (rate == 0.0 || !training)
            {
                dropoutMask = null;
                return;
            }

            if (FreezeDropoutMaskForTests && dropoutMask != null
                && dropoutMask.Length == neurons.Count())
            {
                return;
            }

            if (dropoutMask == null || dropoutMask.Length != neurons.Count())
            {
                dropoutMask = new double[neurons.Count()];
            }

            // Zesílení přeživších drží střední hodnotu výstupu stejnou jako bez dropoutu
            // („inverted dropout"). Díky tomu predikce nemusí nic dopočítávat.
            double scale = 1.0 / (1.0 - rate);
            Random random = Context!.Random;

            for (int i = 0; i < dropoutMask.Length; i++)
            {
                dropoutMask[i] = random.NextDouble() < rate ? 0.0 : scale;
            }
        }

        // Dopočet pro neurony, které se spočítaly (skip je nechal na nule): vynulovat
        // zahozené a zesílit přeživší.
        private void ApplyDropoutMask()
        {
            if (dropoutMask == null) return;

            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= dropoutMask[i];
                neurons[i].output = output[i];
            }
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

            // Krok 2: dropout. Gradient prochází TOUTÉŽ maskou jako forward — vypnutý
            // neuron gradient nedostane, přeživší dostane zesílený stejným faktorem.
            // Musí to být PŘED derivací aktivace, protože maska působí na výstup vrstvy.
            if (dropoutMask != null)
            {
                for (int i = 0; i < gradFromAbove.Length; i++)
                {
                    gradFromAbove[i] *= dropoutMask[i];
                }
            }

            // Krok 3: přes derivaci aktivace na ∂L/∂raw_output.
            return new Tensor(ApplyActivationBackward(gradFromAbove));
        }

        // ---------------------------------------------------------------- konvence B

        public override Tensor Backward(Tensor gradOutput, bool preActivation = false)
        {
            // Kopie: gradOutput patří volajícímu, nesmí se mu přepsat pod rukama.
            double[] grad = (double[])gradOutput.Data.Clone();

            if (grad.Length != neurons.Count)
                throw new ArgumentException(
                    $"Dense má {neurons.Count} neuronů, ale gradient shora má {grad.Length} hodnot.");

            // 1) dropout — TÁŽ maska jako ve forwardu, a PŘED derivací aktivace,
            //    protože maska působí na výstup vrstvy.
            if (dropoutMask != null)
            {
                for (int i = 0; i < grad.Length; i++) grad[i] *= dropoutMask[i];
            }

            // 2) ∂L/∂output → ∂L/∂z. Při fúzi softmax+CE je `grad` už ∂L/∂z (s − t)
            //    a aktivace se přeskakuje, jinak by se softmax derivace použila dvakrát.
            double[] delta = preActivation ? grad : ApplyActivationBackward(grad);

            // 3) gradienty vlastních parametrů
            for (int i = 0; i < neurons.Count; i++)
            {
                if (dropoutMask != null && dropoutMask[i] == 0.0)
                {
                    neurons[i].SkipSample();
                    continue;
                }

                neurons[i].Calculate_gradients_of_W_B(delta[i]);
            }

            // 4) ∂L/∂input — JEDINÁ věc, kterou dnešní Dense nedělá. Není to nový výpočet,
            //    jen se přesouvá tam, kam patří: dnes tuhle smyčku píše vrstva POD Dense
            //    (Conv/MaxPool/Dense v GradientFromAbove) a sahá si přitom do CIZÍCH vah.
            int inputs = neurons[0].Weights.Length;
            double[] gradInput = new double[inputs];

            for (int i = 0; i < neurons.Count; i++)
            {
                double d = delta[i];
                if (d == 0.0) continue;          // vypnutý dropoutem → nepřispívá

                double[] w = neurons[i].Weights;
                for (int j = 0; j < inputs; j++) gradInput[j] += d * w[j];
            }

            return new Tensor(gradInput);
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
        public override void BackPropagation(Tensor TensorE)
        {
            double[] e = TensorE.Data;

            for (int i = 0; i < neurons.Count(); i++)
            {
                // Vypnutý neuron má gradient přesně nula, takže smyčka přes jeho váhy by
                // jen přičítala nuly. Přeskočí se — ale čítač dávky se zvýšit MUSÍ, jinak
                // by se jeho gradienty dělily menším číslem než u ostatních neuronů.
                if (dropoutMask != null && dropoutMask[i] == 0.0)
                {
                    neurons[i].SkipSample();
                    continue;
                }

                neurons[i].Calculate_gradients_of_W_B(e[i]);
            }
        }
        public override void UpdateParams()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.Update_weights_bias(Context?.L2 ?? 0);
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
                    neuron.Update_weights_bias(Context?.L2 ?? 0);
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

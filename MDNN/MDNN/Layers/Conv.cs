using My_DNN.Layers.classes;
using My_DNN.Optimizers;
using My_DNN.Save_neural_network;

namespace My_DNN.Layers
{
    public class Conv : Layer
    {
        public override string Name => "CNN";

        public override int[] Input_size_and_shape => inputsShape;

        public override int[] Output_size_and_shape => outputShape;

        // Tentýž tvar jako vrací FeedForward. Dřív tu bylo `new Tensor(output)` nad jagged
        // polem: `double[][][]` má Rank 1, takže Shape vyšel jako [outH] místo [h,w,kanály].
        // Data byla správně, Shape ne — a projeví se to jen tehdy, když je Conv výstupní
        // vrstva a někdo si Shape přečte. Navíc to bylo okno do pracovního pole, ne snímek.
        public override Tensor Layer_output => Tensor.FromJagged3D(output, outputShape);

        public override Tensor Layer_raw_output => Tensor.FromJagged3D(raw_output, outputShape);

        public override Activation_func Activation_Func
        {
            get { return activation_func; }
            set 
            {
                activation_func = value;
                CheckIsActivationFuncIsNotApplyToLayer();
            }
        }

        public double[][][][] Kernel  // [numFilters][kernelRows][kernelCols][inputChannels]
        {
            get { return kernels; }
        }
        public double[] Biases
        {
            get { return biases; }
        }
        public string Padding
        {
            get { return padding; }
        }

        // Prázdné výchozí hodnoty: buffery se alokují až při zapojení (WireShapes), kernely
        // při materializaci. Všechny konstruktory je přepíšou; prázdné pole je tu jen proto,
        // aby pole nikdy nebylo null.
        private double[][][][] kernels = Array.Empty<double[][][]>();
        internal double[][][][] dKernels = Array.Empty<double[][][]>();   // internal kvůli gradient-check testům
        private double[] biases = Array.Empty<double>();
        internal double[] dBiases = Array.Empty<double>();                // internal kvůli gradient-check testům
        private double[][][] dOutput = Array.Empty<double[][]>();
        private string padding;
        private double[][][] output = Array.Empty<double[][]>();
        private double[][][] raw_output = Array.Empty<double[][]>();
        private double[][][] inputs = Array.Empty<double[][]>();

        private int[] inputsShape;
        private int[] outputShape;

        private int mini_batch_size;
        private Optimizer optimizer;

        // Pro uložení stavu optimizeru do souboru (ExportCNNLayer).
        internal Optimizer OptimizerForSave => optimizer;

        internal override void ReplaceOptimizer(Optimizer prototype)
        {
            optimizer = Optimizer.Clone_optimizer(prototype);
        }

        // Kolik filtrů si vyžádalo poslední WireShapes; MaterializeParameters podle toho staví.
        private int? requestedKernelCount;

        private Activation_func activation_func;

        public Conv(ExportCNNLayer layer)
        {
            kernels = BaseExportLayer.Required(layer.Kernel, Name, nameof(layer.Kernel));
            biases = BaseExportLayer.Required(layer.Biases, Name, nameof(layer.Biases));
            activation_func = Activation_func.inicialization_activation_func(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
            padding = BaseExportLayer.Required(layer.Padding, Name, nameof(layer.Padding));

            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);

            // navázání tréninku — viz Neuron(ExportNeuron, …)
            if (layer.OptimizerState != null)
            {
                optimizer.ImportState(layer.OptimizerState);
            }

            inputsShape = BaseExportLayer.Required(layer.InputShape, Name, nameof(layer.InputShape));

            if (padding.ToLower() == "same")
            {
                outputShape = new int[] { inputsShape[0], inputsShape[1], kernels.Length };
            }
            else
            {
                outputShape = new int[] { inputsShape[0] - kernels[0].Length + 1, inputsShape[1] - kernels[0][0].Length + 1, kernels.Length };
            }

            inputs = new double[inputsShape[0]][][];
            for (int i = 0; i < inputsShape[0]; i++)
            {
                inputs[i] = new double[inputsShape[1]][];
                for (int j = 0; j < inputsShape[1]; j++)
                {
                    inputs[i][j] = new double[inputsShape[2]];
                }
            }

            output = new double[outputShape[0]][][];
            dOutput = new double[outputShape[0]][][];
            raw_output = new double[outputShape[0]][][];
            for (int i = 0; i < outputShape[0]; i++)
            {
                output[i] = new double[outputShape[1]][];
                raw_output[i] = new double[outputShape[1]][];
                dOutput[i] = new double[outputShape[1]][];
                for (int j = 0; j < outputShape[1]; j++)
                {
                    output[i][j] = new double[outputShape[2]];
                    dOutput[i][j] = new double[outputShape[2]];
                    raw_output[i][j] = new double[outputShape[2]];
                }
            }
            dBiases = new double[kernels.Count()];
            dKernels = new double[kernels.Count()][][][];

            for (int i = 0; i < kernels.Count(); i++)
            {
                dKernels[i] = new double[kernels[0].Count()][][];
                for (int j = 0; j < kernels[0].Count(); j++)
                {
                    dKernels[i][j] = new double[kernels[0][0].Count()][];
                    for (int k = 0; k < kernels[0][0].Count(); k++)
                    {
                        dKernels[i][j][k] = new double[kernels[0][0][0].Count()];
                    }
                }
            }

            inicializationK_B_dK_dB();

            mini_batch_size = 0;
        }

        public Conv(int number_of_kernels,int kernel_size, Activation_func? activation_func = null, string padding = "valid")
        {
            this.activation_func = activation_func ?? GeneralNeuralNetworkSettings.default_hidden_layers_activation_func;
            CheckIsActivationFuncIsNotApplyToLayer();

            inputsShape = new int[] { 0 };
            outputShape = new int[] { 0 };
            this.padding = padding;

            // vstupní kanály = 0 (placeholder); správné kanály + kernely dopočítá LayerAdjustment
            inicializationK_B_dK_dB(number_of_kernels, new int[] { kernel_size, kernel_size, 0 }, true);

            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            mini_batch_size = 0;

        }

        public override void WireShapes(int? number_of_kernels = null, int[]? input_Shape = null)
        {
            // Tvar vstupu ještě není známý (Insert/Add před SetInputSizeForFirstLayer předá
            // placeholder {0}). Dřív se to protlačilo do ConvertTo2D → rows = ceil(sqrt(0)) = 0
            // → DivideByZeroException z hloubi knihovny. Necháme vrstvu placeholderem;
            // SetInputSizeForFirstLayer ji doplní, až tvar bude znám.
            if (number_of_kernels != null)
            {
                requestedKernelCount = number_of_kernels;
            }

            if (input_Shape != null && input_Shape.Any(d => d <= 0))
            {
                inputsShape = new int[] { 0 };
                outputShape = new int[] { 0 };
                return;
            }

            if (input_Shape != null)
            {
                // Tentýž normalizátor, jaký použije FeedForward. Dřív to byly dvě různé
                // implementace: tady se uměl rank 1 a 3, ve forwardu navíc rank 2 — takže
                // model s 2D vstupem se nedal zapojit, přestože průchod by ho zvládl.
                inputsShape = SpatialShape.Normalize(input_Shape, Name);

                if (padding.ToLower() == "same")
                {
                    outputShape = new int[] { inputsShape[0], inputsShape[1], kernels.Length };
                }
                else
                {
                    outputShape = new int[] { inputsShape[0] - kernels[0].Length + 1, inputsShape[1] - kernels[0][0].Length + 1, kernels.Length };
                }

                inputs = new double[inputsShape[0]][][];
                for (int i = 0; i < inputsShape[0]; i++)
                {
                    inputs[i] = new double[inputsShape[1]][];
                    for (int j = 0; j < inputsShape[1]; j++)
                    {
                        inputs[i][j] = new double[inputsShape[2]];
                    }
                }

                output = new double[outputShape[0]][][];
                dOutput = new double[outputShape[0]][][];
                raw_output = new double[outputShape[0]][][];
                for (int i = 0; i < outputShape[0]; i++)
                {
                    output[i] = new double[outputShape[1]][];
                    raw_output[i] = new double[outputShape[1]][];
                    dOutput[i] = new double[outputShape[1]][];
                    for (int j = 0; j < outputShape[1]; j++)
                    {
                        output[i][j] = new double[outputShape[2]];
                        dOutput[i][j] = new double[outputShape[2]];
                        raw_output[i][j] = new double[outputShape[2]];
                    }
                }
            }


        }

        public override bool IsMaterialized =>
            inputsShape.Length == 3
            && inputsShape[2] > 0
            && kernels.Length == (requestedKernelCount ?? kernels.Length)
            && kernels[0][0][0].Length == inputsShape[2];

        public override void MaterializeParameters()
        {
            if (inputsShape.Length != 3 || inputsShape[2] <= 0)
            {
                return;   // tvar vstupu ještě neznáme
            }

            int filters = requestedKernelCount ?? kernels.Length;

            // Kernely znovu inicializovat JEN když se mění jejich tvar (počet filtrů nebo
            // počet vstupních kanálů). Jinak by i volání, které nic nemění, přepsalo
            // natrénované kernely náhodnými čísly. ChangeKernel: false jen vynuluje
            // dKernels/dBiases a kernely+biasy nechá být.
            bool kernelShapeMatches = kernels.Length == filters
                                      && kernels[0][0][0].Length == inputsShape[2];

            // per-model optimizer jen když se kernely opravdu stavěly znovu — viz
            // Dense.MaterializeParameters. Klonování při každé materializaci zahazovalo
            // momenty Adamu i vrstvám, kterých se přepojení netýkalo.
            //
            // Klonuje se PŘED přestavbou kernelů: když klonování spadne, vrstva zůstane,
            // jak byla, a ne s novými kernely a starým optimizerem.
            Optimizer? fresh = Context != null && !kernelShapeMatches
                ? Optimizer.Clone_optimizer(Context.Optimizer)
                : null;

            inicializationK_B_dK_dB(
                filters,
                new int[] { Kernel[0].Count(), Kernel[0][0].Count(), inputsShape[2] },
                !kernelShapeMatches);

            if (fresh != null)
            {
                optimizer = fresh;
            }
        }

        // Replika pro paralelní trénink — viz Layer.CreateReplica. Kernely i biasy SDÍLÍ
        // odkazem (optimizer originálu je přepisuje na místě, až po dávce), takže se mezi
        // dávkami nic nekopíruje. Vlastní: vstupní a výstupní buffery (alokuje je WireShapes
        // stejně jako při zapojení) a akumulátory gradientů.
        internal override Layer CreateReplica()
        {
            Conv replica = (Conv)MemberwiseClone();
            replica.WireShapes(null, inputsShape);

            replica.dBiases = new double[kernels.Length];
            replica.dKernels = new double[kernels.Length][][][];
            for (int f = 0; f < kernels.Length; f++)
            {
                replica.dKernels[f] = new double[kernels[f].Length][][];
                for (int i = 0; i < kernels[f].Length; i++)
                {
                    replica.dKernels[f][i] = new double[kernels[f][i].Length][];
                    for (int j = 0; j < kernels[f][i].Length; j++)
                    {
                        replica.dKernels[f][i][j] = new double[kernels[f][i][j].Length];
                    }
                }
            }

            replica.mini_batch_size = 0;
            return replica;
        }

        // Přičíst gradienty repliky a repliku vynulovat. Čítač dávky patří k tomu —
        // UpdateParams jím dělí (viz Neuron.TakeGradientsFrom).
        internal override void TakeGradientsFrom(Layer replica)
        {
            Conv other = (Conv)replica;

            for (int f = 0; f < dKernels.Length; f++)
            {
                for (int i = 0; i < dKernels[f].Length; i++)
                {
                    for (int j = 0; j < dKernels[f][i].Length; j++)
                    {
                        double[] mine = dKernels[f][i][j];
                        double[] theirs = other.dKernels[f][i][j];
                        for (int c = 0; c < mine.Length; c++)
                        {
                            mine[c] += theirs[c];
                            theirs[c] = 0;
                        }
                    }
                }

                dBiases[f] += other.dBiases[f];
                other.dBiases[f] = 0;
            }

            mini_batch_size += other.mini_batch_size;
            other.mini_batch_size = 0;
        }

        public override Tensor FeedForward(Tensor TensorValues)
        {
            // Tvar musí sedět na ten, na který je vrstva ZAPOJENÁ. Dřív se to nikde
            // neporovnávalo: kopírovací smyčka níž se řídila tvarem příchozího tenzoru,
            // ale konvoluce pak jela podle inputsShape/outputShape. Menší vstup proto
            // přepsal jen svůj roh a zbytek se dopočítal z toho, co v bufferu zbylo
            // po minulém vzorku — tiše, bez výjimky.
            //
            // Tvar se normalizuje jen PRO TUHLE METODU. Dřív se místo toho volalo
            // TensorValues.Reshape(...), čímž vrstva přepsala tvar tenzoru, který jí dal
            // volající — po průchodu modelem měl uživatel v ruce jiný Shape, než jaký zadal.
            int[] shape = SpatialShape.RequireWiredShape(TensorValues.Shape, inputsShape, Name);

            // Čte se rovnou z plochých dat. Dřív tu bylo (double[,,])GetOriginalData(),
            // což si vynutilo postavit multidim kopii přes Array.CreateInstance + SetValue
            // po prvcích — reflexe a boxing na každý forward. Row-major pořadí Data
            // odpovídá [i,j,k], takže stačí jeden běžící index.
            double[] inputValues = TensorValues.Data;

            int flatIndex = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        inputs[i][j][k] = inputValues[flatIndex++];
                    }
                }
            }

            int inputRows = inputs.Length;
            int inputCols = inputs[0].Length;
            int inputChannels = inputs[0][0].Length;
            int numFilters = kernels.Length;
            int kernelRows = kernels[0].Length;
            int kernelCols = kernels[0][0].Length;
            int kernelChannels = kernels[0][0][0].Length;

            if (kernelChannels != inputChannels)
            {
                throw new ArgumentException("The number of input channels and kernel channels must match.");
            }

            double[][][] paddedInput = PadForSame(inputs);
            

            int outH = outputShape[0];
            int outW = outputShape[1];

            // Pro každý filtr (výstupní kanál)
            //
            // Řádky zubatých polí se vytahují o úroveň výš (inRow, kRow, pixel, weights), ne až
            // v nejvnitřnější smyčce: `paddedInput[i + ki][j + kj][c]` znamená tři skoky po
            // haldě a tři kontroly mezí na KAŽDÉ násobení. Pořadí sčítání je stejné, takže
            // výsledek je bit po bitu tentýž.
            for (int f = 0; f < numFilters; f++)
            {
                double[][][] kernel = kernels[f];

                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        double sum = 0;
                        for (int ki = 0; ki < kernelRows; ki++)
                        {
                            double[][] inRow = paddedInput[i + ki];
                            double[][] kRow = kernel[ki];

                            for (int kj = 0; kj < kernelCols; kj++)
                            {
                                double[] pixel = inRow[j + kj];
                                double[] weights = kRow[kj];

                                for (int c = 0; c < inputChannels; c++)
                                {
                                    sum += pixel[c] * weights[c];
                                }
                            }
                        }
                        // Přičteme bias pro filtr f
                        double result = sum + biases[f];
                        output[i][j][f] = activation_func.Apply(result);
                        raw_output[i][j][f] = result;
                    }
                }
            }

            return Tensor.FromJagged3D(output, outputShape);
        }

        // Gradienty kernelů a biasů z už naplněného `dOutput`. Sdílí to stará cesta
        // (BackPropagation) i nová (Backward), takže se ten výpočet nikde neduplikuje.
        private void AccumulateKernelAndBiasGradients()
        {
            mini_batch_size++;

            int inChannels = inputs[0][0].Length;

            // Rozměry kernelů
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – pokud je režim "same", padujeme vstup stejně jako ve forward propagaci
            double[][][] paddedInput = PadForSame(inputs);

            // 1. Výpočet dBiases: pro každý filtr se sečtou všechny hodnoty z dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outputShape[0]; i++)
                {
                    for (int j = 0; j < outputShape[1]; j++)
                    {
                        dBiases[f] += dOutput[i][j][f];
                    }
                }
            }

            int outH = outputShape[0];
            int outW = outputShape[1];

            // 2. Výpočet dKernels: pro každý filtr a každý prvek kernelu se vynásobí odpovídající hodnota z paddedInput a dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int ki = 0; ki < kH; ki++)
                {
                    for (int kj = 0; kj < kW; kj++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            // řádky vytažené o úroveň výš — viz FeedForward
                            double sum = 0;
                            for (int i = 0; i < outH; i++)
                            {
                                double[][] inRow = paddedInput[i + ki];
                                double[][] dRow = dOutput[i];

                                for (int j = 0; j < outW; j++)
                                {
                                    sum += inRow[j + kj][c] * dRow[j][f];
                                }
                            }
                            dKernels[f][ki][kj][c] += sum;
                        }
                    }
                }
            }
        }

        public override void UpdateParams()
        {
            // stejná pojistka jako u Neuron.Update_weights_bias — prázdná dávka → NaN kernely
            if (mini_batch_size == 0)
            {
                return;
            }

            int numFilters = kernels.Length;
            int idx = 0;
            // všechny filtry mají stejný tvar; +1 = bias filtru
            int perFilter = kernels[0].Length * kernels[0][0].Length * kernels[0][0][0].Length + 1;
            optimizer.EnsureCapacity(numFilters * perFilter);
            // L2 se aplikuje na kernely (to jsou váhy konvoluce), na biasy ne — viz
            // NetworkContext.L2. Přičítá se k už zprůměrovanému gradientu.
            double weightDecay = Context?.L2 ?? 0;

            for (int f = 0; f < numFilters; f++)
            {
                int kernelHeight = kernels[f].Length;
                int kernelWidth = kernels[f][0].Length;
                int inChannels = kernels[f][0][0].Length;
                // Aktualizace vah pro daný filtr
                for (int i = 0; i < kernelHeight; i++)
                {
                    for (int j = 0; j < kernelWidth; j++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            double gradient = dKernels[f][i][j][c] / mini_batch_size
                                              + weightDecay * kernels[f][i][j][c];
                            kernels[f][i][j][c] = optimizer.Update(kernels[f][i][j][c], gradient, idx++);
                        }
                    }
                }
                // Aktualizace biasu pro daný filtr
                biases[f] = optimizer.Update(biases[f], dBiases[f] / mini_batch_size, idx++);
            }
            mini_batch_size = 0;
            inicializationK_B_dK_dB();
        }

        // ---------------------------------------------------------------- konvence B

        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "Conv cannot have a layer-wide activation (softmax), " +
                    "so the softmax+CE fusion does not apply to it.");

            double[] gradOut = gradOutput.Data;
            int expected = outputShape[0] * outputShape[1] * outputShape[2];

            if (gradOut.Length != expected)
                throw new ArgumentException(
                    $"Conv has output {outputShape[0]}×{outputShape[1]}×{outputShape[2]} " +
                    $"({expected} values), but the gradient from above has {gradOut.Length}.");

            // 1) ∂L/∂output → ∂L/∂z. Tohle dělal SeedOutputGradient pro výstupní vrstvu
            //    a obě větve CalculateLayerGradients pro ty ostatní — teď je to na jednom
            //    místě, protože všechny vrstvy dostávají gradient stejnou cestou.
            int n = 0;
            for (int i = 0; i < outputShape[0]; i++)
            {
                for (int j = 0; j < outputShape[1]; j++)
                {
                    for (int f = 0; f < outputShape[2]; f++)
                    {
                        dOutput[i][j][f] = gradOut[n++] * activation_func.Derivative(raw_output[i][j][f]);
                    }
                }
            }

            // 2) gradienty kernelů a biasů ze stejného dOutput
            AccumulateKernelAndBiasGradients();

            // 3) ∂L/∂input pro vrstvu pod sebou — u první vrstvy se přeskočí, je to
            //    plná korelace, kterou by nikdo nepřevzal.
            if (!computeInputGradient) return Tensor.Empty;

            return ComputeInputGradientFromDOutput();
        }

        // ∂L/∂input z už naplněného `dOutput`. Vytaženo z CalculateLayerGradients, aby to
        // mohla použít i nová Backward() — je to jediná část, kterou obě cesty sdílejí,
        // a duplikovat 80 řádků indexové matematiky by byl přesně ten druh chyby,
        // kterou by pak nikdo nenašel.
        private Tensor ComputeInputGradientFromDOutput()
        {
            double[][][] dInput;
            // Rozměry vstupu
            int H = inputs.Length;
            int W = inputs[0].Length;
            int inChannels = inputs[0][0].Length;

            // Rozměry kernelů
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – pokud je režim "same", padujeme vstup stejně jako ve forward propagaci.
            // Un-pad níž se řídí padTop/padLeft (ne totalem/2) — u sudého kernelu je padding
            // asymetrický, takže by symetrický offset ukázal na špatné pixely.
            (int padTop, int padBottom, int padLeft, int padRight) = SamePaddingAmounts();
            double[][][] paddedInput = PadForSame(inputs);

            int paddedH = paddedInput.Length;
            int paddedW = paddedInput[0].Length;

            // Očekávané rozměry výstupu (dOutput)
            int outH = dOutput.Length;
            int outW = dOutput[0].Length;


            // Inicializace gradientů:
            // dInput_padded má stejný tvar jako paddedInput: [paddedH][paddedW][inChannels]
            double[][][] dInputPadded = new double[paddedH][][];
            for (int i = 0; i < paddedH; i++)
            {
                dInputPadded[i] = new double[paddedW][];
                for (int j = 0; j < paddedW; j++)
                {
                    dInputPadded[i][j] = new double[inChannels];
                }
            }

            // 3. Výpočet dInput (vypočítáme gradient vůči padded vstupu)
            // Pro každý výstupní bod se gradient "rozprostře" zpět na vstupní pixely pomocí hodnot kernelu
            for (int f = 0; f < numFilters; f++)
            {
                double[][][] kernel = kernels[f];

                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        // řádky vytažené o úroveň výš — viz FeedForward
                        double g = dOutput[i][j][f];

                        for (int ki = 0; ki < kH; ki++)
                        {
                            double[][] dstRow = dInputPadded[i + ki];
                            double[][] kRow = kernel[ki];

                            for (int kj = 0; kj < kW; kj++)
                            {
                                double[] dst = dstRow[j + kj];
                                double[] weights = kRow[kj];

                                for (int c = 0; c < inChannels; c++)
                                {
                                    dst[c] += weights[c] * g;
                                }
                            }
                        }
                    }
                }
            }

            // Pokud jsme používali "same", odstraníme padding, abychom získali dInput se stejnými rozměry jako původní inputs
            if (padding.ToLower() == "same")
            {
                dInput = new double[H][][];
                for (int i = 0; i < H; i++)
                {
                    dInput[i] = new double[W][];
                    for (int j = 0; j < W; j++)
                    {
                        dInput[i][j] = new double[inChannels];
                        for (int c = 0; c < inChannels; c++)
                        {
                            dInput[i][j][c] = dInputPadded[i + padTop][j + padLeft][c];
                        }
                    }
                }
            }
            else
            {
                // V režimu "valid" má dInputPadded rozměry odpovídající inputs
                dInput = dInputPadded;
            }

            return Tensor.FromJagged3D(dInput, new int[] { dInput.Length, dInput[0].Length, dInput[0][0].Length });
        }

        private void CheckIsActivationFuncIsNotApplyToLayer()
        {
            if (activation_func.Apply_to_layer)
            {
                throw new ArgumentException("unfortunately it is not possible to apply an activation function to a convulsion layer that is applied to the whole layer like softmax() instead use for example ReLu() or Tanh().");
            }
        }
        // Kolik nulových řádků/sloupců přidat pro režim "same".
        //
        // Dřív se počítalo symetricky jako k/2 na každou stranu. Pro LICHÝ kernel to sedí,
        // pro SUDÝ ne: padded rozměr vyšel in + 2·(k/2) = in + k, takže konvoluce měla dát
        // in + 1 řádků — jenže smyčka jela jen do outputShape[0] = in, takže poslední řádek
        // a sloupec tiše zmizely (žádná výjimka, jen chybějící data).
        //
        // Nově celkem k-1 (to je přesně tolik, aby výstup vyšel na velikost vstupu),
        // rozdělené před/za. U lichého k to vyjde symetricky jako dřív, u sudého asymetricky
        // — stejná konvence jako TensorFlow/Keras/PyTorch.
        private (int top, int bottom, int left, int right) SamePaddingAmounts()
        {
            int totalRows = kernels[0].Length - 1;
            int totalCols = kernels[0][0].Length - 1;

            int top = totalRows / 2;
            int left = totalCols / 2;

            return (top, totalRows - top, left, totalCols - left);
        }

        // Vrátí vstup opadovaný podle režimu; pro "valid" vrací původní pole beze změny.
        // Sdílené forwardem i oběma backward cestami, aby se nemohly rozejít.
        private double[][][] PadForSame(double[][][] input)
        {
            if (padding.ToLower() != "same")
            {
                return input;
            }

            (int top, int bottom, int left, int right) = SamePaddingAmounts();
            return Pad3D(input, top, bottom, left, right);
        }

        private double[][][] Pad3D(double[][][] input, int padTop, int padBottom, int padLeft, int padRight)
        {
            int originalRows = input.Length;
            int originalCols = input[0].Length;
            int channels = input[0][0].Length;
            int newRows = originalRows + padTop + padBottom;
            int newCols = originalCols + padLeft + padRight;

            double[][][] padded = new double[newRows][][];
            for (int i = 0; i < newRows; i++)
            {
                padded[i] = new double[newCols][];
                for (int j = 0; j < newCols; j++)
                {
                    padded[i][j] = new double[channels];
                }
            }

            for (int i = 0; i < originalRows; i++)
            {
                for (int j = 0; j < originalCols; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        padded[i + padTop][j + padLeft][c] = input[i][j][c];
                    }
                }
            }

            return padded;
        }

        private void inicializationK_B_dK_dB(int? number_of_kernels = null, int[]? finall_kernel_Shape = null, bool ChangeKernel = false)
        {
            int nkernels;
            int[] kernel_Shape;

            if (number_of_kernels == null)
            {
                nkernels = kernels.Count();
            }
            else
            {
                nkernels = (int)number_of_kernels;
            }

            if (finall_kernel_Shape == null)
            {
                kernel_Shape = new int[] { kernels[0].Length, kernels[0][0].Length, kernels[0][0][0].Length };
            }
            else
            {
                kernel_Shape = (int[])finall_kernel_Shape;
            }

            
            

            if (ChangeKernel)
            {
                dBiases = new double[nkernels];
                biases = new double[nkernels];
                kernels = new double[(int)nkernels][][][];
                dKernels = new double[(int)nkernels][][][];

                for (int i = 0; i < nkernels; i++)
                {
                    kernels[i] = new double[kernel_Shape[0]][][];
                    dKernels[i] = new double[kernel_Shape[0]][][];
                    for (int j = 0; j < kernel_Shape[0]; j++)
                    {
                        kernels[i][j] = new double[kernel_Shape[1]][];
                        dKernels[i][j] = new double[kernel_Shape[1]][];
                        for (int k = 0; k < kernel_Shape[1]; k++)
                        {
                            kernels[i][j][k] = new double[kernel_Shape[2]];
                            dKernels[i][j][k] = new double[kernel_Shape[2]];
                        }
                    }
                }

                // zdroj náhody z modelu (reprodukovatelnost přes seed); samostatná vrstva
                // bez modelu spadne na globální generátor
                Random kernelRandom = Context?.Random ?? GeneralNeuralNetworkSettings.rnd;

                for (int l = 0; l < nkernels; l++)
                {
                    biases[l] = 0;
                    dBiases[l] = 0;
                    for (int i = 0; i < kernel_Shape[0]; i++)
                    {
                        for (int j = 0; j < kernel_Shape[1]; j++)
                        {
                            for (int k = 0; k < kernel_Shape[2]; k++)
                            {
                                double fanIn = kernel_Shape[0] * kernel_Shape[1] * kernel_Shape[2];
                                double limit = Math.Sqrt(6.0 / fanIn);
                                kernels[l][i][j][k] = (kernelRandom.NextDouble() * 2 - 1) * limit;
                                dKernels[l][i][j][k] = 0;
                            }
                        }
                    }
                }

                

            }
            else
            {
                for (int l = 0; l < nkernels; l++)
                {
                    dBiases[l] = 0;
                    for (int i = 0; i < kernel_Shape[0]; i++)
                    {
                        for (int j = 0; j < kernel_Shape[1]; j++)
                        {
                            for (int k = 0; k < kernel_Shape[2]; k++)
                            {
                                dKernels[l][i][j][k] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

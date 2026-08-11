using System.Data;
using System.Globalization;
using My_DNN.Layers.classes;
using My_DNN.Loss_functions;
namespace My_DNN
{
    public class Train
    {
        private readonly MDNN _model;

        private uint _epoch;
        private ulong _optimizerSteps;
        private uint _totalEpoch;
        private uint _sizeOfMiniBatch;
        private double _lowestLoss;
        private uint _bestEpoch;
        private string? _bestSnapshot;   // nejlepší model v paměti (JSON snapshot), obnovuje se na konci
        private uint _reportsWithoutImprovement;   // early stopping: počet valid reportů bez zlepšení
        private bool _stopEarly;                    // early stopping: flag k ukončení smyčky

        // Jeden zdroj náhody pro celý model (výběr vzorků i míchání datasetu). Dřív tu byl
        // vlastní `new Random()` a ShuffleTensor sahal na Random.Shared — tři nezávislé
        // zdroje dohromady znamenaly, že stejný experiment nešlo zopakovat.
        private Random Rnd => _model.Context.Random;

        private List<int> _listOfepoch = [];
        private List<double> _listOfValidLoss = [];
        private List<double> _listOfTrainLoss = [];

        public uint NumberOfSkipFristEpochInPlotter = 0;
        public uint NumberOfShowEpochInConsole = 100;

        // Po kolika epochách reportovat. Dřív se počítalo inline jako
        // `_totalEpoch / NumberOfShowEpochInConsole` a klamp se zapisoval PŘÍMO do veřejného
        // NumberOfShowEpochInConsole → trénink si uživatelovo nastavení natrvalo přepsal
        // (po 5 epochách zůstalo 5 i pro další trénink na 5000 epoch).
        private uint _reportInterval = 1;

        // Stav pro odhad zbývajícího času. DŘÍV to byly statiky v ConsoleControleru →
        // druhý trénink v procesu viděl hodnoty z prvního, vyšlo pastEpochs == 0 a
        // (subTime * n) / 0 hodilo OverflowException. Per-instance + reset na začátku běhu.
        internal DateTime LastReportTime = DateTime.MinValue;
        internal uint LastReportEpoch = 0;

        public bool ShowLossChartInTrainLoop = true;
        public bool TestNeuralNetworkAfterTraining = true;
        public bool ShowModelInfoIntrainLoop = true;
        public bool AutoSaveInTrainLoop = true;

        public string AutoSaveInTrainLoopFileName = "AutoSave";

        // Early stopping: zastav trénink, když se valid loss nezlepší po `patience` valid reportech.
        // Nejlepší model je díky in-memory snapshotu na konci stejně obnoven (restore-best zadarmo).
        public bool EarlyStoppingEnabled = false;      // opt-in, default vyp → zpětně kompatibilní
        public uint EarlyStoppingPatience = 10;        // kolik valid reportů bez zlepšení = stop
        public double EarlyStoppingMinDelta = 0.0;     // min. pokles loss, aby se počítal jako zlepšení

        // Poměry rozdělení datasetu. null = dopočítat ze zbytku (rovným dílem mezi vynechané).
        // Když jsou všechny null, použije se 0.7 / 0.15 / 0.15 (dosavadní default).
        // Např. Train=0.6, Valid=0.2, Test=null → test se dopočítá na 0.2.
        public double? TrainSplitRatio = null;
        public double? ValidSplitRatio = null;
        public double? TestSplitRatio = null;

        private Tensor? _trainDataInputs;
        private Tensor? _testDataInputs;
        private Tensor? _validDataInputs;

        private Tensor? _trainDataCurrentOutput;
        private Tensor? _testDataCurrentOutput;
        private Tensor? _validDataCurrentOutput;
        // Co dodal UŽIVATEL (na rozdíl od toho, co si dopočítalo dělení datasetu).
        // Bez tohohle rozlišení druhý TrainLoop na stejném modelu viděl valid/test z prvního
        // běhu, spadl do jiné větve DividingDataIntoDatasets a natrénoval na CELÉM datasetu
        // včetně testovací části — tiše, bez varování.
        private Tensor? _userValidInputs;
        private Tensor? _userValidOutputs;
        private Tensor? _userTestInputs;
        private Tensor? _userTestOutputs;

        public Tensor? TestDataInputs
        {
            get => _testDataInputs;
            set { _testDataInputs = value; _userTestInputs = value; }
        }
        public Tensor? ValidDataInputs
        {
            get => _validDataInputs;
            set { _validDataInputs = value; _userValidInputs = value; }
        }
        public Tensor? TestDataCurrentOutput
        {
            get => _testDataCurrentOutput;
            set { _testDataCurrentOutput = value; _userTestOutputs = value; }
        }
        public Tensor? ValidDataCurrentOutput
        {
            get => _validDataCurrentOutput;
            set { _validDataCurrentOutput = value; _userValidOutputs = value; }
        }

        // Read-only protějšky k valid/test — ať jde zvenčí zjistit, na čem se doopravdy
        // trénovalo (a ať je vidět, když se dělení rozjede).
        public Tensor? TrainDataInputs => _trainDataInputs;
        public Tensor? TrainDataCurrentOutput => _trainDataCurrentOutput;

        // Počet DOKONČENÝCH epoch, tedy plných průchodů trénovacím setem.
        // POZOR na změnu významu: dřív `UpdateParams()` inkrementoval tenhle čítač, a protože
        // se volal jednou za „epochu", vycházelo to nastejno. Po převodu na konvenční epochy
        // připadá na jednu epochu tolik kroků optimizeru, kolik je dávek — čítače se rozešly.
        // Ruční trénink (Fit + UpdateParams ve vlastní smyčce) žádné epochy nemá, takže
        // CurrentEpoch zůstane 0 a počítá se jen OptimizerSteps.
        public uint CurrentEpoch
        {
            get { return _epoch; }
        }

        // Kolik kroků optimizeru (volání UpdateParams) proběhlo za život modelu.
        public ulong OptimizerSteps
        {
            get { return _optimizerSteps; }
        }

        // nejnižší valid loss dosud a epocha, ve které padl (early-stopping indikátor).
        // _lowestLoss == double.MaxValue znamená, že zatím žádný valid report neproběhl.
        public double LowestValidLoss
        {
            get { return _lowestLoss; }
        }

        public uint BestEpoch
        {
            get { return _bestEpoch; }
        }

        public uint TotalEpoch
        {
            get { return _totalEpoch; }
            set { _totalEpoch = value; }
        }

        public uint MiniBatch
        {
            get { return _sizeOfMiniBatch; }
            set { _sizeOfMiniBatch = value; }
        }

        public Train(MDNN model) 
        {
            this._model = model;
            _epoch = 0;
            _totalEpoch = 0;
            _sizeOfMiniBatch = 1;
        }

        public Train(MDNN model, uint epoch,uint totalEpoch, uint sizeOfMiniBatch)
        {
            this._model = model;
            this._epoch = epoch;
            this._totalEpoch = totalEpoch;
            this._sizeOfMiniBatch=sizeOfMiniBatch;
        }

        public Tensor Fit(Tensor inputsValues, Tensor targetValues)
        {
            if (_model.Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                _model.Context.InputShape = inputsValues.Shape;
                _model.Layers.SetInputSizeForFirstLayer();
            }

            CheckLayersAreNotEmpty();

            Tensor output = _model.GetResults(inputsValues);

            BackPropagation(targetValues);

            return output;
        }

        public async Task<Tensor> FitAsync(Tensor inputsValues, Tensor targetValues)
        {
            if (_model.Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                _model.Context.InputShape = inputsValues.Shape;
                _model.Layers.SetInputSizeForFirstLayer();
            }
            CheckLayersAreNotEmpty();

            Tensor output = await _model.GetResultsAsync(inputsValues);

            Tensor[] de = await Gradient.GetGradientsAsync(targetValues, _model);

            await PropagationAsync(de);

            return output;
        }

        public void BackPropagation(Tensor targetValues)
        {
            CheckLayersAreNotEmpty();

            Tensor[] de = Gradient.GetGradients(targetValues, _model);

            Propagation(de);
        }

        public async Task BackPropagationAsync(Tensor targetValues)
        {
            CheckLayersAreNotEmpty();

            Tensor[] de = await Gradient.GetGradientsAsync(targetValues, _model);

            await PropagationAsync(de);
        }

        public void BackPropagation(Tensor[] layerGradients)
        {
            CheckLayersAreNotEmpty();
            Propagation(layerGradients);
        }

        public async Task BackPropagationAsync(Tensor[] layerGradients)
        {
            CheckLayersAreNotEmpty();
            await PropagationAsync(layerGradients);
        }

        public Tensor FeedForward(Tensor inputsValue)
        {
            return _model.GetResults(inputsValue);
        }

        public async Task<Tensor> FeedForwardAsync(Tensor inputsValue)
        {
            return await _model.GetResultsAsync(inputsValue);
        }

        public void UpdateParams()
        {
            CheckLayersAreNotEmpty();

            _optimizerSteps++;
            foreach (Layer layer in _model.Layers.Layers)
            {
                layer.UpdateParams();
            }
        }

        public async Task UpdateParamsAsync()
        {
            CheckLayersAreNotEmpty();

            _optimizerSteps++;
            int index = -1;
            Task[] tasks = new Task[_model.Layers.Layers.Count()];
            foreach (Layer layer in _model.Layers.Layers)
            {
                index++;
                tasks[index] = Task.Run(async () =>
                {
                    await layer.UpdateParamsAsync();
                });
            }

            await Task.WhenAll(tasks);
        }

        public void ComplexTestNeuralNetwork()
        {
            (int,int) testScore, trainScore, validationScore;
            
            if (_testDataInputs != null && _testDataCurrentOutput != null && _trainDataInputs != null &&
                _trainDataCurrentOutput != null && _validDataInputs != null && _validDataCurrentOutput != null)
            {
                testScore =  TestNeuralNetwork(_testDataInputs, _testDataCurrentOutput,false);
                trainScore = TestNeuralNetwork(_trainDataInputs, _trainDataCurrentOutput,false);
                validationScore = TestNeuralNetwork(_validDataInputs, _validDataCurrentOutput,false);
            }
            else
            {
                throw new Exception("Dataset is Empty");
            }
            
            ConsoleControler.ShowComplexScoreOfmodel(testScore, trainScore, validationScore);
                    
        }

        public (int, int) TestNeuralNetwork(Tensor inputsValues, Tensor currentOutputValues, bool showInConsole = true)
        {
            if (_model.Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                if (_model.Context.SequenceTrain)
                {
                    _model.Context.InputShape = inputsValues.GetTensorValue([0,0]).Shape;
                }

                else
                {
                    _model.Context.InputShape = inputsValues.GetTensorValue([0]).Shape;
                }

                _model.Layers.SetInputSizeForFirstLayer();
            }
            CheckLayersAreNotEmpty();

            int score = 0;
            int maxScore = 0;

            if (_model.Context.SequenceTrain)
            {
                for (int i = 0; i < inputsValues.Shape[0]; i++)
                {
                    _model.ResetSequence();
                    for (int j = 0; j < inputsValues.Shape[1]; j++)
                    {
                        Tensor outputTensor = _model.GetResults(inputsValues.GetTensorValue([i, j]));

                        double[] output = outputTensor.Data;

                        // Počítáme ČASOVÉ KROKY, ne výstupní neurony. Dřív tu bylo
                        // `maxScore += output.Length`, zatímco score se zvyšovalo o 1 za krok
                        // → poměr score/maxScore byl v sekvenčním režimu nesmyslné číslo
                        // (dělilo se počtem neuronů). Nesekvenční větev to měla správně.
                        maxScore++;

                        bool zeroError = true;

                        // upravit na softMax
                        if (_model.Layers.Layers[_model.Layers.Layers.Count() - 1].Activation_Func.Apply_to_layer)
                        {
                            double maxOutputValue = output.Max();
                            int maxOutputIndex = output.ToList().IndexOf(maxOutputValue);

                            double maxCurrentOutputValue = ((double[])currentOutputValues.GetValue([i,j])).Max();
                            int maxCurrentOutputIndex = ((double[])currentOutputValues.GetValue([i,j])).ToList().IndexOf(maxCurrentOutputValue);

                            if (maxCurrentOutputIndex != maxOutputIndex)
                            {
                                zeroError = false;
                            }

                        }
                        else
                        {
                            for (int k = 0; k < output.Length; k++)
                            {
                                if (Math.Abs(Math.Round(output[k]) - Math.Round(((double[])currentOutputValues.GetValue(
                                        [i,j]))[k])) > 0.1)
                                {
                                    zeroError = false;
                                    break;
                                }
                            }
                        }


                        if (zeroError)
                        {
                            score++;
                        }
                    }
                }
                _model.ResetSequence();
            }

            else
            {
                for (int i = 0; i < inputsValues.Shape[0]; i++)
                {
                    Tensor outputTensor = _model.GetResults(inputsValues.GetTensorValue([i]));

                    double[] output = outputTensor.Data;
                    double[] currentOutput = currentOutputValues.GetTensorValue([i]).Data;

                    bool zeroError = true;
                    maxScore = inputsValues.Shape[0];

                    // upravit na softMax
                    if (_model.Layers.Layers[_model.Layers.Layers.Count() - 1].Activation_Func.Apply_to_layer)
                    {
                        double maxOutputValue = output.Max();
                        int maxOutputIndex = output.ToList().IndexOf(maxOutputValue);

                        double maxCurrentOutputValue = currentOutput.Max();
                        int maxCurrentOutputIndex = currentOutput.ToList().IndexOf(maxCurrentOutputValue);

                        if (maxCurrentOutputIndex != maxOutputIndex)
                        {
                            zeroError = false;
                        }

                    }
                    else
                    {
                        for (int j = 0; j < output.Length; j++)
                        {
                            if (Math.Abs(Math.Round(output[j]) - Math.Round(currentOutput[j])) > 0.1)
                            {
                                zeroError = false;
                                break;
                            }
                        }
                    }


                    if (zeroError)
                    {
                        score++;
                    }
                }
            }
            if (showInConsole)
                ConsoleControler.ShowScoreOfmodel(score, maxScore);
            
            return (score, maxScore);
        }

        public void TrainLoop(Array inputsValues, Array currentOutputValues, uint numberOfEpoch, uint sizeOfMiniBatch = 1, bool isSequence = false)
        {

            Tensor tensorInputsValues = Tensor.ConvertArrayToTensor(inputsValues);
            Tensor tensorCurrentOutputValues = Tensor.ConvertArrayToTensor(currentOutputValues);

            PreparationForTrainLoop(tensorInputsValues, tensorCurrentOutputValues, numberOfEpoch, sizeOfMiniBatch, isSequence);
            if (ShowModelInfoIntrainLoop)
            {
                _model.info();
            }
            
            for (uint epoch = this._epoch; epoch < _totalEpoch; epoch++)
            {
                _model.Context.Loss.ResetAverageLossPerIteration();

                if (_trainDataInputs == null || _trainDataCurrentOutput == null)
                {
                    throw new InvalidOperationException(
                        "_trainDataInputs/_trainDataCurrentOutput je null — trénovací data nebyla rozdělena (DividingDataIntoDatasets).");
                }

                RunOneEpoch(_trainDataInputs, _trainDataCurrentOutput);

                // Epochy počítáme tady; UpdateParams() teď inkrementuje kroky optimizeru,
                // kterých je na epochu tolik, kolik je dávek.
                _epoch = epoch + 1;

                if (epoch % _reportInterval == 0)
                {
                    TrainLoopControlFunc();
                    if (_stopEarly)
                    {
                        ConsoleControler.ShowEarlyStopping(_epoch, EarlyStoppingPatience, _lowestLoss, _bestEpoch);
                        break;
                    }
                }
            }

            if (TestNeuralNetworkAfterTraining)
            {
                // vždy testujeme NEJLEPŠÍ model (nejnižší valid loss), ne poslední epochu:
                // obnovíme in-memory snapshot in-place (identita + datasety zůstanou).
                if (_bestSnapshot != null)
                {
                    _model.LoadWeightsFromString(_bestSnapshot);
                }
                ComplexTestNeuralNetwork();
            }
            if (ShowLossChartInTrainLoop)
            {
                GraphPlotter.ShowLossGraph(_listOfepoch.ToArray(), _listOfTrainLoss.ToArray(), _listOfValidLoss.ToArray());
            }
        }

        public async Task TrainLoopAsync(Array inputsValues, Array currentOutputValues, uint numberOfEpoch, uint sizeOfMiniBatch = 1, bool isSequence = false)
        {
            Tensor tensorInputsValues = Tensor.ConvertArrayToTensor(inputsValues);
            Tensor tensorCurrentOutputValues = Tensor.ConvertArrayToTensor(currentOutputValues);

            PreparationForTrainLoop(tensorInputsValues, tensorCurrentOutputValues, numberOfEpoch, sizeOfMiniBatch, isSequence);

            if (ShowModelInfoIntrainLoop)
            {
                _model.info();
            }

            for (uint epoch = this._epoch; epoch < _totalEpoch; epoch++)
            {
                _model.Context.Loss.ResetAverageLossPerIteration();

                if (_trainDataInputs == null || _trainDataCurrentOutput == null)
                {
                    throw new InvalidOperationException(
                        "_trainDataInputs/_trainDataCurrentOutput je null — trénovací data nebyla rozdělena (DividingDataIntoDatasets).");
                }

                Tensor trainInputs = _trainDataInputs;
                Tensor trainOutputs = _trainDataCurrentOutput;

                int[] order = ShuffledIndices(trainInputs.Shape[0]);
                int batchSize = (int)_sizeOfMiniBatch;

                for (int start = 0; start < order.Length; start += batchSize)
                {
                    int end = Math.Min(start + batchSize, order.Length);

                    for (int k = start; k < end; k++)
                    {
                        int num = order[k];
                        if (!_model.Context.SequenceTrain)
                        {
                            await FitAsync(trainInputs.GetTensorValue([num]), trainOutputs.GetTensorValue([num]));
                        }
                        else
                        {
                            _model.ResetSequence();
                            for (int i = 0; i < trainInputs.Shape[1]; i++)
                            {
                                await FitAsync(trainInputs.GetTensorValue([num, i]), trainOutputs.GetTensorValue([num, i]));
                            }
                            _model.ResetSequence();
                        }
                    }

                    await UpdateParamsAsync();
                }

                _epoch = epoch + 1;

                if (epoch % _reportInterval == 0)
                {
                    await TrainLoopControlFuncAsync();
                    if (_stopEarly)
                    {
                        ConsoleControler.ShowEarlyStopping(_epoch, EarlyStoppingPatience, _lowestLoss, _bestEpoch);
                        break;
                    }
                }
            }

            if (TestNeuralNetworkAfterTraining)
            {
                // vždy testujeme NEJLEPŠÍ model (nejnižší valid loss), ne poslední epochu:
                // obnovíme in-memory snapshot in-place (identita + datasety zůstanou).
                if (_bestSnapshot != null)
                {
                    _model.LoadWeightsFromString(_bestSnapshot);
                }
                ComplexTestNeuralNetwork();
            }

            if (ShowLossChartInTrainLoop)
            {
                GraphPlotter.ShowLossGraph(_listOfepoch.ToArray(), _listOfTrainLoss.ToArray(), _listOfValidLoss.ToArray());
            }

        }

        public void SimpleTrainLoop(double[][] inputsValues, double[][] currentOutputValues, uint numberOfEpoch, uint sizeOfMiniBatch = 1)
        {
            uint epoch = this._epoch;
            double minLoss = 100;

            if (sizeOfMiniBatch < 1)
            {
                throw new ArgumentException("Velikost minibatche musí být aspoň 1.", nameof(sizeOfMiniBatch));
            }

            this._sizeOfMiniBatch = sizeOfMiniBatch;
            _totalEpoch = numberOfEpoch;
            
            ResolveReportInterval();


            _model.Context.InputShape = [inputsValues[0].Length];
            _model.Layers.SetInputSizeForFirstLayer();

            CheckLayersAreNotEmpty();

            for (; epoch < _totalEpoch; epoch++)
            {
                // stejná sémantika jako TrainLoop: plný průchod po dávkách, ne náhodné tahy
                int[] order = ShuffledIndices(inputsValues.Length);
                int batchSize = (int)sizeOfMiniBatch;

                for (int start = 0; start < order.Length; start += batchSize)
                {
                    int end = Math.Min(start + batchSize, order.Length);

                    for (int k = start; k < end; k++)
                    {
                        int num = order[k];
                        Fit(new Tensor(inputsValues[num]), new Tensor(currentOutputValues[num]));
                    }

                    UpdateParams();
                }

                _epoch = epoch + 1;

                if (epoch % _reportInterval == 0)
                {
                    double loss = _model.Context.Loss.GetAverageLossPerIteration();

                    // Stejná politika jako v TrainLoop (ExtraControlFunc): divergence =
                    // výjimka, ne tichý `return`. Dřív se tady jen vypsalo „Error: Nan number"
                    // a metoda skončila jakoby úspěšně — volající neměl jak poznat rozdíl
                    // mezi „dotrénováno" a „rozpadlo se to hned v první epoše".
                    if (!double.IsFinite(loss))
                    {
                        ConsoleControler.ErrorHandler("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values ​​in the training parameters (for example: learning rate or hyperammetry )", true);
                        throw new TrainingDivergedException(_epoch, loss);
                    }

                    if (loss < minLoss)
                    {
                        minLoss = loss;
                        _model.Note = minLoss.ToString(CultureInfo.InvariantCulture);
                        if (AutoSaveInTrainLoop)
                        {
                            _model.SaveAsJson(AutoSaveInTrainLoopFileName);
                        }
                    }

                    ConsoleControler.ShowEpochInfo(_model);
                }
            }
        }

        private void PreparationForTrainLoop(Tensor inputsValues, Tensor currentOutputValues, uint numberOfEpoch, uint sizeOfMiniBatch = 1, bool isSequence = false)
        {
            CheckTensorShapes(inputsValues, currentOutputValues);

            ShuffleTensor(inputsValues, currentOutputValues, out inputsValues, out currentOutputValues);

            if (isSequence)
            {
                _model.Context.SequenceTrain = true;
            }

            if (_model.Context.SequenceTrain)
            {
                if (inputsValues.Shape.Length == 1)
                {
                    throw new Exception("for sequential training of the model, the inputs must be at least in a two-dimensional array");
                }

                if (inputsValues.Shape.Length == 2)
                {
                    inputsValues.Reshape([inputsValues.Shape[0], inputsValues.Shape[1], 1]);
                    currentOutputValues.Reshape([inputsValues.Shape[0], inputsValues.Shape[1], 1]);
                }

                else if (inputsValues.Shape.Length >= 6)
                {
                    throw new Exception("for sequential training, the maximum input is a four-dimensional array.");
                }

                _model.Context.InputShape = inputsValues.GetTensorValue([0, 0]).Shape;
            }

            else
            {

                if (inputsValues.Shape.Length == 1)
                {
                    inputsValues.Reshape([inputsValues.Shape[0], 1]);
                    currentOutputValues.Reshape([inputsValues.Shape[0], 1]);
                }

                else if (inputsValues.Shape.Length >= 4)
                {
                    throw new Exception("for non-sequential training, the maximum input is a three-dimensional array.");
                }

                _model.Context.InputShape = inputsValues.GetTensorValue([0]).Shape;
            }

            

            if (_model.Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                _model.Layers.SetInputSizeForFirstLayer();
            }

            CheckLayersAreNotEmpty();

            ResetDerivedDatasets();
            DividingDataIntoDatasets(inputsValues, currentOutputValues);

            _listOfepoch = [];
            _listOfValidLoss = [];
            _listOfTrainLoss = [];

            // Velikost dávky musí být aspoň 1 — `start += batchSize` s nulou by byla
            // nekonečná smyčka. (Dřív se s nulou prostě netrénovalo, tiše.)
            if (sizeOfMiniBatch < 1)
            {
                throw new ArgumentException("Velikost minibatche musí být aspoň 1.", nameof(sizeOfMiniBatch));
            }

            this._sizeOfMiniBatch = sizeOfMiniBatch;
            _totalEpoch = numberOfEpoch;

            _lowestLoss = double.MaxValue;
            _bestEpoch = 0;
            _bestSnapshot = null;
            _reportsWithoutImprovement = 0;
            _stopEarly = false;

            ResolveReportInterval();
        }

        // Přeloží uživatelské „kolikrát chci report" na „po kolika epochách reportovat".
        // Klamp jde do privátního _reportInterval, veřejné nastavení zůstane nedotčené.
        // Zároveň reset stavu ETA, aby druhý trénink nepočítal z časů toho prvního.
        private void ResolveReportInterval()
        {
            uint reports = NumberOfShowEpochInConsole;

            if (reports < 1)
            {
                reports = 1;
            }
            else if (reports > _totalEpoch)
            {
                reports = _totalEpoch;
            }

            _reportInterval = (_totalEpoch == 0 || reports == 0) ? 1 : Math.Max(1u, _totalEpoch / reports);

            LastReportTime = DateTime.MinValue;
            LastReportEpoch = 0;
        }

        private void CheckLayersAreNotEmpty()
        {
            if (_model.Layers.Layers.Count == 0)
            {
                throw new Exception("Only model with layers can be trained, please add at least one Layer using Layer.add() function.");
            }

            if (_model.Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                throw new Exception("the input layer must always have at least one input (use the SetInputSizeForFirstLayer() method in the layer class to set the input, for example model.layer.SetInputSizeForFirstLayer(new unit[] {1}))");
            }
        }

        private void CheckTensorShapes(Tensor a, Tensor b)
        {
            if(a == null || b == null)
            {
                throw new Exception("it is not possible to train with empty inputs (Tensor), inputs_values or current_output_values is null");
            }

            
            /*
            if(B.Shape.Length != A.Shape.Length)
            {
                throw new Exception("both inputs_values and current_output_values must have the same number of dimension array");
            }
            */

            //int size = A.Shape.Length;

            /*
            for(int i = 0; i < (size-1); i++)
            {
                if (B.Shape[i] != A.Shape[i])
                {
                    throw new Exception("both inputs_values and current_output_values must have the same array dimensions");
                }
            }
            */

            if (b.Shape[0] != a.Shape[0])
            {
                throw new Exception("both inputs_values and current_output_values must have the same first array dimensions");
            }


        }

        // Vyřeší poměry train/valid/test: vynechané (null) dostanou rovným dílem zbytek do 1.
        // Všechny null → kanonický default 0.7/0.15/0.15 (zachová dosavadní chování).
        private (double train, double valid, double test) ResolveSplitRatios()
        {
            if (TrainSplitRatio == null && ValidSplitRatio == null && TestSplitRatio == null)
            {
                return (0.7, 0.15, 0.15);
            }

            double setSum = 0;
            int nullCount = 0;
            foreach (double? r in new[] { TrainSplitRatio, ValidSplitRatio, TestSplitRatio })
            {
                if (r == null)
                {
                    nullCount++;
                }
                else
                {
                    if (r <= 0 || r >= 1)
                        throw new Exception("Split poměr musí být v intervalu (0,1).");
                    setSum += r.Value;
                }
            }

            if (nullCount == 0 && Math.Abs(setSum - 1.0) > 1e-9)
                throw new Exception("Když jsou zadané všechny tři poměry, musí dát dohromady 1.");
            if (nullCount > 0 && setSum >= 1.0)
                throw new Exception("Součet zadaných poměrů musí být < 1, aby na dopočítané (null) zbyl kladný podíl.");

            double each = nullCount > 0 ? (1.0 - setSum) / nullCount : 0.0;

            return (TrainSplitRatio ?? each, ValidSplitRatio ?? each, TestSplitRatio ?? each);
        }

        // Každý běh musí startovat ze stejného výchozího stavu: buď z toho, co uživatel
        // explicitně dodal, nebo z prázdna. Odvozené (nasliceované) datasety z minulého
        // běhu se zahodí — jinak by DividingDataIntoDatasets vzalo slice předchozího valid
        // setu za „uživatelův valid set" a rozdělení by se s každým během posouvalo.
        private void ResetDerivedDatasets()
        {
            _validDataInputs = _userValidInputs;
            _validDataCurrentOutput = _userValidOutputs;

            _testDataInputs = _userTestInputs;
            _testDataCurrentOutput = _userTestOutputs;

            _trainDataInputs = null;
            _trainDataCurrentOutput = null;
        }

        public void DividingDataIntoDatasets(Tensor inputsValues, Tensor currentOutputValues)
        {
            int trainDataSize;
            int validDataSize;
            int testDataSize;

            int totalSize = inputsValues.Shape[0];

            if (_validDataInputs == null && _testDataInputs == null)
            {
                var (trainRatio, validRatio, _) = ResolveSplitRatios();
                trainDataSize = (int)(totalSize * trainRatio);
                validDataSize = (int)(totalSize * validRatio);
                testDataSize = totalSize - trainDataSize - validDataSize;   // zbytek → žádný vzorek se neztratí

                _validDataInputs = inputsValues.Slice(trainDataSize, validDataSize);
                _testDataInputs = inputsValues.Slice(trainDataSize + validDataSize, testDataSize);

                _validDataCurrentOutput = currentOutputValues.Slice(trainDataSize, validDataSize);
                _testDataCurrentOutput = currentOutputValues.Slice(trainDataSize + validDataSize, testDataSize);

                _trainDataInputs = inputsValues.Slice(0, trainDataSize);
                _trainDataCurrentOutput = currentOutputValues.Slice(0, trainDataSize);
            }
            else if (_validDataInputs != null && _testDataInputs == null)
            {
                int oldValidSize = _validDataInputs.Shape[0];

                // předaný valid dělíme na valid/test ve stejném poměru valid:test
                var (_, validRatio, testRatio) = ResolveSplitRatios();
                validDataSize = (int)(oldValidSize * validRatio / (validRatio + testRatio));
                testDataSize = oldValidSize - validDataSize;

                _trainDataInputs = inputsValues;
                _trainDataCurrentOutput = currentOutputValues;

                // Test se musí vyříznout z PŮVODNÍHO valid tensoru; kdyby se _validDataInputs
                // nejdřív přepsal na Slice(0, validDataSize), test-slice od offsetu validDataSize
                // by byl mimo rozsah zmenšeného tensoru → "Invalid slice range!".
                Tensor originalValidInputs = _validDataInputs;
                Tensor? originalValidOutput = _validDataCurrentOutput;

                _validDataInputs = originalValidInputs.Slice(0, validDataSize);
                _validDataCurrentOutput = originalValidOutput?.Slice(0, validDataSize);

                _testDataInputs = originalValidInputs.Slice(validDataSize, testDataSize);
                _testDataCurrentOutput = originalValidOutput?.Slice(validDataSize, testDataSize);
            }
            else if (_validDataInputs == null && _testDataInputs != null)
            {
                // test je předaný → zbytek (train+valid) dělíme ve stejném poměru train:valid
                var (trainRatio, validRatio, _) = ResolveSplitRatios();
                trainDataSize = (int)(totalSize * trainRatio / (trainRatio + validRatio));
                validDataSize = totalSize - trainDataSize;

                _trainDataInputs = inputsValues.Slice(0, trainDataSize);
                _trainDataCurrentOutput = currentOutputValues.Slice(0, trainDataSize);

                _validDataInputs = inputsValues.Slice(trainDataSize, validDataSize);
                _validDataCurrentOutput = currentOutputValues.Slice(trainDataSize, validDataSize);
            }
            else
            {
                _trainDataInputs = inputsValues;
                _trainDataCurrentOutput = currentOutputValues;
            }
        }

        // JEDNA EPOCHA = jeden plný průchod trénovacím setem.
        //
        // Dřív se za „epochu" považovalo `size_of_mini_batch` náhodných tahů S OPAKOVÁNÍM,
        // po kterých přišel jeden krok optimizeru. Důsledky: některé vzorky model za celý
        // trénink neviděl, jiné dostal několikrát, pokrytí dat bylo náhodné a `number_of_epoch`
        // byl fakticky počet kroků optimizeru, ne průchodů daty.
        //
        // Nově: každou epochu se zamíchá pořadí VŠECH trénovacích vzorků a projde se po
        // dávkách velikosti `size_of_mini_batch`, s jedním `UpdateParams()` na dávku.
        // Poslední dávka může být menší — `Neuron.Update_weights_bias` dělí skutečným
        // počtem nasčítaných vzorků, takže průměr gradientu vyjde správně i tak.
        private void RunOneEpoch(Tensor inputs, Tensor targets)
        {
            int[] order = ShuffledIndices(inputs.Shape[0]);
            int batchSize = (int)_sizeOfMiniBatch;

            for (int start = 0; start < order.Length; start += batchSize)
            {
                int end = Math.Min(start + batchSize, order.Length);

                for (int k = start; k < end; k++)
                {
                    FitSample(inputs, targets, order[k]);
                }

                UpdateParams();   // jeden krok optimizeru na dávku
            }
        }

        // Zamíchané pořadí indexů pro jednu epochu. Fisher–Yates z Contextu, takže je to
        // se seedem reprodukovatelné. Míchají se INDEXY, ne data — u velkého datasetu by
        // kopírovat tenzory každou epochu bylo zbytečně drahé.
        private int[] ShuffledIndices(int count)
        {
            int[] indices = Enumerable.Range(0, count).ToArray();

            for (int i = count - 1; i > 0; i--)
            {
                int j = Rnd.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            return indices;
        }

        // Jeden trénovací vzorek (nebo celá sekvence) podle indexu.
        private void FitSample(Tensor inputs, Tensor targets, int index)
        {
            if (!_model.Context.SequenceTrain)
            {
                Fit(inputs.GetTensorValue([index]), targets.GetTensorValue([index]));
                return;
            }

            _model.ResetSequence();
            for (int step = 0; step < inputs.Shape[1]; step++)
            {
                Fit(inputs.GetTensorValue([index, step]), targets.GetTensorValue([index, step]));
            }
            _model.ResetSequence();
        }

        public void ShuffleTensor(Tensor tensorA, Tensor tensorB, out Tensor shuffledA, out Tensor shuffledB)
        {
            if (tensorA.Shape[0] != tensorB.Shape[0])
                throw new ArgumentException("Oba tensory musí mít stejný počet vzorků");

            int batchSize = tensorA.Shape[0];
            int[] indices = Enumerable.Range(0, batchSize).OrderBy(_ => Rnd.Next()).ToArray();


            double[] shuffledDataA = new double[tensorA.Data.Length];
            double[] shuffledDataB = new double[tensorB.Data.Length];

            int sampleSizeA = tensorA.Data.Length / batchSize; 
            int sampleSizeB = tensorB.Data.Length / batchSize;

            for (int i = 0; i < batchSize; i++)
            {
                int srcIndex = indices[i];

                Array.Copy(tensorA.Data, srcIndex * sampleSizeA, shuffledDataA, i * sampleSizeA, sampleSizeA);
                Array.Copy(tensorB.Data, srcIndex * sampleSizeB, shuffledDataB, i * sampleSizeB, sampleSizeB);
            }

            shuffledA = new Tensor(shuffledDataA, tensorA.Shape);
            shuffledB = new Tensor(shuffledDataB, tensorB.Shape);
        }

        // Shodná logika jako v TestNeuralNetwork: u klasifikace (softmax / apply-to-layer)
        // se porovná argmax výstupu s argmaxem targetu, u regrese zaokrouhlené hodnoty.
        // Sdílené, aby průběžná accuracy počítala stejně jako závěrečný TestNeuralNetwork.
        private bool IsPredictionCorrect(double[] output, double[] target)
        {
            if (_model.Layers.Layers[_model.Layers.Layers.Count() - 1].Activation_Func.Apply_to_layer)
            {
                return output.ToList().IndexOf(output.Max()) == target.ToList().IndexOf(target.Max());
            }

            for (int k = 0; k < output.Length; k++)
            {
                if (Math.Abs(Math.Round(output[k]) - Math.Round(target[k])) > 0.1)
                {
                    return false;
                }
            }
            return true;
        }

        private void TrainLoopControlFunc()
        {
            Loss lossFunc = _model.Context.Loss;
            double loss = lossFunc.GetResetAverageLossPerIteration();
            // Non-null kontrakt zajišťujeme ZDE, ne v Loss: valid data přes guard + lokály,
            // výstup bereme z návratové hodnoty GetResults (vždy non-null), ne z nullable
            // property Layer_output. Do CalculateLoss se tak null nemá jak dostat.
            if (ValidDataInputs == null || ValidDataCurrentOutput == null)
            {
                throw new InvalidOperationException("Valid data nejsou nastavena (DividingDataIntoDatasets) — nelze spočítat valid loss.");
            }

            Tensor validInputs = ValidDataInputs;
            Tensor validOutputs = ValidDataCurrentOutput;

            // úspěšnost počítáme ve STEJNÉM průchodu jako loss (žádný forward navíc)
            int correct = 0;
            int total = 0;

            if (_model.Context.SequenceTrain)
            {
                for (int i = 0; i < validInputs.Shape[0]; i++)
                {
                    _model.ResetSequence();
                    for (int j = 0; j < validInputs.Shape[1]; j++)
                    {
                        Tensor output = _model.GetResults(validInputs.GetTensorValue([i, j]));
                        double[] target = validOutputs.GetTensorValue([i, j]).Data;
                        lossFunc.CalculateLoss(output.Data, target);
                        if (IsPredictionCorrect(output.Data, target)) correct++;
                        total++;
                    }
                }
            }

            else
            {
                for (int i = 0; i < validInputs.Shape[0]; i++)
                {
                    Tensor output = _model.GetResults(validInputs.GetTensorValue([i]));
                    double[] target = validOutputs.GetTensorValue([i]).Data;
                    lossFunc.CalculateLoss(output.Data, target);
                    if (IsPredictionCorrect(output.Data, target)) correct++;
                    total++;
                }
            }

            double validAccuracy = total > 0 ? (double)correct / total * 100.0 : 0;

            ExtraControlFunc(loss);
            ConsoleControler.ShowEpochInfo(_model, loss, validAccuracy: validAccuracy);
        }

        private async Task TrainLoopControlFuncAsync()
        {
            Loss lossFunc = _model.Context.Loss;
            List<double> layerOutput = [];

            double loss = lossFunc.GetResetAverageLossPerIteration();
            // Stejný non-null kontrakt jako v synchronní verzi: guard + non-null lokály,
            // výstup z návratové hodnoty GetResultsAsync (nikdy null).
            if (ValidDataInputs == null || ValidDataCurrentOutput == null)
            {
                throw new InvalidOperationException("Valid data nejsou nastavena (DividingDataIntoDatasets) — nelze spočítat valid loss.");
            }

            Tensor validInputs = ValidDataInputs;
            Tensor validOutputs = ValidDataCurrentOutput;

            if (_model.Context.SequenceTrain)
            {
                for (int i = 0; i < validInputs.Shape[0]; i++)
                {
                    _model.ResetSequence();
                    for (int j = 0; j < validInputs.Shape[1]; j++)
                    {
                        Tensor output = await _model.GetResultsAsync(validInputs.GetTensorValue([i, j]));
                        lossFunc.CalculateLoss(output.Data, validOutputs.GetTensorValue([i, j]).Data);
                    }
                }
            }

            else
            {
                Task[] tasks = new Task[validInputs.Shape[0]];

                for (int i = 0; i < validInputs.Shape[0]; i++)
                {
                    int index = i;
                    tasks[index] = Task.Run(async () =>
                    {
                        Tensor output = await _model.GetResultsAsync(validInputs.GetTensorValue([index]));
                        double lossOutput = lossFunc.CalculateAndGetLoss(output.Data, validOutputs.GetTensorValue([index]).Data);

                        if (lossOutput is not double.NaN)
                        {
                            layerOutput.Add(lossOutput);
                        }
                    });

                }
                await Task.WhenAll(tasks);

                lossFunc.ResetAverageLossPerIteration();
            }



            ExtraControlFunc(loss);
            if (_model.Context.SequenceTrain)
            {
                ConsoleControler.ShowEpochInfo(_model, loss);
            }
            else
            {
                ConsoleControler.ShowEpochInfo(_model, loss, layerOutput.Sum() / layerOutput.Count());
            }

        }

        private void ExtraControlFunc(double loss)
        {
            Loss lossFunc = _model.Context.Loss;

            // Kromě NaN hlídáme i nekonečno — zdivergovaný trénink projde typicky přes ±∞
            // dřív, než se dostane k NaN, a v obou případech jsou váhy stejně nepoužitelné.
            double validLossNow = lossFunc.GetAverageLossPerIteration();

            if (!double.IsFinite(loss) || !double.IsFinite(validLossNow))
            {
                // graf jen když si ho uživatel přeje — jinak by každý zdivergovaný běh
                // (a v AutoML jich budou desítky) tiše zapsal loss.png do pracovního adresáře
                if (ShowLossChartInTrainLoop)
                {
                    GraphPlotter.ShowLossGraph(_listOfepoch.ToArray(), _listOfTrainLoss.ToArray(), _listOfValidLoss.ToArray());
                }

                ConsoleControler.ErrorHandler("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values ​​in the training parameters (for example: learning rate or hyperammetry )", true);

                throw new TrainingDivergedException(_epoch, double.IsFinite(loss) ? validLossNow : loss);
            }

            if (_epoch >= (_reportInterval * NumberOfSkipFristEpochInPlotter))
            {
                _listOfepoch.Add((int)_epoch);
                _listOfTrainLoss.Add(loss);
                _listOfValidLoss.Add(lossFunc.GetAverageLossPerIteration());
            }

            double validLoss = lossFunc.GetAverageLossPerIteration();

            // zlepšení = pokles valid loss o víc než minDelta. Jen tehdy aktualizujeme nejlepší:
            // nejlepší model si držíme VŽDY (in-memory snapshot) — na konci se obnoví;
            // na disk zapisujeme jen když AutoSave (crash/NaN recovery, reuse na příště).
            if (validLoss < _lowestLoss - EarlyStoppingMinDelta)
            {
                _lowestLoss = validLoss;
                _bestEpoch = _epoch;
                _bestSnapshot = _model.SaveAsJsonString();
                _reportsWithoutImprovement = 0;

                if (AutoSaveInTrainLoop)
                {
                    _model.SaveAsJson(AutoSaveInTrainLoopFileName);
                }
            }
            else
            {
                _reportsWithoutImprovement++;
                if (EarlyStoppingEnabled && _reportsWithoutImprovement >= EarlyStoppingPatience)
                {
                    _stopEarly = true;
                }
            }
        }

        private void Propagation(Tensor[] layerGradients)
        {
            for (int i = 0; i < _model.Layers.Layers.Count(); i++)
            {
                _model.Layers.Layers[i].BackPropagation(layerGradients[i]);
            }
        }

        private async Task PropagationAsync(Tensor[] layerGradients)
        {
            Task[] tasks = new Task[_model.Layers.Layers.Count()];

            for (int i = 0; i < _model.Layers.Layers.Count(); i++)
            {
                int index = i;
                tasks[index] = Task.Run(async () =>
                {
                    await _model.Layers.Layers[index].BackPropagationAsync(layerGradients[index]);
                });
            }

            await Task.WhenAll(tasks);
        }

    }
    
}

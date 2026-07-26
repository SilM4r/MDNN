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
        private uint _totalEpoch;
        private uint _sizeOfMiniBatch;
        private double _lowestLoss;
        private uint _bestEpoch;
        private string? _bestSnapshot;   // nejlepší model v paměti (JSON snapshot), obnovuje se na konci
        private uint _reportsWithoutImprovement;   // early stopping: počet valid reportů bez zlepšení
        private bool _stopEarly;                    // early stopping: flag k ukončení smyčky
        private readonly Random _rnd = new Random();

        private List<int> _listOfepoch = [];
        private List<double> _listOfValidLoss = [];
        private List<double> _listOfTrainLoss = [];

        public uint NumberOfSkipFristEpochInPlotter = 0;
        public uint NumberOfShowEpochInConsole = 100;

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

        private Tensor? _trainDataInputs;
        private Tensor? _testDataInputs;
        private Tensor? _validDataInputs;

        private Tensor? _trainDataCurrentOutput;
        private Tensor? _testDataCurrentOutput;
        private Tensor? _validDataCurrentOutput;
        public Tensor? TestDataInputs
        { 
            get => _testDataInputs; 
            set => _testDataInputs = value; 
        }
        public Tensor? ValidDataInputs
        {
            get => _validDataInputs;
            set => _validDataInputs = value;
        }
        public Tensor? TestDataCurrentOutput
        {
            get => _testDataCurrentOutput;
            set => _testDataCurrentOutput = value;
        }
        public Tensor? ValidDataCurrentOutput
        {
            get => _validDataCurrentOutput;
            set => _validDataCurrentOutput = value;
        }

        public uint CurrentEpoch
        {
            get { return _epoch; }
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

            _epoch++;
            foreach (Layer layer in _model.Layers.Layers)
            {
                layer.UpdateParams();
            }
        }

        public async Task UpdateParamsAsync()
        {
            CheckLayersAreNotEmpty();

            _epoch++;
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

                        maxScore += output.Length;

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

                for (uint miniBatch = 0; miniBatch < sizeOfMiniBatch; miniBatch++)
                {
                    if (_trainDataInputs != null)
                    {
                        int num = _rnd.Next(_trainDataInputs.Shape[0]);
                        if (!_model.Context.SequenceTrain)
                        {
                            Fit(_trainDataInputs.GetTensorValue([num]), _trainDataCurrentOutput?.GetTensorValue([num]) ?? throw new InvalidOperationException("_trainDataCurrentOutput je null — trénovací data nebyla rozdělena (DividingDataIntoDatasets)."));
                        }

                        else
                        {
                            _model.ResetSequence();
                            for (int i = 0; i < _trainDataInputs.Shape[1]; i++)
                            {
                                Fit(_trainDataInputs.GetTensorValue([num,i]), _trainDataCurrentOutput?.GetTensorValue([
                                    num, i
                                ]) ?? throw new InvalidOperationException("_trainDataCurrentOutput je null — trénovací data nebyla rozdělena (DividingDataIntoDatasets)."));
                            }
                        }
                    }
                    else
                    {
                        throw new Exception("_trainDataInputs is Empty");
                    }
                }

                if (_model.Context.SequenceTrain)
                {
                    _model.ResetSequence();
                }

                UpdateParams();

                if (epoch % (_totalEpoch/ NumberOfShowEpochInConsole) == 0)
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
                for (uint miniBatch = 0; miniBatch < sizeOfMiniBatch; miniBatch++)
                {
                    // Skutečný runtime guard (ne Debug.Assert — ten se v Release vykompiluje pryč,
                    // takže by async cesta neměla ochranu, na rozdíl od synchronního TrainLoop).
                    // Lokální kopie navíc udrží non-null null-state i přes await.
                    if (_trainDataInputs == null || _trainDataCurrentOutput == null)
                    {
                        throw new InvalidOperationException(
                            "_trainDataInputs/_trainDataCurrentOutput je null — trénovací data nebyla rozdělena (DividingDataIntoDatasets).");
                    }

                    Tensor trainInputs = _trainDataInputs;
                    Tensor trainOutputs = _trainDataCurrentOutput;

                    int num = _rnd.Next(trainInputs.Shape[0]);
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
                    }
                }

                if (_model.Context.SequenceTrain)
                {
                    _model.ResetSequence();
                }

                await UpdateParamsAsync();

                if (epoch % (_totalEpoch / NumberOfShowEpochInConsole) == 0)
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

            this._sizeOfMiniBatch = sizeOfMiniBatch;
            _totalEpoch = numberOfEpoch;
            
            if (NumberOfShowEpochInConsole <= 0)
                NumberOfShowEpochInConsole = 1;
            else if (NumberOfShowEpochInConsole > _totalEpoch)
                NumberOfShowEpochInConsole = _totalEpoch;


            _model.Context.InputShape = [inputsValues[0].Length];
            _model.Layers.SetInputSizeForFirstLayer();

            CheckLayersAreNotEmpty();

            for (; epoch < _totalEpoch; epoch++)
            {
                for (uint miniBatch = 0; miniBatch < sizeOfMiniBatch; miniBatch++)
                {
                    int num = _rnd.Next(inputsValues.Count());

                    Fit(new Tensor(inputsValues[num]), new Tensor(currentOutputValues[num]));
                }

                UpdateParams();

                if (epoch % (_totalEpoch / NumberOfShowEpochInConsole) == 0)
                {
                    double loss = _model.Context.Loss.GetAverageLossPerIteration();
                    if (loss is not double.NaN)
                    {
                        if (loss < minLoss)
                        {
                            minLoss = loss;
                            _model.Note = minLoss.ToString(CultureInfo.InvariantCulture);
                            if (AutoSaveInTrainLoop)
                            {
                                _model.SaveAsJson(AutoSaveInTrainLoopFileName);
                            }
                            
                        }
                    }
                    else
                    {
                        Console.WriteLine("Error: Nan number");
                        return;
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

            DividingDataIntoDatasets(inputsValues, currentOutputValues);

            _listOfepoch = [];
            _listOfValidLoss = [];
            _listOfTrainLoss = [];

            this._sizeOfMiniBatch = sizeOfMiniBatch;
            _totalEpoch = numberOfEpoch;

            _lowestLoss = double.MaxValue;
            _bestEpoch = 0;
            _bestSnapshot = null;
            _reportsWithoutImprovement = 0;
            _stopEarly = false;

            if (NumberOfShowEpochInConsole <= 0)
            {
                NumberOfShowEpochInConsole = 1;
            }
            else if (NumberOfShowEpochInConsole > _totalEpoch)
            {
                NumberOfShowEpochInConsole = _totalEpoch;
            }
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

        public void DividingDataIntoDatasets(Tensor inputsValues, Tensor currentOutputValues)
        {
            int trainDataSize;
            int validDataSize;
            int testDataSize;

            int totalSize = inputsValues.Shape[0];

            if (_validDataInputs == null && _testDataInputs == null)
            {
                trainDataSize = (int)(totalSize * 0.7);
                validDataSize = (int)(totalSize * 0.15);
                testDataSize = totalSize - trainDataSize - validDataSize;

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

                validDataSize = (int)(oldValidSize * 0.8);
                testDataSize = oldValidSize - validDataSize;

                _trainDataInputs = inputsValues;
                _trainDataCurrentOutput = currentOutputValues;

                // Test se musí vyříznout z PŮVODNÍHO valid tensoru; kdyby se _validDataInputs
                // nejdřív přepsal na Slice(0, validDataSize), test-slice od offsetu validDataSize
                // by byl mimo rozsah zmenšeného tensoru → "Invalid slice range!".
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
                trainDataSize = (int)(totalSize * 0.8);
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

        public void ShuffleTensor(Tensor tensorA, Tensor tensorB, out Tensor shuffledA, out Tensor shuffledB)
        {
            if (tensorA.Shape[0] != tensorB.Shape[0])
                throw new ArgumentException("Oba tensory musí mít stejný počet vzorků");

            int batchSize = tensorA.Shape[0];
            int[] indices = Enumerable.Range(0, batchSize).OrderBy(_ => Random.Shared.Next()).ToArray();


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

            if (loss is double.NaN || lossFunc.GetAverageLossPerIteration() is double.NaN)
            {
                GraphPlotter.ShowLossGraph(_listOfepoch.ToArray(), _listOfTrainLoss.ToArray(), _listOfValidLoss.ToArray());
                ConsoleControler.ErrorHandler("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values ​​in the training parameters (for example: learning rate or hyperammetry )", true);
                return;
            }

            if (_epoch >= ((_totalEpoch / NumberOfShowEpochInConsole) * NumberOfSkipFristEpochInPlotter))
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

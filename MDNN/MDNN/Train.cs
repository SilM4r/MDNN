using System.Data;
using System.Globalization;
using MDNN.Layers;
using MDNN.Losses;
namespace MDNN
{
    /// <summary>
    /// Training of a <see cref="NeuralNetwork"/>: training loops, manual training steps,
    /// dataset splitting, evaluation, early stopping and training settings.
    /// </summary>
    public class Train
    {
        private readonly NeuralNetwork _model;

        private uint _epoch;
        private ulong _optimizerSteps;
        private uint _totalEpoch;
        private uint _sizeOfMiniBatch;
        private double _lowestLoss;
        private uint _bestEpoch;
        private string? _bestSnapshot;   // best model in memory (JSON snapshot), restored at the end
        private uint _reportsWithoutImprovement;   // early stopping: number of valid reports without improvement
        private bool _stopEarly;                    // early stopping: flag to exit the loop

        // A single source of randomness for the whole model (sample order and dataset shuffling).
        // Previously there was a separate `new Random()` here and ShuffleTensor used
        // Random.Shared — three independent sources meant an experiment could not be repeated.
        private Random Rnd => _model.Context.Random;

        private List<int> _listOfepoch = [];
        private List<double> _listOfValidLoss = [];
        private List<double> _listOfTrainLoss = [];

        /// <summary>Number of initial reports left out of the loss chart.</summary>
        public uint LossChartSkipFirstReports { get; set; } = 0;
        /// <summary>How many progress reports to print during one training run (default 100).</summary>
        public uint ConsoleReportCount { get; set; } = 100;

        // Report every N epochs. Previously this was computed inline as
        // `_totalEpoch / ConsoleReportCount` and the clamp was written DIRECTLY into the public
        // ConsoleReportCount, permanently overwriting the user's setting
        // (after 5 epochs it stayed 5 even for a following 5000-epoch run).
        private uint _reportInterval = 1;

        // internal for tests: "how many reports per run" is otherwise visible only in the console,
        // and scraping Console.Out would be fragile (tests run in parallel).
        internal uint ReportInterval => _reportInterval;

        // State for the remaining-time estimate. Previously these were statics in ConsoleController,
        // so a second training run in the process saw values from the first, got pastEpochs == 0 and
        // (subTime * n) / 0 threw OverflowException. Now per-instance and reset at the start of a run.
        internal DateTime LastReportTime = DateTime.MinValue;
        internal uint LastReportEpoch = 0;

        /// <summary>Whether to draw the loss chart after training (requires <see cref="LossChart"/> to be set).</summary>
        public bool ShowLossChart { get; set; } = true;

        // The core library cannot render — ScottPlot lives in the separate MDNN.Plotting package,
        // which plugs in its GraphPlotter via model.Train.UseLossChart().
        /// <summary>
        /// Draws the training progress chart from (epochs, training loss, validation loss).
        /// When null, no chart is drawn and <see cref="ShowLossChart"/> has no effect.
        /// </summary>
        public Action<int[], double[], double[]>? LossChart { get; set; }

        private void DrawLossChart()
        {
            if (ShowLossChart && LossChart != null)
            {
                LossChart(_listOfepoch.ToArray(), _listOfTrainLoss.ToArray(), _listOfValidLoss.ToArray());
            }
        }
        /// <summary>Whether to evaluate the model on all datasets (<see cref="EvaluateAll"/>) after the training loop.</summary>
        public bool TestAfterTraining { get; set; } = true;
        /// <summary>Whether to print the model summary at the start of the training loop.</summary>
        public bool ShowModelInfo { get; set; } = true;
        /// <summary>Whether to save the best model so far to disk during training (see <see cref="AutoSaveFileName"/>).</summary>
        public bool AutoSave { get; set; } = true;

        /// <summary>File name used by <see cref="AutoSave"/>.</summary>
        public string AutoSaveFileName { get; set; } = "AutoSave";

        // Early stopping: stop training when the valid loss has not improved for `patience` valid reports.
        // Thanks to the in-memory snapshot the best model is restored at the end anyway (restore-best for free).
        /// <summary>Enables early stopping (opt-in, off by default).</summary>
        public bool EarlyStoppingEnabled { get; set; } = false;      // opt-in, off by default → backward compatible
        /// <summary>Number of valid reports without improvement after which training stops.</summary>
        public uint EarlyStoppingPatience { get; set; } = 10;        // valid reports without improvement = stop
        /// <summary>Minimum decrease of the valid loss that counts as an improvement.</summary>
        public double EarlyStoppingMinDelta { get; set; } = 0.0;     // min. loss decrease that counts as an improvement

        // Dataset split ratios. null = computed from the remainder (shared equally among the unset ones).
        // When all are null, 0.7 / 0.15 / 0.15 is used (the long-standing default).
        // E.g. Train=0.6, Valid=0.2, Test=null → test becomes 0.2.
        /// <summary>Share of the data used for training; null = computed from the remainder.</summary>
        public double? TrainSplitRatio { get; set; }
        /// <summary>Share of the data used for validation; null = computed from the remainder.</summary>
        public double? ValidSplitRatio { get; set; }
        /// <summary>Share of the data used for the final test; null = computed from the remainder.</summary>
        public double? TestSplitRatio { get; set; }

        private Tensor? _trainDataInputs;
        private Tensor? _testDataInputs;
        private Tensor? _validDataInputs;

        private Tensor? _trainDataCurrentOutput;
        private Tensor? _testDataCurrentOutput;
        private Tensor? _validDataCurrentOutput;
        // Datasets supplied by the USER via SetDatasets(). Kept separate from the working fields
        // above, because those are overwritten (sliced) during a run — without this distinction a
        // second TrainLoop on the same model saw leftovers of the first run and trained on the
        // WHOLE dataset including the test part.
        private LabeledData? _explicitTrain;
        private LabeledData? _explicitValid;
        private LabeledData? _explicitTest;

        // ALL read-only. Previously valid/test could be set through four independent setters,
        // which allowed inputs without targets and bloated SplitDatasets into four branches
        // depending on what the user had filled in. The only way in is now SetDatasets(),
        // where LabeledData keeps inputs and targets together.
        /// <summary>Training inputs of the current run (after splitting); null before training.</summary>
        public Tensor? TrainInputs => _trainDataInputs;
        /// <summary>Training targets of the current run (after splitting); null before training.</summary>
        public Tensor? TrainTargets => _trainDataCurrentOutput;
        /// <summary>Validation inputs of the current run (after splitting); null before training.</summary>
        public Tensor? ValidInputs => _validDataInputs;
        /// <summary>Validation targets of the current run (after splitting); null before training.</summary>
        public Tensor? ValidTargets => _validDataCurrentOutput;
        /// <summary>Test inputs of the current run; null before training or when there is no test set.</summary>
        public Tensor? TestInputs => _testDataInputs;
        /// <summary>Test targets of the current run; null before training or when there is no test set.</summary>
        public Tensor? TestTargets => _testDataCurrentOutput;

        /// <summary>True when the datasets were set explicitly via <see cref="SetDatasets"/>.</summary>
        public bool HasExplicitDatasets => _explicitTrain != null;

        // The missing set is carved out — but from DIFFERENT places, depending on which one is missing:
        //
        //   SetDatasets(train, valid)        → TEST is carved from valid (train stays whole)
        //   SetDatasets(train, null, test)   → VALID is carved from TRAIN (typical for MNIST)
        //   SetDatasets(train, valid, test)  → nothing is carved
        //
        //   SetDatasets(train)               → both valid AND test are carved from train (70/15/15)
        //
        // The asymmetry is intentional: the test set is the final unbiased estimate, so it is NEVER
        // carved from. A missing valid set comes from train, as is common practice. A missing test
        // set comes from valid, so the user's training set stays whole. When both are missing,
        // train is the only source.
        //
        //
        // To avoid losing part of the validation set to a test set you will not use, set
        // `TestAfterTraining = false` — then no test is carved from valid and valid stays whole.
        // You can then test the model yourself with Evaluate(...).
        /// <summary>
        /// Sets your own datasets. A missing validation or test set is carved from another set.
        /// Then train with the data-less overload <c>TrainLoop(epochs, miniBatchSize)</c>.
        /// </summary>
        /// <param name="train">Training set.</param>
        /// <param name="valid">Validation set; when null, it is carved from <paramref name="train"/>.</param>
        /// <param name="test">Test set; when null, it is carved from <paramref name="valid"/> (only if <see cref="TestAfterTraining"/> is on), or from <paramref name="train"/> when <paramref name="valid"/> is null too.</param>
        public void SetDatasets(LabeledData train, LabeledData? valid = null, LabeledData? test = null)
        {
            ArgumentNullException.ThrowIfNull(train);

            _explicitTrain = train;
            _explicitValid = valid;
            _explicitTest = test;
        }

        // NOTE the change of meaning: previously `UpdateParams()` incremented this counter, and since
        // it was called once per "epoch" it came out the same. With conventional epochs there are as
        // many optimizer steps per epoch as there are batches, so the counters diverged.
        /// <summary>
        /// Number of COMPLETED epochs, i.e. full passes over the training set. Manual training
        /// (Fit + UpdateParams in your own loop) has no epochs, so it stays 0; see <see cref="OptimizerSteps"/>.
        /// </summary>
        public uint CurrentEpoch
        {
            get { return _epoch; }
        }

        /// <summary>Number of optimizer steps (calls to <see cref="UpdateParams"/>) over the model's lifetime.</summary>
        public ulong OptimizerSteps
        {
            get { return _optimizerSteps; }
        }

        /// <summary>
        /// Lowest validation loss so far (early-stopping indicator). <c>double.MaxValue</c> means
        /// no validation report has happened yet.
        /// </summary>
        public double LowestValidLoss
        {
            get { return _lowestLoss; }
        }

        /// <summary>Epoch in which <see cref="LowestValidLoss"/> was reached.</summary>
        public uint BestEpoch
        {
            get { return _bestEpoch; }
        }

        /// <summary>Target epoch of the current run (epochs completed before the run plus the epochs of this run).</summary>
        public uint TotalEpoch => _totalEpoch;

        /// <summary>Minibatch size of the last training run.</summary>
        public uint MiniBatchSize => _sizeOfMiniBatch;

        private int _workers = 1;

        /// <summary>
        /// Number of threads that compute the samples of one batch in <c>TrainLoop</c> and <c>SimpleTrainLoop</c>.
        /// 1 (default) = sequential.
        /// </summary>
        /// <remarks>
        /// With more threads, each has its own model replica and the gradients are summed after the batch.
        /// Only the work INSIDE a batch is parallelized, so more threads than samples per batch add
        /// nothing — and with the default batch size of 1 nothing gets faster.
        ///
        /// Reproducibility: the same seed and the same Workers give a bit-for-bit identical result
        /// on any machine. Compared with Workers = 1 the result differs by rounding (double addition
        /// is not associative), and with dropout and other masks too — each replica has its own
        /// random generator derived from the model's seed.
        ///
        /// The manual Fit + UpdateParams loop always runs sequentially.
        /// </remarks>
        public int Workers
        {
            get => _workers;
            set
            {
                if (value < 1)
                    throw new ArgumentOutOfRangeException(nameof(Workers), value,
                        "The number of workers must be at least 1 (1 = sequential training).");
                _workers = value;
            }
        }

        internal Train(NeuralNetwork model)
        {
            this._model = model;
            _epoch = 0;
            _totalEpoch = 0;
            _sizeOfMiniBatch = 1;
        }

        internal Train(NeuralNetwork model, uint epoch, uint totalEpoch, uint miniBatchSize)
        {
            this._model = model;
            this._epoch = epoch;
            this._totalEpoch = totalEpoch;
            this._sizeOfMiniBatch=miniBatchSize;
        }

        /// <summary>
        /// One manual training step for a single sample: forward pass in training mode followed by
        /// backpropagation. Gradients are only accumulated; apply them with <see cref="UpdateParams"/>.
        /// </summary>
        /// <param name="inputs">Input of one sample.</param>
        /// <param name="targets">Expected output for the sample.</param>
        /// <returns>The model output for <paramref name="inputs"/>.</returns>
        public Tensor Fit(Tensor inputs, Tensor targets)
        {
            if (_model.Layers.Layers[0].InputShape[0] <= 0)
            {
                _model.Context.InputShape = inputs.Shape;
                _model.Layers.SetInputShape();
            }

            CheckLayersAreNotEmpty();

            // Training mode only for this pass. It controls dropout: inside Fit neurons are
            // dropped, a standalone GetResults (prediction, valid loss) runs without it —
            // otherwise the same input would give a different answer each time.
            // Restored in finally so an exception mid-training does not leave the model "training".
            bool wasTraining = _model.Context.IsTraining;
            _model.Context.IsTraining = true;

            Tensor output;
            try
            {
                output = _model.GetResults(inputs);
            }
            finally
            {
                _model.Context.IsTraining = wasTraining;
            }

            BackPropagation(targets);

            return output;
        }

        /// <summary>
        /// Backpropagates the loss against <paramref name="targets"/> for the last forward pass and
        /// accumulates the gradients in the layers (does not update the weights).
        /// </summary>
        /// <param name="targets">Expected output for the last forward pass.</param>
        public void BackPropagation(Tensor targets)
        {
            CheckLayersAreNotEmpty();

            // Convention B: a single pass. Previously the whole array of gradients was computed
            // first and applied in a second pass; now each layer accumulates its own gradients
            // directly and passes only ∂L/∂input down.
            Gradient.RunBackward(targets, _model);
        }

        // Must turn on IsTraining just like Fit — this method IS the first half of Fit, so
        // "splitting Fit into two steps" must give the same result. Without it manual training
        // would silently run without dropout.
        /// <summary>
        /// Training forward pass (dropout active). The first half of <see cref="Fit"/>; follow it with
        /// <see cref="BackPropagation"/>. For prediction use <c>model.GetResults()</c>, which keeps dropout off.
        /// </summary>
        /// <param name="inputs">Input of one sample.</param>
        /// <returns>The model output.</returns>
        public Tensor FeedForward(Tensor inputs)
        {
            bool wasTraining = _model.Context.IsTraining;
            _model.Context.IsTraining = true;

            try
            {
                return _model.GetResults(inputs);
            }
            finally
            {
                _model.Context.IsTraining = wasTraining;
            }
        }

        /// <summary>
        /// One optimizer step: updates the parameters of all layers from the accumulated gradients.
        /// </summary>
        public void UpdateParams()
        {
            CheckLayersAreNotEmpty();

            _optimizerSteps++;
            foreach (Layer layer in _model.Layers.Layers)
            {
                layer.UpdateParams();
            }
        }

        /// <summary>
        /// Evaluates the model on the training, validation and (if present) test sets of the last
        /// run and prints the scores.
        /// </summary>
        /// <exception cref="InvalidOperationException">The training or validation set is missing.</exception>
        public void EvaluateAll()
        {
            if (_trainDataInputs == null || _trainDataCurrentOutput == null ||
                _validDataInputs == null || _validDataCurrentOutput == null)
            {
                throw new InvalidOperationException(
                    "The training or validation dataset is missing — training has not run yet, " +
                    "or the datasets were not set with SetDatasets().");
            }

            (int, int) trainScore = Evaluate(_trainDataInputs, _trainDataCurrentOutput, false);
            (int, int) validationScore = Evaluate(_validDataInputs, _validDataCurrentOutput, false);

            // The test set is optional: if the user did not supply it via SetDatasets(), the
            // summary simply omits it. Making one up from the validation set would be worse than
            // not showing it at all.
            (int, int)? testScore = (_testDataInputs != null && _testDataCurrentOutput != null)
                ? Evaluate(_testDataInputs, _testDataCurrentOutput, false)
                : null;

            ConsoleController.ShowDatasetScores(testScore, trainScore, validationScore);
                    
        }

        /// <summary>
        /// Counts correct predictions on the given data. For a whole-layer output activation (e.g. Softmax)
        /// the argmax is compared, otherwise the rounded values. In sequential mode each time step counts.
        /// </summary>
        /// <param name="inputs">Input samples (or sequences).</param>
        /// <param name="targets">Expected outputs.</param>
        /// <param name="showInConsole">Whether to print the score.</param>
        /// <returns>The number of correct predictions and the total number evaluated.</returns>
        public (int Correct, int Total) Evaluate(Tensor inputs, Tensor targets, bool showInConsole = true)
        {
            if (_model.Layers.Layers[0].InputShape[0] == 0)
            {
                if (_model.Context.SequenceTrain)
                {
                    _model.Context.InputShape = inputs.GetSubTensor([0,0]).Shape;
                }

                else
                {
                    _model.Context.InputShape = inputs.GetSubTensor([0]).Shape;
                }

                _model.Layers.SetInputShape();
            }
            CheckLayersAreNotEmpty();

            int score = 0;
            int maxScore = 0;

            // All outputs are computed first (in parallel when Workers > 1), then the score is
            // summed sequentially in sample order — see PredictSamples.
            if (_model.Context.SequenceTrain)
            {
                double[][][] outputs = PredictSequences(inputs);

                for (int i = 0; i < outputs.Length; i++)
                {
                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        // We count TIME STEPS, not output neurons. Previously this was
                        // `maxScore += output.Length` while score grew by 1 per step, so
                        // score/maxScore was meaningless in sequential mode.
                        maxScore++;

                        if (IsPredictionCorrect(outputs[i][j], targets.GetSubTensor([i, j]).Data))
                        {
                            score++;
                        }
                    }
                }
                _model.ResetSequence();
            }

            else
            {
                double[][] outputs = PredictSamples(inputs);
                maxScore = outputs.Length;

                for (int i = 0; i < outputs.Length; i++)
                {
                    if (IsPredictionCorrect(outputs[i], targets.GetSubTensor([i]).Data))
                    {
                        score++;
                    }
                }
            }
            if (showInConsole)
                ConsoleController.ShowScore(score, maxScore);
            
            return (score, maxScore);
        }

        // The data is not passed here — passing it a second time when the model already has it
        // would be contradictory.
        /// <summary>
        /// Trains on the datasets set via <see cref="SetDatasets"/>: full passes over the training set
        /// in minibatches, with periodic validation reports. At the end the best model (lowest valid loss)
        /// is restored.
        /// </summary>
        /// <param name="epochs">Number of epochs of this run (added to the epochs already completed).</param>
        /// <param name="miniBatchSize">Number of samples per optimizer step.</param>
        /// <param name="isSequence">Train on sequences (inputs shaped [sequence, step, …]).</param>
        /// <exception cref="TrainingDivergedException">The loss became NaN or infinite.</exception>
        public void TrainLoop(uint epochs, uint miniBatchSize = 1, bool isSequence = false)
        {
            LabeledData source = CurrentDataSource();
            RunTrainLoop(source.Inputs, source.Targets, epochs, miniBatchSize, isSequence);
        }

        // Does exactly the same as
        //     SetDatasets(new LabeledData(inputs, targets));
        //     TrainLoop(epochs, miniBatchSize, isSequence);
        // so there is only one path inside; this is a convenience, not a second mechanism.
        /// <summary>
        /// Shortcut: splits the given data into train/valid/test by the split ratios and trains on it.
        /// </summary>
        /// <param name="inputs">All input samples.</param>
        /// <param name="targets">All expected outputs.</param>
        /// <param name="epochs">Number of epochs of this run.</param>
        /// <param name="miniBatchSize">Number of samples per optimizer step.</param>
        /// <param name="isSequence">Train on sequences (inputs shaped [sequence, step, …]).</param>
        public void TrainLoop(Array inputs, Array targets, uint epochs, uint miniBatchSize = 1, bool isSequence = false)
        {
            SetDatasets(new LabeledData(inputs, targets));
            TrainLoop(epochs, miniBatchSize, isSequence);
        }

        // Where the data for this run comes from — the train set given to SetDatasets()
        // (the shortcut TrainLoop overload sets it too).
        private LabeledData CurrentDataSource()
        {
            return _explicitTrain ?? throw new InvalidOperationException(
                "No data has been set. Use SetDatasets(train, valid, test), " +
                "or the shortcut TrainLoop(inputs, targets, ...).");
        }

        private void RunTrainLoop(Tensor tensorInputsValues, Tensor tensorCurrentOutputValues, uint epochs, uint miniBatchSize = 1, bool isSequence = false)
        {
            PreparationForTrainLoop(tensorInputsValues, tensorCurrentOutputValues, epochs, miniBatchSize, isSequence);
            if (ShowModelInfo)
            {
                _model.Info();
            }
            
            uint firstEpoch = this._epoch;

            for (uint epoch = firstEpoch; epoch < _totalEpoch; epoch++)
            {
                _model.Context.Loss.ResetAverageLossPerIteration();

                if (_trainDataInputs == null || _trainDataCurrentOutput == null)
                {
                    throw new InvalidOperationException(
                        "_trainDataInputs/_trainDataCurrentOutput is null — the training data has not been split (SplitDatasets).");
                }

                RunOneEpoch(_trainDataInputs, _trainDataCurrentOutput);

                // Epochs are counted here; UpdateParams() counts optimizer steps,
                // of which there are as many per epoch as there are batches.
                _epoch = epoch + 1;

                // from the start of the RUN, not from zero — see SimpleTrainLoop
                if ((epoch - firstEpoch) % _reportInterval == 0)
                {
                    TrainLoopControlFunc();
                    if (_stopEarly)
                    {
                        ConsoleController.ShowEarlyStopping(_epoch, EarlyStoppingPatience, _lowestLoss, _bestEpoch);
                        break;
                    }
                }
            }

            // At the end the model holds the BEST weights (lowest valid loss), not the last
            // epoch — the in-memory snapshot is restored in place (identity and datasets stay).
            // This happens regardless of TestAfterTraining: previously the restore was inside the
            // condition below, so with the test turned off the model kept the last (worse) state
            // even though early stopping and the "Best valid loss" output reported the best one.
            if (_bestSnapshot != null)
            {
                _model.LoadWeightsFromString(_bestSnapshot);
            }

            if (TestAfterTraining)
            {
                EvaluateAll();
            }
            DrawLossChart();
        }

        /// <summary>
        /// A simpler training loop on 1D samples without a validation set: full passes over the
        /// data in minibatches, reporting the training loss. With <see cref="AutoSave"/> the model
        /// is saved whenever the training loss improves.
        /// </summary>
        /// <param name="inputs">Input samples.</param>
        /// <param name="targets">Expected outputs.</param>
        /// <param name="epochs">Number of epochs of this run.</param>
        /// <param name="miniBatchSize">Number of samples per optimizer step.</param>
        /// <exception cref="TrainingDivergedException">The loss became NaN or infinite.</exception>
        public void SimpleTrainLoop(double[][] inputs, double[][] targets, uint epochs, uint miniBatchSize = 1)
        {
            uint epoch = this._epoch;

            // Lowest training loss reached — AutoSave depends on it. Previously it started at 100,
            // so a task with a loss above 100 (unscaled targets, MSE as a sum) saved nothing
            // until the loss fell below that threshold.
            double minLoss = double.MaxValue;

            if (miniBatchSize < 1)
            {
                throw new ArgumentException("The minibatch size must be at least 1.", nameof(miniBatchSize));
            }

            this._sizeOfMiniBatch = miniBatchSize;

            // NUMBER of epochs of this run, not the target epoch — see PreparationForTrainLoop
            uint firstEpoch = epoch;
            _totalEpoch = firstEpoch + epochs;

            ResolveReportInterval(epochs);


            _model.Context.InputShape = [inputs[0].Length];
            _model.Layers.SetInputShape();

            CheckLayersAreNotEmpty();

            for (; epoch < _totalEpoch; epoch++)
            {
                // same semantics as TrainLoop: a full pass in batches, not random draws
                int[] order = ShuffledIndices(inputs.Length);
                int batchSize = (int)miniBatchSize;
                ReplicaPool? pool = CreateReplicaPool();

                for (int start = 0; start < order.Length; start += batchSize)
                {
                    int end = Math.Min(start + batchSize, order.Length);

                    RunBatch(pool, order, start, end, (train, num) =>
                        train.Fit(new Tensor(inputs[num]), new Tensor(targets[num])));
                }

                _epoch = epoch + 1;

                // The modulo counts from the start of the RUN, not from zero — otherwise in a
                // continued training run the reports would land on absolute multiples and the
                // first epoch of the run would not be reported.
                if ((epoch - firstEpoch) % _reportInterval == 0)
                {
                    double loss = _model.Context.Loss.GetAverageLossPerIteration();

                    // Same policy as in TrainLoop (ExtraControlFunc): divergence = an exception,
                    // not a silent `return`. Previously this only printed "Error: Nan number" and
                    // the method ended as if successful — the caller could not tell "trained"
                    // from "fell apart in the first epoch".
                    if (!double.IsFinite(loss))
                    {
                        ConsoleController.ShowError("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values in the training parameters (for example the learning rate or other hyperparameters).", true);
                        throw new TrainingDivergedException(_epoch, loss);
                    }

                    if (loss < minLoss)
                    {
                        minLoss = loss;
                        _model.Note = minLoss.ToString(CultureInfo.InvariantCulture);
                        if (AutoSave)
                        {
                            _model.SaveAsJson(AutoSaveFileName);
                        }
                    }

                    ConsoleController.ShowEpochInfo(_model);
                }
            }
        }

        private void PreparationForTrainLoop(Tensor inputs, Tensor targets, uint epochs, uint miniBatchSize = 1, bool isSequence = false)
        {
            CheckTensorShapes(inputs, targets);

            // Shuffling only makes sense when train/valid/test are still to be carved from this
            // data — the order decides what ends up where. For explicitly given datasets it would
            // be pointless (the split is fixed) and would touch tensors the user owns.
            // RunOneEpoch shuffles the order within each epoch anyway.
            if (_explicitValid == null && _explicitTest == null)
            {
                ShuffleTensor(inputs, targets, out inputs, out targets);
            }

            if (isSequence)
            {
                _model.Context.SequenceTrain = true;
            }

            if (_model.Context.SequenceTrain)
            {
                if (inputs.Shape.Length == 1)
                {
                    throw new ArgumentException("For sequential training the inputs must be at least a two-dimensional array.");
                }

                if (inputs.Shape.Length == 2)
                {
                    inputs.Reshape([inputs.Shape[0], inputs.Shape[1], 1]);
                    targets.Reshape([inputs.Shape[0], inputs.Shape[1], 1]);
                }

                // [sequence, step, sample…], where a sample has at most 3 dimensions ([h, w, c] for Conv)
                // → at most 5D. The condition always did this; only the message used to say "four-dimensional".
                else if (inputs.Shape.Length >= 6)
                {
                    throw new ArgumentException(
                        "For sequential training the input can have at most five dimensions " +
                        "[sequence, step, height, width, channels].");
                }

                _model.Context.InputShape = inputs.GetSubTensor([0, 0]).Shape;
            }

            else
            {

                if (inputs.Shape.Length == 1)
                {
                    inputs.Reshape([inputs.Shape[0], 1]);
                    targets.Reshape([inputs.Shape[0], 1]);
                }

                // [sample, …], where a sample has at most 3 dimensions → at most 4D. Previously this
                // was >= 4, so TrainLoop rejected an image dataset with channels [n, h, w, c] —
                // exactly what Conv takes — and the only workaround was flattening to [n, h·w·c].
                else if (inputs.Shape.Length >= 5)
                {
                    throw new ArgumentException(
                        "For non-sequential training the input can have at most four dimensions " +
                        "[sample, height, width, channels].");
                }

                _model.Context.InputShape = inputs.GetSubTensor([0]).Shape;
            }

            

            if (_model.Layers.Layers[0].InputShape[0] == 0)
            {
                _model.Layers.SetInputShape();
            }

            CheckLayersAreNotEmpty();

            ResetDerivedDatasets();
            SplitDatasets(inputs, targets);

            _listOfepoch = [];
            _listOfValidLoss = [];
            _listOfTrainLoss = [];

            // The batch size must be at least 1 — `start += batchSize` with zero would be an
            // infinite loop. (Previously zero silently meant no training.)
            if (miniBatchSize < 1)
            {
                throw new ArgumentException("The minibatch size must be at least 1.", nameof(miniBatchSize));
            }

            this._sizeOfMiniBatch = miniBatchSize;

            // `epochs` is the NUMBER of epochs of this run, not the model's target epoch.
            // Previously it was stored directly in _totalEpoch, so a second TrainLoop(5) on a model
            // with five completed epochs silently did nothing (the loop starts at _epoch and 5 < 5
            // is false). On a fresh model both give the same result, so nobody noticed.
            _totalEpoch = _epoch + epochs;

            _lowestLoss = double.MaxValue;
            _bestEpoch = 0;
            _bestSnapshot = null;
            _reportsWithoutImprovement = 0;
            _stopEarly = false;

            ResolveReportInterval(epochs);
        }

        // Translates the user's "how many reports" into "report every N epochs".
        // The clamp goes into the private _reportInterval; the public setting stays untouched.
        // Also resets the ETA state so a second training run does not use the first run's times.
        //
        // Computed from the LENGTH OF THIS RUN, not the absolute target epoch. Otherwise in a
        // continued run (100 completed epochs + 10 new) `ConsoleReportCount` would be applied
        // to 110, leaving a single report for the 10 new epochs.
        private void ResolveReportInterval(uint epochsInThisRun)
        {
            uint reports = ConsoleReportCount;

            if (reports < 1)
            {
                reports = 1;
            }
            else if (reports > epochsInThisRun)
            {
                reports = epochsInThisRun;
            }

            _reportInterval = (epochsInThisRun == 0 || reports == 0) ? 1 : Math.Max(1u, epochsInThisRun / reports);

            LastReportTime = DateTime.MinValue;
            LastReportEpoch = 0;
        }

        private void CheckLayersAreNotEmpty()
        {
            if (_model.Layers.Layers.Count == 0)
            {
                throw new InvalidOperationException("Only a model with layers can be trained. Add at least one layer with model.Layers.Add().");
            }

            if (_model.Layers.Layers[0].InputShape[0] <= 0)
            {
                throw new InvalidOperationException("The input layer must have at least one input. Set it with model.Layers.SetInputShape(new int[] { 1 }), or run a sample through the model first.");
            }
        }

        private void CheckTensorShapes(Tensor a, Tensor b)
        {
            if(a == null || b == null)
            {
                throw new ArgumentNullException(a == null ? "inputs" : "targets", "It is not possible to train with empty inputs (Tensor).");
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
                throw new ArgumentException("Inputs and targets must have the same first dimension (the number of samples).");
            }


        }

        // Resolves the train/valid/test ratios: the unset (null) ones share the remainder up to 1 equally.
        // All null → the canonical default 0.7/0.15/0.15.
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
                        throw new InvalidOperationException("A split ratio must be in the range (0, 1).");
                    setSum += r.Value;
                }
            }

            if (nullCount == 0 && Math.Abs(setSum - 1.0) > 1e-9)
                throw new InvalidOperationException("When all three ratios are set, they must add up to 1.");
            if (nullCount > 0 && setSum >= 1.0)
                throw new InvalidOperationException("The sum of the set ratios must be < 1, so the ones left unset (null) get a positive share.");

            double each = nullCount > 0 ? (1.0 - setSum) / nullCount : 0.0;

            return (TrainSplitRatio ?? each, ValidSplitRatio ?? each, TestSplitRatio ?? each);
        }

        // Every run must start from the same state: either what the user supplied explicitly, or
        // nothing. Derived (sliced) datasets from the previous run are dropped — otherwise
        // SplitDatasets would take a slice of the previous valid set for "the user's valid set"
        // and the split would shift with every run.
        private void ResetDerivedDatasets()
        {
            _trainDataInputs = null;
            _trainDataCurrentOutput = null;
            _validDataInputs = null;
            _validDataCurrentOutput = null;
            _testDataInputs = null;
            _testDataCurrentOutput = null;
        }

        // Splits the dataset into train/valid/test. Only TWO branches:
        //   1) the user supplied valid and/or test via SetDatasets() → used as given (ApplyExplicitDatasets)
        //   2) otherwise → everything is derived from the ratios
        //
        // Previously there were four, because valid and test could be set independently through
        // four setters and every combination of "what the user filled in" had its own path. The
        // combinations grew exponentially, and one of those branches silently trained on the
        // whole dataset including the test set.
        internal void SplitDatasets(Tensor inputs, Tensor targets)
        {
            // At least one of valid/test supplied → the missing one is carved (ApplyExplicitDatasets).
            // When neither was supplied, fall through to splitting the whole bundle by the ratios.
            if (_explicitTrain != null && (_explicitValid != null || _explicitTest != null))
            {
                ApplyExplicitDatasets();
                return;
            }

            int totalSize = inputs.Shape[0];

            // Each of the three parts must get at least one sample, otherwise Slice() throws
            // "Invalid slice range!". Without this EVERY dataset smaller than 7 samples crashed:
            // 15 % of 4 rounds down to zero, so the validation set came out empty.
            if (totalSize < 3)
            {
                throw new ArgumentException(
                    $"Splitting into train/valid/test needs at least 3 samples, got {totalSize}. " +
                    "For a smaller dataset, provide the sets yourself with SetDatasets(...).");
            }

            var (trainRatio, validRatio, _) = ResolveSplitRatios();

            // The clamp keeps each part at one sample or more while leaving room for the others.
            // For reasonably large datasets it has no effect and the split follows the ratios.
            int trainDataSize = Math.Clamp((int)(totalSize * trainRatio), 1, totalSize - 2);
            int validDataSize = Math.Clamp((int)(totalSize * validRatio), 1, totalSize - trainDataSize - 1);
            int testDataSize = totalSize - trainDataSize - validDataSize;   // the remainder → no sample is lost

            _trainDataInputs = inputs.Slice(0, trainDataSize);
            _trainDataCurrentOutput = targets.Slice(0, trainDataSize);

            _validDataInputs = inputs.Slice(trainDataSize, validDataSize);
            _validDataCurrentOutput = targets.Slice(trainDataSize, validDataSize);

            _testDataInputs = inputs.Slice(trainDataSize + validDataSize, testDataSize);
            _testDataCurrentOutput = targets.Slice(trainDataSize + validDataSize, testDataSize);
        }

        // ONE EPOCH = one full pass over the training set.
        //
        // Previously an "epoch" was `miniBatchSize` random draws WITH replacement followed by one
        // optimizer step, so some samples were never seen and `epochs` was really the number of
        // optimizer steps, not passes over the data.
        //
        // Now: every epoch shuffles the order of ALL training samples and walks through them in
        // batches of `miniBatchSize`, with one `UpdateParams()` per batch. The last batch may be
        // smaller — `Neuron.UpdateWeightsAndBias` divides by the actual number of accumulated
        // samples, so the gradient average is still correct.
        private void RunOneEpoch(Tensor inputs, Tensor targets)
        {
            int[] order = ShuffledIndices(inputs.Shape[0]);
            int batchSize = (int)_sizeOfMiniBatch;
            ReplicaPool? pool = CreateReplicaPool();

            if (pool != null)
            {
                // Tensor computes its flat data lazily on first read. Touch it here, still
                // sequentially, so several threads do not do it at once.
                _ = inputs.Data;
                _ = targets.Data;
            }

            for (int start = 0; start < order.Length; start += batchSize)
            {
                int end = Math.Min(start + batchSize, order.Length);

                RunBatch(pool, order, start, end, (train, index) => train.FitSample(inputs, targets, index));
            }
        }

        // Replicas for parallel training, or null = sequential (Workers == 1). Built at the start
        // of every epoch: rewiring the model between epochs (Add/Insert) would otherwise leave the
        // replicas with stale references to the weights. The cost is negligible next to an epoch.
        private ReplicaPool? CreateReplicaPool()
        {
            return _workers > 1 ? new ReplicaPool(_model, _workers) : null;
        }

        // One batch: samples order[start..end) and one optimizer step.
        // `fitOne(train, index)` computes one sample on the given Train — the original's Train
        // when sequential, a replica's Train when parallel. The same code in both paths so they
        // cannot diverge (every extra path to the same computation is a place to forget something).
        private void RunBatch(ReplicaPool? pool, int[] order, int start, int end, Action<Train, int> fitOne)
        {
            if (pool == null)
            {
                for (int k = start; k < end; k++)
                {
                    fitOne(this, order[k]);
                }
            }
            else
            {
                pool.RunBatch(order, start, end, fitOne);
            }

            UpdateParams();   // one optimizer step per batch
        }

        // Shuffled index order for one epoch. Fisher–Yates with the Context's generator, so it is
        // reproducible with a seed. INDICES are shuffled, not data — copying tensors every epoch
        // would be needlessly expensive for a large dataset.
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

        // One training sample (or a whole sequence) by index.
        private void FitSample(Tensor inputs, Tensor targets, int index)
        {
            if (!_model.Context.SequenceTrain)
            {
                Fit(inputs.GetSubTensor([index]), targets.GetSubTensor([index]));
                return;
            }

            _model.ResetSequence();
            for (int step = 0; step < inputs.Shape[1]; step++)
            {
                Fit(inputs.GetSubTensor([index, step]), targets.GetSubTensor([index, step]));
            }
            _model.ResetSequence();
        }

        // User-supplied datasets → working fields. A missing test set is carved from valid; a
        // missing valid set is carved from train (never from test — see the branches below).
        //
        // Carving always uses the ORIGINAL supplied set (_explicitValid / _explicitTrain), not the
        // working field. There used to be a bug here: test was cut from an already shrunk valid
        // set, so the offset fell out of range ("Invalid slice range!"). With a source that is
        // never overwritten this cannot happen.
        private void ApplyExplicitDatasets()
        {
            _trainDataInputs = _explicitTrain!.Inputs;
            _trainDataCurrentOutput = _explicitTrain.Targets;

            if (_explicitValid != null && _explicitTest != null)
            {
                SetValid(_explicitValid);
                SetTest(_explicitTest);
                return;
            }

            var (trainRatio, validRatio, testRatio) = ResolveSplitRatios();

            if (_explicitValid != null)
            {
                // TEST is missing. Carving it from valid only makes sense when the model is
                // actually tested after training — otherwise the user would lose part of the
                // validation set to data nobody uses. Hence TestAfterTraining controls it rather
                // than a separate flag: it is one decision ("do I want a final test?"), not two.
                //
                // Train stays whole either way: the user supplied it as their training set and
                // shrinking it behind their back would be surprising.
                if (!TestAfterTraining)
                {
                    SetValid(_explicitValid);
                    ClearTest();
                    return;
                }

                var (keptValid, carvedTest) = CarveInTwo(
                    _explicitValid, validRatio / (validRatio + testRatio), "validation");

                SetValid(keptValid);
                SetTest(carvedTest);
            }
            else
            {
                // VALID is missing → carve it from TRAIN, not from test. The test set is the final
                // unbiased estimate; carving validation data from it (used for model selection
                // during training) would spoil that estimate. In practice the validation set is
                // taken from train. Typical case: MNIST = train + test.
                var (keptTrain, carvedValid) = CarveInTwo(
                    _explicitTrain!, trainRatio / (trainRatio + validRatio), "training");

                _trainDataInputs = keptTrain.Inputs;
                _trainDataCurrentOutput = keptTrain.Targets;

                SetValid(carvedValid);
                SetTest(_explicitTest!);
            }
        }

        // Splits a set into two parts by the given share. Both must have at least one sample —
        // otherwise training would crash later on an empty dataset.
        private static (LabeledData first, LabeledData second) CarveInTwo(LabeledData source, double firstShare, string name)
        {
            int total = source.Count;

            if (total < 2)
            {
                throw new ArgumentException(
                    $"The missing set cannot be carved from the {name} set of {total} sample(s) — " +
                    "it needs at least 2. Provide both sets, or turn off TestAfterTraining.");
            }

            int firstSize = Math.Clamp((int)(total * firstShare), 1, total - 1);

            return (
                new LabeledData(source.Inputs.Slice(0, firstSize), source.Targets.Slice(0, firstSize)),
                new LabeledData(source.Inputs.Slice(firstSize, total - firstSize), source.Targets.Slice(firstSize, total - firstSize)));
        }

        private void SetValid(LabeledData data)
        {
            _validDataInputs = data.Inputs;
            _validDataCurrentOutput = data.Targets;
        }

        private void SetTest(LabeledData data)
        {
            _testDataInputs = data.Inputs;
            _testDataCurrentOutput = data.Targets;
        }

        private void ClearTest()
        {
            _testDataInputs = null;
            _testDataCurrentOutput = null;
        }

        internal void ShuffleTensor(Tensor tensorA, Tensor tensorB, out Tensor shuffledA, out Tensor shuffledB)
        {
            if (tensorA.Shape[0] != tensorB.Shape[0])
                throw new ArgumentException("Both tensors must have the same number of samples.");

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

        // Model output for each sample, in sample order. Forward only (no dropout), so with
        // Workers > 1 the samples are split among replicas and the result is still bit-for-bit
        // identical: a forward pass of one sample on the same weights gives the same result
        // whoever computes it. The caller does the summing (loss, score) sequentially.
        private double[][] PredictSamples(Tensor inputs)
        {
            int count = inputs.Shape[0];
            Func<NeuralNetwork, int, double[]> predict = (model, i) => model.GetResults(inputs.GetSubTensor([i])).Data;

            ReplicaPool? pool = CreateEvaluationPool(inputs);
            if (pool != null)
            {
                return pool.Map(count, predict);
            }

            var outputs = new double[count][];
            for (int i = 0; i < count; i++)
            {
                outputs[i] = predict(_model, i);
            }
            return outputs;
        }

        // The same for sequences: [sequence][step]. A whole sequence always goes through one
        // replica, and the hidden state is reset at its start — just as in training.
        private double[][][] PredictSequences(Tensor inputs)
        {
            int count = inputs.Shape[0];
            int steps = inputs.Shape[1];

            Func<NeuralNetwork, int, double[][]> predict = (model, i) =>
            {
                model.ResetSequence();
                var sequence = new double[steps][];
                for (int j = 0; j < steps; j++)
                {
                    sequence[j] = model.GetResults(inputs.GetSubTensor([i, j])).Data;
                }
                return sequence;
            };

            ReplicaPool? pool = CreateEvaluationPool(inputs);
            if (pool != null)
            {
                return pool.Map(count, predict);
            }

            var outputs = new double[count][][];
            for (int i = 0; i < count; i++)
            {
                outputs[i] = predict(_model, i);
            }
            return outputs;
        }

        // Replicas for evaluation, or null = sequential. Evaluation mode does not draw from the
        // model's generator (no dropout during prediction), so validation between epochs does
        // not change the data order of the next epoch.
        private ReplicaPool? CreateEvaluationPool(Tensor inputs)
        {
            if (_workers == 1 || inputs.Shape[0] < 2)
            {
                return null;
            }

            _ = inputs.Data;   // lazy flattening while still sequential — see RunOneEpoch
            return new ReplicaPool(_model, _workers, training: false);
        }

        // For classification (a whole-layer output activation such as Softmax) the argmax of the
        // output is compared with the argmax of the target; for regression the rounded values.
        // Shared so the accuracy during training is computed the same way as the final Evaluate.
        private bool IsPredictionCorrect(double[] output, double[] target)
        {
            if (_model.Layers.Layers[_model.Layers.Layers.Count() - 1].Activation.AppliesToLayer)
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
            // The non-null contract is enforced HERE, not in Loss: the valid data via a guard +
            // locals, the output from the return value of GetResults (always non-null), not from
            // the nullable Output property. So null cannot reach CalculateLoss.
            if (ValidInputs == null || ValidTargets == null)
            {
                throw new InvalidOperationException("The validation data is not set (SplitDatasets) — the validation loss cannot be computed.");
            }

            Tensor validInputs = ValidInputs;
            Tensor validOutputs = ValidTargets;

            // accuracy is computed in the SAME pass as the loss (no extra forward pass)
            int correct = 0;
            int total = 0;

            // All outputs first (in parallel when Workers > 1), then loss and accuracy sequentially
            // in sample order — so the loss sum is bit-for-bit the same as with Workers = 1, and
            // with it the best-model selection and early stopping.
            if (_model.Context.SequenceTrain)
            {
                double[][][] outputs = PredictSequences(validInputs);

                for (int i = 0; i < outputs.Length; i++)
                {
                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        double[] target = validOutputs.GetSubTensor([i, j]).Data;
                        lossFunc.CalculateLoss(outputs[i][j], target);
                        if (IsPredictionCorrect(outputs[i][j], target)) correct++;
                        total++;
                    }
                }
            }

            else
            {
                double[][] outputs = PredictSamples(validInputs);

                for (int i = 0; i < outputs.Length; i++)
                {
                    double[] target = validOutputs.GetSubTensor([i]).Data;
                    lossFunc.CalculateLoss(outputs[i], target);
                    if (IsPredictionCorrect(outputs[i], target)) correct++;
                    total++;
                }
            }

            double validAccuracy = total > 0 ? (double)correct / total * 100.0 : 0;

            ExtraControlFunc(loss);
            ConsoleController.ShowEpochInfo(_model, loss, validAccuracy: validAccuracy);
        }

        private void ExtraControlFunc(double loss)
        {
            Loss lossFunc = _model.Context.Loss;

            // Besides NaN we also check for infinity — diverging training typically passes through
            // ±∞ before reaching NaN, and in both cases the weights are unusable.
            double validLossNow = lossFunc.GetAverageLossPerIteration();

            if (!double.IsFinite(loss) || !double.IsFinite(validLossNow))
            {
                // chart only when the user wants it — otherwise every diverged run (e.g. dozens in an
                // AutoML runner training many candidates) would silently write loss.png to the working directory
                DrawLossChart();

                ConsoleController.ShowError("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values in the training parameters (for example the learning rate or other hyperparameters).", true);

                throw new TrainingDivergedException(_epoch, double.IsFinite(loss) ? validLossNow : loss);
            }

            if (_epoch >= (_reportInterval * LossChartSkipFirstReports))
            {
                _listOfepoch.Add((int)_epoch);
                _listOfTrainLoss.Add(loss);
                _listOfValidLoss.Add(lossFunc.GetAverageLossPerIteration());
            }

            double validLoss = lossFunc.GetAverageLossPerIteration();

            // improvement = valid loss dropped by more than minDelta. Only then is the best updated:
            // the best model is ALWAYS kept (in-memory snapshot) and restored at the end;
            // it is written to disk only with AutoSave (crash/NaN recovery, reuse next time).
            if (validLoss < _lowestLoss - EarlyStoppingMinDelta)
            {
                _lowestLoss = validLoss;
                _bestEpoch = _epoch;
                _bestSnapshot = _model.SaveAsJsonString();
                _reportsWithoutImprovement = 0;

                if (AutoSave)
                {
                    _model.SaveAsJson(AutoSaveFileName);
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

    }
    
}

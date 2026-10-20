using MDNN.Losses;
using MDNN.Optimizers;

namespace MDNN
{
    /// <summary>
    /// Per-model configuration: loss, optimizer, random source, regularization and training mode.
    /// </summary>
    /// <remarks>
    /// Each model has its own context, so two models in one process do not interfere with each other.
    /// </remarks>
    public class NetworkContext
    {
        /// <summary>
        /// Loss function used for training. Defaults to <see cref="MSE"/>.
        /// </summary>
        public Loss Loss { get; set; } = new MSE();
        private Optimizer _optimizer = new SGD(0.0001);

        // Each neuron (and each Conv layer) holds its own clone — the state is per parameter.
        // Changing it after the model is built propagates to all layers: they get a new
        // optimizer with empty state. Previously the change did not propagate at all — layers
        // kept training with the old one while the new one was saved to the file, so after
        // loading the model trained with a different optimizer. Assigning the same instance is a no-op.
        /// <summary>
        /// The model's optimizer. Setting a different instance after the model is built
        /// replaces the optimizer in all layers (with empty state).
        /// </summary>
        public Optimizer Optimizer
        {
            get => _optimizer;
            set
            {
                ArgumentNullException.ThrowIfNull(value);
                if (ReferenceEquals(value, _optimizer)) return;

                _optimizer = value;
                OptimizerChanged?.Invoke(value);
            }
        }

        // Who needs to know about an optimizer change — the model's LayerManager. A single slot,
        // not an event: it is overwritten when layers are reloaded, so a stale manager does not linger.
        internal Action<Optimizer>? OptimizerChanged { get; set; }

        // Random source for the WHOLE model: weight/kernel initialization, dataset shuffling and
        // sample selection in a minibatch. Previously these were three independent sources
        // (GeneralNeuralNetworkSettings.Random, Train._rnd, Random.Shared in ShuffleTensor), none of
        // them settable, so the same experiment could not be run twice.
        //
        // Without a seed it falls back to the shared global generator.
        //
        // NOTE: `Random` is not thread-safe. It is drawn from only on sequential paths; parallel
        // training gives each replica its own generator (see CreateReplica).
        private Random? _random;
        private int? _seed;
        private bool _randomAlreadyUsed;

        // Readable back so that a caller (e.g. an AutoML runner training many candidates) can record
        // how to reproduce a run without tracking the seed separately.
        /// <summary>
        /// Seed of the model's random generator, or <c>null</c> to use the shared global generator.
        /// Must be set before any weights are initialized.
        /// </summary>
        public int? Seed
        {
            get => _seed;
            set
            {
                ThrowIfRandomAlreadyUsed(nameof(Seed));
                _seed = value;
                _random = value == null ? null : new Random(value.Value);
            }
        }

        /// <summary>
        /// Random generator used for initialization, shuffling and sampling. Setting it clears <see cref="Seed"/>.
        /// </summary>
        public Random Random
        {
            get
            {
                // After the first draw it is too late to change anything — the weights are already drawn.
                _randomAlreadyUsed = true;
                return _random ??= GeneralNeuralNetworkSettings.Random;
            }
            set
            {
                ThrowIfRandomAlreadyUsed(nameof(Random));
                _random = value;
                _seed = null;   // custom generator → seed unknown, so Seed must not report a false one
            }
        }

        // Without this guard there was a silent trap: LayerManager.Add() draws the output layer's
        // weights IMMEDIATELY, so a seed set after Add() did not reach it. The model came out only
        // PARTLY reproducible (hidden layers yes, output no) with no warning — worse than an
        // obvious error, because reproducibility appeared to work.
        private void ThrowIfRandomAlreadyUsed(string propertyName)
        {
            if (_randomAlreadyUsed)
            {
                throw new InvalidOperationException(
                    $"{propertyName} cannot be set after the generator has already been drawn from " +
                    "(weights or kernels are already initialized). The model would only be partly " +
                    "reproducible. Pass the seed to the constructor instead: " +
                    "new NeuralNetwork(layer, optimizer, loss, seed: 42), or NeuralNetwork.LoadModel(path, seed: 42).");
            }
        }
        /// <summary>
        /// Whether training runs over sequences (recurrent layers keep state across samples).
        /// </summary>
        public bool SequenceTrain { get; set; } = false;
        /// <summary>
        /// Shape of a single input sample. <c>{ 0 }</c> means not known yet (inferred from the first input).
        /// </summary>
        public int[] InputShape { get; set; } = new int[] { 0 };

        // ---------------------------------------------------------------- regularization

        private double _l2 = 0.0;

        /// <summary>
        /// L2 regularization coefficient (weight decay). 0 = off (default).
        /// </summary>
        /// <remarks>
        /// Penalizes large weights to reduce overfitting: <c>λ/2 · Σw²</c> is added to the objective,
        /// i.e. <c>+ λ·w</c> in the gradient.
        ///
        /// Two caveats:
        /// 1. It applies ONLY to weights and kernels, NOT to biases. A bias only shifts the threshold;
        ///    shrinking it does not simplify the model, it just takes away its ability to shift.
        /// 2. The training loss reported by the library does NOT include the λ/2·Σw² term — same as
        ///    `weight_decay` in PyTorch. The reported loss is the pure prediction error.
        ///
        /// This is classic L2 (added to the gradient, so with Adam it also passes through the adaptive
        /// scaling). Decoupled weight decay as in AdamW is a different thing and is not implemented.
        /// </remarks>
        public double L2
        {
            get => _l2;
            set
            {
                if (double.IsNaN(value) || value < 0)
                    throw new ArgumentOutOfRangeException(nameof(L2), value,
                        "The L2 coefficient must be a non-negative number (0 = regularization off).");

                _l2 = value;
            }
        }

        private double _dropout = 0.0;

        /// <summary>
        /// Global dropout rate for the whole model. 0 = off (default).
        /// </summary>
        /// <remarks>
        /// Applies to every layer that does NOT set its own value, so a five-layer network needs one line
        /// instead of five. A layer overrides it with its own value
        /// (<c>new Dense(64, new ReLU(), dropout: 0.2)</c>), including zero, which disables dropout on that layer.
        ///
        /// It is NOT applied to the output layer. Dropping the network's outputs is not regularization, it
        /// corrupts the prediction — a mistake a global switch would otherwise introduce silently. To really
        /// do it, set it explicitly on the output layer.
        /// </remarks>
        public double Dropout
        {
            get => _dropout;
            set
            {
                if (double.IsNaN(value) || value < 0 || value >= 1)
                    throw new ArgumentOutOfRangeException(nameof(Dropout), value,
                        "Dropout must be in the range [0, 1). 0 = off; 1 would drop " +
                        "everything and the scaling factor 1/(1-p) would divide by zero.");

                _dropout = value;
            }
        }

        /// <summary>
        /// Whether the model is in training mode. Controls layers that behave differently in training
        /// and in prediction (currently dropout).
        /// </summary>
        /// <remarks>
        /// Set automatically: <c>Fit</c> turns it on for the duration of its pass, a standalone
        /// <c>GetResults</c> leaves it off. So a prediction for the same input always gives the same
        /// answer, even with dropout enabled.
        /// </remarks>
        public bool IsTraining { get; set; } = false;

        // Replica context for parallel training. Same training settings, but its own loss accumulator
        // and its OWN random generator — `Random` is not thread-safe and dropout draws from it for every
        // sample. The caller derives the seed from the original's generator, so a seeded model stays
        // reproducible.
        internal NetworkContext CreateReplica(int randomSeed)
        {
            return new NetworkContext
            {
                Loss = Loss.CreateReplica(),
                Optimizer = Optimizer,   // a replica does not update parameters; set only for completeness
                SequenceTrain = SequenceTrain,
                InputShape = InputShape,
                L2 = L2,
                Dropout = Dropout,
                Random = new Random(randomSeed),
            };
        }
    }
}

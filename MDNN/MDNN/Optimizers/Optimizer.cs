using MDNN.Serialization;


namespace MDNN.Optimizers
{
    /// <summary>
    /// Base class for optimizers that update parameters from their gradients.
    /// </summary>
    /// <remarks>
    /// Each neuron (and each Conv layer) holds its own instance, because the optimizer state is per parameter.
    /// </remarks>
    public abstract class Optimizer
    {
        /// <summary>
        /// Name used in the model file; built-in optimizers are reconstructed from it on load.
        /// </summary>
        public abstract string Name { get; }
        /// <summary>
        /// Hyperparameters saved to the model file (e.g. learning rate).
        /// </summary>
        public abstract double[] Hyperparameters { get; }
        /// <summary>
        /// Returns the updated value of parameter <paramref name="w"/> given its <paramref name="gradient"/>.
        /// </summary>
        /// <param name="w">Current parameter value.</param>
        /// <param name="gradient">Gradient of the loss with respect to the parameter.</param>
        /// <param name="paramIndex">Index of the parameter within this optimizer's state.</param>
        public abstract double Update(double w, double gradient, int paramIndex);

        // The layer calls it before a batch of Update calls, because only the layer knows how many
        // parameters it has — otherwise the state would grow one weight at a time (reallocating per
        // weight = quadratic on the first batch). Existing state is kept.
        /// <summary>
        /// Prepares state for parameters <c>0..parameterCount-1</c> ahead of time.
        /// The default does nothing (for stateless optimizers such as SGD).
        /// </summary>
        public virtual void EnsureCapacity(int parameterCount) { }

        // Abstract on purpose. A default implementation via MemberwiseClone would be a shallow copy,
        // so clones would share the state arrays (Adam's moments) — a silent bug guarded by the test
        // Two_models_with_adam_do_not_share_momentum. The optimizer itself knows how to build itself.
        /// <summary>
        /// Creates a new instance of the same optimizer: same hyperparameters, EMPTY state.
        /// </summary>
        public abstract Optimizer CreateFresh();

        // Without it a loaded model restarts from zero — Adam has zeroed moments and step 0, so the
        // first steps after loading are as large as at the start of training and resuming is not smooth.
        /// <summary>
        /// Exports the optimizer state for saving: named arrays (for Adam "m", "v", "t"), one value per parameter.
        /// </summary>
        /// <returns>
        /// The state, or <c>null</c> if the optimizer has no state (SGD) or cannot save it
        /// (a custom optimizer that does not override this) — it then loads with empty state.
        /// </returns>
        public virtual Dictionary<string, double[]>? ExportState() => null;

        /// <summary>
        /// Counterpart of <see cref="ExportState"/>. Receives a COPY of the arrays, so it may keep them.
        /// </summary>
        public virtual void ImportState(Dictionary<string, double[]> state) { }

        /// <summary>
        /// Returns a copy of the state array <paramref name="key"/>; throws if it is missing or its length
        /// differs from <paramref name="expectedLength"/> (the file is damaged).
        /// </summary>
        /// <param name="state">State dictionary passed to <see cref="ImportState"/>.</param>
        /// <param name="key">Name of the array.</param>
        /// <param name="expectedLength">Required length, or <c>null</c> to accept any.</param>
        protected static double[] RequireState(Dictionary<string, double[]> state, string key, int? expectedLength)
        {
            if (!state.TryGetValue(key, out double[]? values))
            {
                throw new ArgumentException($"The optimizer state is missing the array '{key}'.");
            }

            if (expectedLength != null && values.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"The array '{key}' in the optimizer state has {values.Length} values, the others {expectedLength}.");
            }

            return (double[])values.Clone();
        }

        internal static Optimizer FromExport(ExportOptimizer optimizer)
        {
            return GetOptimizer(optimizer.Name, optimizer.Hyperparameters);
        }

        // Can loading construct it from the name in the file alone?
        internal static bool IsBuiltIn(string name) => name is "SGD" or "Adam" or "Momentum";

        // Optimizer for a model being loaded. `own` = the optimizer the caller has at hand:
        // passed to LoadModel, or the one the model already has (snapshot restore in TrainLoop).
        //   - one passed to LoadModel always wins, but the name must match (otherwise the state from
        //     the file would be loaded into an unrelated optimizer);
        //   - otherwise a built-in one is constructed from the name and hyperparameters in the file;
        //   - a custom one we cannot construct is taken from `own` with the same name.
        internal static Optimizer ResolveForLoad(ExportOptimizer saved, Optimizer? own, bool ownWins)
        {
            if (own != null && own.Name == saved.Name && (ownWins || !IsBuiltIn(saved.Name)))
            {
                return own;
            }

            if (own != null && ownWins)
            {
                throw new ArgumentException(
                    $"The model was saved with optimizer '{saved.Name}', but '{own.Name}' was passed in. " +
                    "The saved optimizer state would be loaded into a different optimizer.");
            }

            if (!IsBuiltIn(saved.Name))
            {
                throw new ArgumentException(
                    $"Optimizer '{saved.Name}' is not built in, so loading cannot create it by name. " +
                    "Pass an instance: NeuralNetwork.LoadModel(path, optimizer: new …).");
            }

            return FromExport(saved);
        }

        // Only for loading from a file: there a name is available, not an instance.
        private static Optimizer GetOptimizer(string name, double[] hyperparameters)
        {
            switch (name)
            {
                case "SGD":
                    return new SGD(hyperparameters[0]);
                case "Adam":
                    return new Adam(hyperparameters[0], hyperparameters[1], hyperparameters[2]);
                case "Momentum":
                    return new Momentum(hyperparameters[0], hyperparameters[1]);
                default: throw new ArgumentException($"this optimizer ({name}) does not exist");
            }
        }
    }
}

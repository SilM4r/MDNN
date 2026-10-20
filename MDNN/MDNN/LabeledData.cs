namespace MDNN
{
    // Previously valid/test were supplied through four independent setters (`ValidInputs`,
    // `ValidTargets`, …). Setting the inputs and forgetting the targets was silently allowed:
    // training started and crashed only at the first valid report, far from the actual mistake.
    // Binding inputs and targets into one object makes that state impossible.
    //
    // Naming note: NOT `Dataset`/`DataSet` — `Train.cs` has `using System.Data`, where `DataSet`
    // is something else entirely.
    /// <summary>
    /// A pair of inputs and their expected outputs (targets). Exists so that one cannot be
    /// passed without the other.
    /// </summary>
    public sealed class LabeledData
    {
        /// <summary>Input samples; the first dimension is the sample (or sequence) index.</summary>
        public Tensor Inputs { get; }
        /// <summary>Expected outputs, one per input sample.</summary>
        public Tensor Targets { get; }

        /// <summary>Number of samples (number of sequences for sequential training).</summary>
        public int Count => Inputs.Shape[0];

        /// <summary>Creates a labeled dataset; inputs and targets must have the same number of samples.</summary>
        public LabeledData(Tensor inputs, Tensor targets)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(targets);

            if (inputs.Shape.Length == 0 || targets.Shape.Length == 0)
            {
                throw new ArgumentException("Both inputs and targets must have at least one dimension.");
            }

            if (inputs.Shape[0] != targets.Shape[0])
            {
                throw new ArgumentException(
                    $"The number of samples does not match: inputs {inputs.Shape[0]}, targets {targets.Shape[0]}.");
            }

            Inputs = inputs;
            Targets = targets;
        }

        /// <summary>
        /// Convenience overload for <c>double[][]</c>, <c>double[,]</c> and similar arrays, so they
        /// do not have to be converted to <see cref="Tensor"/> by hand.
        /// </summary>
        public LabeledData(Array inputs, Array targets)
            : this(new Tensor(inputs), new Tensor(targets))
        {
        }
    }
}

using System.Collections;
using MDNN.Layers;

namespace MDNN
{
    /// <summary>
    /// The hidden layers of a model — every layer except the output layer, in order from input to output.
    /// </summary>
    /// <remarks>
    /// Positions count ONLY the hidden layers: <c>Insert(Count, layer)</c> puts a layer just before the
    /// output layer, never after it, and <c>RemoveAt</c> can never remove the output layer. The output
    /// layer is changed only by <see cref="LayerManager.ReplaceOutput"/>.
    ///
    /// Previously <c>Insert</c>/<c>RemoveAt</c> indexed the whole chain, so they could also change the
    /// output layer as a side effect: <c>Insert(Count, x)</c> made <c>x</c> the new output layer and
    /// <c>RemoveAt(last)</c> turned the last hidden layer into it.
    /// </remarks>
    public sealed class HiddenLayers : IReadOnlyList<Layer>
    {
        private readonly LayerManager manager;

        internal HiddenLayers(LayerManager manager)
        {
            this.manager = manager;
        }

        /// <summary>Number of hidden layers.</summary>
        public int Count => manager.Layers.Count - 1;

        /// <summary>
        /// The hidden layer at <paramref name="index"/> (0 = first layer of the model). Setting it replaces
        /// that one hidden layer; the output layer cannot be reached through it.
        /// </summary>
        /// <remarks>
        /// <c>Hidden[1] = new Dense(32, new Tanh())</c> is the same as <c>RemoveAt(1)</c> followed by
        /// <c>Insert(1, …)</c>, in one step: the old layer is detached from the model, the new one is built
        /// on the next pass, and the other layers keep their weights as long as their input does not change.
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">The index is outside 0 … <see cref="Count"/> − 1.</exception>
        /// <exception cref="ArgumentException">(set) The layer is already elsewhere in this model, or in another model.</exception>
        public Layer this[int index]
        {
            get
            {
                RequireIndex(index);
                return manager.Layers[index];
            }
            set
            {
                ArgumentNullException.ThrowIfNull(value);
                RequireIndex(index);
                manager.SetHidden(index, value);
            }
        }

        private void RequireIndex(int index)
        {
            if (index < 0 || index >= Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index), index,
                    Count == 0
                        ? "The model has no hidden layers (the output layer is not one of them; replace it with ReplaceOutput)."
                        : $"The model has {Count} hidden layer(s); the index must be in the range 0 ≤ index < {Count} " +
                          "(the output layer is not one of them; replace it with ReplaceOutput).");
            }
        }

        /// <inheritdoc/>
        public IEnumerator<Layer> GetEnumerator()
        {
            IReadOnlyList<Layer> all = manager.Layers;
            for (int i = 0; i < all.Count - 1; i++)
            {
                yield return all[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Adds a hidden layer after the existing ones, just before the output layer.
        /// </summary>
        /// <exception cref="ArgumentException">The layer instance is already in this or another model.</exception>
        public void Add(Layer layer) => Insert(Count, layer);

        /// <summary>
        /// Inserts a hidden layer at <paramref name="position"/>: 0 = first layer of the model,
        /// <see cref="Count"/> = just before the output layer.
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException">The position is outside 0 … <see cref="Count"/>.</exception>
        /// <exception cref="ArgumentException">The layer instance is already in this or another model.</exception>
        public void Insert(int position, Layer layer)
        {
            ArgumentNullException.ThrowIfNull(layer);

            if (position < 0 || position > Count)
            {
                throw new ArgumentOutOfRangeException(nameof(position), position,
                    $"A hidden layer can be inserted at 0 … {Count} (the output layer is always last; " +
                    "replace it with ReplaceOutput).");
            }

            manager.InsertHidden(position, layer);
        }

        /// <summary>
        /// Removes the hidden layer at <paramref name="position"/>. The output layer cannot be removed;
        /// replace it with <see cref="LayerManager.ReplaceOutput"/>.
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException">The position is outside 0 … <see cref="Count"/> − 1.</exception>
        public void RemoveAt(int position)
        {
            if (position < 0 || position >= Count)
            {
                throw new ArgumentOutOfRangeException(nameof(position), position,
                    Count == 0
                        ? "The model has no hidden layers to remove (the output layer cannot be removed; replace it with ReplaceOutput)."
                        : $"A hidden layer can be removed at 0 … {Count - 1} (the output layer cannot be removed; replace it with ReplaceOutput).");
            }

            manager.RemoveHidden(position);
        }

        /// <summary>Removes all hidden layers; the output layer stays.</summary>
        public void Clear() => Replace(Array.Empty<Layer>());

        /// <summary>
        /// Replaces all hidden layers with <paramref name="layers"/> (in order from input to output); the
        /// output layer stays. An empty list removes all hidden layers.
        /// </summary>
        /// <exception cref="ArgumentException">The list contains null, or a layer instance twice or from another model.</exception>
        public void Replace(IEnumerable<Layer> layers) => manager.ReplaceHidden(layers);
    }
}

namespace MDNN.Layers
{
    /// <summary>
    /// How an input shape becomes <c>[height, width, channels]</c> — shared by the layers
    /// that work spatially (<see cref="MDNN.Layers.Conv"/>, <see cref="MDNN.Layers.MaxPool"/>).
    /// </summary>
    /// <remarks>
    /// Why it has its own place: this logic used to be written TWICE in each of those layers —
    /// once when wiring the chain and once in <c>FeedForward</c> (every pass) —
    /// and the two copies disagreed. Wiring handled rank 1 and 3, forward also rank 2.
    /// A model with Conv or MaxPool as the first layer and a <c>double[h,w]</c> input therefore
    /// could not be built at all: it hit <c>throw new Exception()</c> with no message.
    ///
    /// With normalization in one place the two paths cannot diverge. Duplicated code paths
    /// drifting apart is a pattern that has bitten this library three more times (async backward
    /// without dropout, FitAsync without IsTraining, Train.FeedForward without IsTraining).
    /// </remarks>
    internal static class SpatialShape
    {
        /// <summary>Converts a shape to <c>[h,w,c]</c>. ALWAYS returns a new array — a layer must not
        /// hold a reference to its neighbour's shape, which may be recomputed underneath it.</summary>
        public static int[] Normalize(int[] shape, string layerName)
        {
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException($"{layerName}: the input shape is empty.");
            }

            switch (shape.Length)
            {
                case 3:
                    return new int[] { shape[0], shape[1], shape[2] };

                case 2:
                    // 2D image without a channel dimension → one channel
                    return new int[] { shape[0], shape[1], 1 };

                case 1:
                    (int rows, int cols) = Factorize(shape[0], layerName);
                    return new int[] { rows, cols, 1 };

                default:
                    throw new ArgumentException(
                        $"{layerName} accepts input as [height,width,channels], [height,width] or a flat " +
                        $"vector, but got a {shape.Length}-dimensional shape [{Describe(shape)}].");
            }
        }

        /// <summary>Factorizes a flat vector length into 2D.</summary>
        /// <remarks>
        /// ⚠️ This is a GUESS, not information: 784 gives 28×28, but 600 gives 25×24 even though
        /// the user may have meant 20×30, and RGB 28×28×3 (2352) is factorized as 49×48.
        /// The behaviour is DELIBERATELY kept unchanged — the proper fix is an API to declare
        /// the input shape explicitly, not smarter guessing.
        /// </remarks>
        public static (int rows, int cols) Factorize(int count, string layerName)
        {
            if (count <= 0)
            {
                throw new ArgumentException(
                    $"{layerName}: a flat input must have at least one element, got {count}.");
            }

            int rows = (int)Math.Ceiling(Math.Sqrt(count));

            while (count % rows != 0)
            {
                rows++;
            }

            return (rows, count / rows);
        }

        /// <summary>Checks that the incoming input matches the shape the layer is BUILT for,
        /// and returns it normalized.</summary>
        /// <remarks>
        /// A layer is built once — on the first pass, left to right through the whole chain — and
        /// its shapes are fixed from then on. The layer preallocates its input buffer from them, but the copy
        /// loop in <c>FeedForward</c> used to follow the shape of the INCOMING tensor. A smaller input
        /// overwrote only its corner and the rest of the buffer kept the previous sample, silently
        /// returning stale numbers; a larger input crashed with <c>IndexOutOfRange</c> deep in the
        /// loop, without mentioning shapes.
        /// </remarks>
        public static int[] RequireBuiltShape(int[] incoming, int[] built, string layerName)
        {
            if (built.Length != 3 || built[0] <= 0)
            {
                throw new InvalidOperationException(
                    $"{layerName} is not built yet. Add the layer to a model " +
                    "and let the first pass build it, or call Build(inputShape) yourself.");
            }

            int[] normalized = Normalize(incoming, layerName);

            if (!SameAs(normalized, built))
            {
                string from = SameAs(normalized, incoming)
                    ? ""
                    : $" (from shape [{Describe(incoming)}])";

                throw new ArgumentException(
                    $"{layerName} is built for input {Describe(built)}, but got " +
                    $"{Describe(normalized)}{from}. {MDNN.Layers.Layer.ShapeChangeHint}");
            }

            return normalized;
        }

        public static bool SameAs(int[] a, int[] b)
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Formats a shape for an error message: <c>4×4×1</c>.</summary>
        public static string Describe(int[] shape) => string.Join("×", shape);
    }
}

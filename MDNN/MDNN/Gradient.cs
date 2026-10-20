using MDNN.Activations;
using MDNN.Layers;



namespace MDNN
{
    /// <summary>
    /// Runs the backward pass of a model (loss gradient through all layers).
    /// </summary>
    internal static class Gradient
    {

        // ---------------------------------------------------------------- convention B
        //
        // Previously backpropagation had two phases: first the whole array of gradients was
        // computed, then applied in a second pass. Now each layer accumulates its gradients
        // directly and returns ∂L/∂input for the layer below.
        /// <summary>
        /// Backpropagates the loss for <paramref name="targets"/> through all layers in a single pass.
        /// Nothing is returned; the result is the gradients accumulated in the layers.
        /// </summary>
        /// <param name="targets">Expected output for the last forward pass.</param>
        /// <param name="model">The model whose layers receive the gradients.</param>
        /// <param name="outputs">Output to use instead of the last layer's <c>Output</c>; optional.</param>
        public static void RunBackward(Tensor targets, NeuralNetwork model, Tensor? outputs = null)
        {
            IReadOnlyList<Layer> layers = model.Layers.Layers;
            Layer lastLayer = layers[^1];

            if (model.Context.Loss.RequiresSoftmax && !lastLayer.Activation.AppliesToLayer)
            {
                throw new InvalidOperationException(
                    "CrossEntropy requires a Softmax output layer.");
            }

            if ((lastLayer.Output == null && outputs == null)
                || lastLayer.RawOutput == null)
            {
                throw new ArgumentException(
                    "FeedForward must run before BackPropagation. Use Train.Fit() or one of the training loops, which run both.");
            }

            Tensor outputTensor = outputs ?? lastLayer.Output!;

            if (targets.Data.Length != outputTensor.Data.Length)
            {
                throw new ArgumentException(
                    $"The output layer has {outputTensor.Data.Length} values but the target has " +
                    $"{targets.Data.Length}. The number of targets must match the number of output neurons.");
            }

            // Fused softmax + CrossEntropy: the loss returns ∂L/∂z = s − t directly. Here softmax
            // logically belongs to the loss function, not to the layer — hence the preActivation
            // flag rather than a second entry point into Backward.
            bool fusedSoftmaxCE = model.Context.Loss.RequiresSoftmax
                                  && lastLayer.Activation.AppliesToLayer;

            // ∂L/∂output WITHOUT the activation derivative. Each layer applies act' its own way
            // (Conv does, RNN does not — its RTRL sensitivities already include it).
            double[] dLossDOutput = new double[outputTensor.Data.Length];
            for (int i = 0; i < dLossDOutput.Length; i++)
            {
                dLossDOutput[i] = model.Context.Loss.DerivativeOfLossFunction(
                    outputTensor.Data[i], targets.Data[i]);
            }

            model.Context.Loss.CalculateLoss(outputTensor.Data, targets.Data);

            Tensor grad = new Tensor(dLossDOutput);
            grad = lastLayer.Backward(grad, preActivation: fusedSoftmaxCE);

            for (int i = layers.Count - 2; i >= 0; i--)
            {
                // Nobody consumes the FIRST layer's input gradient — there is nothing below it.
                grad = layers[i].Backward(grad, computeInputGradient: i > 0);
            }
        }

    }
}

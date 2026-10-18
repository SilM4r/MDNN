using My_DNN.Activation_functions;
using My_DNN.Layers.classes;



namespace My_DNN
{
    static public class Gradient
    {

        // ---------------------------------------------------------------- konvence B
        //
        // Backward jedním průchodem. Proti staré cestě mizí dvoufázovost: dřív se nejdřív
        // spočítalo celé pole gradientů (GetGradients) a teprve pak se ve druhém průchodu
        // aplikovalo (Train.Propagation). Teď každá vrstva rovnou naakumuluje své gradienty
        // a vrátí ∂L/∂input pro tu pod sebou.
        //
        // Nic se nevrací — výsledkem jsou naakumulované gradienty ve vrstvách.
        public static void RunBackward(Tensor target_values, MDNN model, Tensor? output_values_from_model = null)
        {
            List<Layer> layers = model.Layers.Layers;
            Layer lastLayer = layers[^1];

            if (model.Context.Loss.RequiresSoftmax && !lastLayer.Activation_Func.Apply_to_layer)
            {
                throw new InvalidOperationException(
                    "CrossEntropy requires a Softmax output layer.");
            }

            if ((lastLayer.Layer_output == null && output_values_from_model == null)
                || lastLayer.Layer_raw_output == null)
            {
                throw new ArgumentException(
                    "Feedforward must be run before backpropagation. it is recommended to use trainLoop() in the Train class or fit()");
            }

            Tensor outputTensor = output_values_from_model ?? lastLayer.Layer_output!;

            if (target_values.Data.Length != outputTensor.Data.Length)
            {
                throw new ArgumentException(
                    $"The output layer has {outputTensor.Data.Length} values but the target has " +
                    $"{target_values.Data.Length}. The number of targets must match the number of output neurons.");
            }

            // Fúze softmax + CrossEntropy: loss vrací rovnou ∂L/∂z = s − t. Softmax tam
            // logicky patří k loss funkci, ne k vrstvě — proto preActivation, ne druhý
            // vstupní bod do backwardu.
            bool fusedSoftmaxCE = model.Context.Loss.RequiresSoftmax
                                  && lastLayer.Activation_Func.Apply_to_layer;

            // ∂L/∂output BEZ derivace aktivace. Každá vrstva si act' dodá po svém
            // (Conv ano, RNN ne — RTRL citlivosti ji už obsahují).
            double[] dLossDOutput = new double[outputTensor.Data.Length];
            for (int i = 0; i < dLossDOutput.Length; i++)
            {
                dLossDOutput[i] = model.Context.Loss.DerivativeOfLossFunction(
                    outputTensor.Data[i], target_values.Data[i]);
            }

            model.Context.Loss.CalculateLoss(outputTensor.Data, target_values.Data);

            Tensor grad = new Tensor(dLossDOutput);
            grad = lastLayer.Backward(grad, preActivation: fusedSoftmaxCE);

            for (int i = layers.Count - 2; i >= 0; i--)
            {
                // Vstupní gradient PRVNÍ vrstvy nikdo nepřevezme — pod ní už nic není.
                grad = layers[i].Backward(grad, computeInputGradient: i > 0);
            }
        }

    }
}

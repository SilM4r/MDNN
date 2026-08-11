using My_DNN.Activation_functions;
using My_DNN.Layers.classes;



namespace My_DNN
{
    static public class Gradient
    {
        // ∂L/∂output → ∂L/∂raw_output pro VÝSTUPNÍ vrstvu. Sdílené sync i async cestou,
        // ať se nemůžou rozejít.
        private static double[] ApplyOutputActivationBackward(Layer lastLayer, double[] dLossDOutput, bool fusedSoftmaxCE)
        {
            // Fúze softmax+CE: loss vrací rovnou dL/dz = s − t, aktivace se přeskakuje
            // (jinak by se softmax derivace aplikovala dvakrát).
            if (fusedSoftmaxCE)
            {
                return dLossDOutput;
            }

            double[] raw = lastLayer.Layer_raw_output!.Data;

            if (lastLayer.Activation_Func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = lastLayer.Activation_Func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                return layerActivationFunc.BackwardForLayer(raw, dLossDOutput);
            }

            double[] de = new double[dLossDOutput.Length];
            for (int i = 0; i < de.Length; i++)
            {
                de[i] = dLossDOutput[i] * lastLayer.Activation_Func.Derivative(raw[i]);
            }
            return de;
        }

        public static Tensor[] GetGradients(Tensor target_values, MDNN model, Tensor? output_values_from_model = null)
        {
            List<Layer> layers = model.Layers.Layers;
            List<Tensor?> e = new List<Tensor?>();
            Tensor outputTensor;

            int lastElement = layers.Count() - 1;
            Layer lastLayer = layers[lastElement];
            
            if (model.Context.Loss.RequiresSoftmax && !lastLayer.Activation_Func.Apply_to_layer)
            {
                throw new InvalidOperationException(
                    "CrossEntropy vyžaduje softmax výstupní vrstvu.");
            }

            if ((lastLayer.Layer_output == null && output_values_from_model == null) || lastLayer.Layer_raw_output == null)
            {
                throw new ArgumentException("Feedforward must be run before backpropagation. it is recommended to use trainLoop() in the Train class or fit()");
            }


            if (output_values_from_model == null)
            {
                outputTensor = lastLayer.Layer_output;
            }
            else
            {
                outputTensor = output_values_from_model;
            }

            // Fúze softmax + CrossEntropy: loss vrací rovnou deltu dL/dz = s - t,
            // takže se NEbuduje Jacobian a NEnásobí se softmax derivací (jinak dvojitá aplikace).
            bool fusedSoftmaxCE = model.Context.Loss.RequiresSoftmax
                                  && lastLayer.Activation_Func.Apply_to_layer;

            // Nesedící počet cílů je běžná uživatelská chyba (výstupní vrstva má N neuronů,
            // ale cíl je skalár). Bez téhle kontroly z toho padal IndexOutOfRangeException
            // z útrob knihovny, ze kterého se příčina nepozná.
            if (target_values.Data.Length != outputTensor.Data.Length)
            {
                throw new ArgumentException(
                    $"Výstupní vrstva má {outputTensor.Data.Length} hodnot, ale cíl jich má " +
                    $"{target_values.Data.Length}. Počet cílů musí odpovídat počtu neuronů výstupní vrstvy.");
            }

            // ∂L/∂output BEZ derivace aktivace — vstup pro SeedOutputGradient i pro
            // BackwardForLayer (vrstvy/aktivace si act' dodají podle své konvence).
            double[] dLossDOutput = new double[outputTensor.Data.Length];
            for (int i = 0; i < outputTensor.Data.Length; i++)
            {
                dLossDOutput[i] = model.Context.Loss.DerivativeOfLossFunction(outputTensor.Data[i], target_values.Data[i]);
            }

            double[] de = ApplyOutputActivationBackward(lastLayer, dLossDOutput, fusedSoftmaxCE);

            // Bez tohohle se výstupní Conv/RNN tiše neučí (jejich BackPropagation čte vnitřní
            // stav, ne argument). Dense/MaxPool mají no-op default.
            lastLayer.SeedOutputGradient(dLossDOutput);

            e.Add(new Tensor(de));

            model.Context.Loss.CalculateLoss(outputTensor.Data, target_values.Data);

            for (int i = layers.Count() - 2; i > -1; i--)
            {
                e.Insert(0, null);

                // Žádná příprava aktivace tady — vrstvová aktivace (softmax) se řeší uvnitř
                // CalculateLayerGradients přes BackwardForLayer, kde je k dispozici celý vektor.

                e[0] = layers[i].CalculateLayerGradients(e[1], layers[i + 1]);

            }
            return (e as List<Tensor>).ToArray();
        }
        public static async Task<Tensor[]> GetGradientsAsync(Tensor target_values, MDNN model, Tensor? output_values_from_model = null)
        {
            List<Layer> layers = model.Layers.Layers;
            List<Tensor?> e = new List<Tensor?>();
            Tensor outputTensor;

            int lastElement = layers.Count() - 1;
            Layer lastLayer = layers[lastElement];
            
            if (model.Context.Loss.RequiresSoftmax && !lastLayer.Activation_Func.Apply_to_layer)
            {
                throw new InvalidOperationException(
                    "CrossEntropy vyžaduje softmax výstupní vrstvu.");
            }


            if ((lastLayer.Layer_output == null && output_values_from_model == null) || lastLayer.Layer_raw_output == null)
            {
                throw new ArgumentException("Feedforward must be run before backpropagation. it is recommended to use trainLoop() in the Train class or fit()");
            }


            if (output_values_from_model == null)
            {
                outputTensor = lastLayer.Layer_output;
            }
            else
            {
                outputTensor = output_values_from_model;
            }

            // Fúze softmax + CrossEntropy (viz synchronní verze). Ve fúzní cestě se
            // nevolá Derivative() → odpadá i race na stavovém čítači 'a' v Softmaxu.
            bool fusedSoftmaxCE = model.Context.Loss.RequiresSoftmax
                                  && lastLayer.Activation_Func.Apply_to_layer;

            if (target_values.Data.Length != outputTensor.Data.Length)
            {
                throw new ArgumentException(
                    $"Výstupní vrstva má {outputTensor.Data.Length} hodnot, ale cíl jich má " +
                    $"{target_values.Data.Length}. Počet cílů musí odpovídat počtu neuronů výstupní vrstvy.");
            }

            double[] dLossDOutput = new double[outputTensor.Data.Length];

            Task[] tasks = new Task[outputTensor.Data.Length];
            for (int i = 0; i < outputTensor.Data.Length; i++)
            {
                int index = i;
                tasks[index] = Task.Run(() =>
                {
                    dLossDOutput[index] = model.Context.Loss.DerivativeOfLossFunction(outputTensor.Data[index], target_values.Data[index]);
                });
            }

            await Task.WhenAll(tasks);

            double[] de = ApplyOutputActivationBackward(lastLayer, dLossDOutput, fusedSoftmaxCE);

            // stejně jako v synchronní verzi — jinak se výstupní Conv/RNN tiše neučí
            lastLayer.SeedOutputGradient(dLossDOutput);

            e.Add(new Tensor(de));

            model.Context.Loss.CalculateLoss(outputTensor.Data, target_values.Data);

            for (int i = layers.Count() - 2; i > -1; i--)
            {
                e.Insert(0, null);

                // Žádná příprava aktivace tady — vrstvová aktivace (softmax) se řeší uvnitř
                // CalculateLayerGradients přes BackwardForLayer, kde je k dispozici celý vektor.

                e[0] = await layers[i].CalculateLayerGradientsAsync(e[1], layers[i + 1]);

            }
            return (e as List<Tensor>).ToArray();
        }
    }
}

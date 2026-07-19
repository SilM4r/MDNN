using My_DNN.Activation_functions;
using My_DNN.Layers.classes;



namespace My_DNN
{
    static public class Gradient
    {
        public static Tensor[] GetGradients(Tensor target_values, MDNN model, Tensor? output_values_from_model = null)
        {
            List<Layer> layers = model.Layers.Layers;
            List<Tensor?> e = new List<Tensor?>();
            Tensor outputTensor;

            int lastElement = layers.Count() - 1;
            Layer lastLayer = layers[lastElement];
            
            if (GeneralNeuralNetworkSettings.loss_func.RequiresSoftmax && !lastLayer.Activation_Func.Apply_to_layer)
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
            bool fusedSoftmaxCE = GeneralNeuralNetworkSettings.loss_func.RequiresSoftmax
                                  && lastLayer.Activation_Func.Apply_to_layer;

            if (lastLayer.Activation_Func.Apply_to_layer && !fusedSoftmaxCE)
            {
                LayerActivationFunc? layerActivationFunc = lastLayer.Activation_Func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                layerActivationFunc.DerivativeForLayer(lastLayer.Layer_raw_output.Data);
            }

            double[] de = new double[outputTensor.Data.Length];

            for (int i = 0; i < outputTensor.Data.Length; i++)
            {
                if (fusedSoftmaxCE)
                {
                    de[i] = GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(outputTensor.Data[i], target_values.Data[i]);
                }
                else
                {
                    double Output = lastLayer.Layer_raw_output.Data[i];

                    double derivateActivationOutputLayer = lastLayer.Activation_Func.Derivative(Output);

                    de[i] = (GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(outputTensor.Data[i], target_values.Data[i]) * derivateActivationOutputLayer);
                }
            }

            e.Add(new Tensor(de));

            GeneralNeuralNetworkSettings.loss_func.CalculateLoss(outputTensor.Data, target_values.Data);

            for (int i = layers.Count() - 2; i > -1; i--)
            {
                e.Insert(0, null);

                if (layers[i].Activation_Func.Apply_to_layer)
                {
                    LayerActivationFunc? layerActivationFunc = layers[i].Activation_Func as LayerActivationFunc;

                    if (layerActivationFunc == null)
                    {
                        throw new ArgumentException("Bad activation func");
                    }

                    layerActivationFunc.DerivativeForLayer(layers[i].Layer_raw_output.Data);
                }

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
            
            if (GeneralNeuralNetworkSettings.loss_func.RequiresSoftmax && !lastLayer.Activation_Func.Apply_to_layer)
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
            bool fusedSoftmaxCE = GeneralNeuralNetworkSettings.loss_func.RequiresSoftmax
                                  && lastLayer.Activation_Func.Apply_to_layer;

            if (lastLayer.Activation_Func.Apply_to_layer && !fusedSoftmaxCE)
            {
                LayerActivationFunc? layerActivationFunc = lastLayer.Activation_Func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                layerActivationFunc.DerivativeForLayer(lastLayer.Layer_raw_output.Data);
            }

            double[] de = new double[outputTensor.Data.Length];

            Task[] tasks = new Task[outputTensor.Data.Length];
            for (int i = 0; i < outputTensor.Data.Length; i++)
            {
                int index = i;
                tasks[index] = Task.Run(() =>
                {
                    if (fusedSoftmaxCE)
                    {
                        de[index] = GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(outputTensor.Data[index], target_values.Data[index]);
                    }
                    else
                    {
                        double Output = lastLayer.Layer_raw_output.Data[index];

                        double derivateActivationOutputLayer = lastLayer.Activation_Func.Derivative(Output);

                        de[index] = (GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(outputTensor.Data[index], target_values.Data[index]) * derivateActivationOutputLayer);
                    }
                });
            }

            await Task.WhenAll(tasks);

            e.Add(new Tensor(de));

            GeneralNeuralNetworkSettings.loss_func.CalculateLoss(outputTensor.Data, target_values.Data);

            for (int i = layers.Count() - 2; i > -1; i--)
            {
                e.Insert(0, null);

                if (layers[i].Activation_Func.Apply_to_layer)
                {
                    LayerActivationFunc? layerActivationFunc = layers[i].Activation_Func as LayerActivationFunc;

                    if (layerActivationFunc == null)
                    {
                        throw new ArgumentException("Bad activation func");
                    }

                    layerActivationFunc.DerivativeForLayer(layers[i].Layer_raw_output.Data);
                }

                e[0] = await layers[i].CalculateLayerGradientsAsync(e[1], layers[i + 1]);

            }
            return (e as List<Tensor>).ToArray();
        }
    }
}

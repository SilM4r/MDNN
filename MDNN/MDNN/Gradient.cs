using mdnn.Activation_functions.classes;
using mdnn.Layers.classes;



namespace mdnn
{
    static public class Gradient
    {
        public static List<Tensor> GetGradients(double[] target_values, MDNN model)
        {
            List<Layer> layers = model.Layers.Layers;
            List<Tensor?> e = new List<Tensor?>();

            int lastElement = layers.Count() - 1;
            Layer lastLayer = layers[lastElement];


            if (lastLayer.Layer_output == null || lastLayer.Layer_raw_output == null)
            {
                throw new ArgumentException("Feedforward must be run before backpropagation. it is recommended to use trainLoop() in the Train class or fit()");
            }

            if (lastLayer.Activation_Func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = lastLayer.Activation_Func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                layerActivationFunc.DerivativeForLayer(lastLayer.Layer_raw_output.Data);
            }

            double[] de = new double[lastLayer.Layer_output.Data.Length];

            for (int i = 0; i < lastLayer.Layer_output.Data.Length; i++)
            {
                double Output = lastLayer.Layer_raw_output.Data[i];

                double derivateActivationOutputLayer = lastLayer.Activation_Func.Derivative(Output);

                de[i] = (GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(lastLayer.Layer_output.Data[i], target_values[i]) * derivateActivationOutputLayer);
            }

            e.Add(new Tensor(de));

            GeneralNeuralNetworkSettings.loss_func.CalculateLoss(lastLayer.Layer_output.Data, target_values);

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
            return e as List<Tensor>;
        }
        public static async Task<List<Tensor>> GetGradientsAsync(double[] target_values, MDNN model)
        {
            List<Layer> layers = model.Layers.Layers;
            List<Tensor?> e = new List<Tensor?>();

            int lastElement = layers.Count() - 1;
            Layer lastLayer = layers[lastElement];


            if (lastLayer.Layer_output == null || lastLayer.Layer_raw_output == null)
            {
                throw new ArgumentException("Feedforward must be run before backpropagation. it is recommended to use trainLoop() in the Train class or fit()");
            }

            if (lastLayer.Activation_Func.Apply_to_layer)
            {
                LayerActivationFunc? layerActivationFunc = lastLayer.Activation_Func as LayerActivationFunc;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                layerActivationFunc.DerivativeForLayer(lastLayer.Layer_raw_output.Data);
            }

            double[] de = new double[lastLayer.Layer_output.Data.Length];

            Task[] tasks = new Task[lastLayer.Layer_output.Data.Length];
            for (int i = 0; i < lastLayer.Layer_output.Data.Length; i++)
            {
                int index = i;
                tasks[index] = Task.Run(() =>
                {
                    double Output = lastLayer.Layer_raw_output.Data[index];

                    double derivateActivationOutputLayer = lastLayer.Activation_Func.Derivative(Output);

                    de[index] = (GeneralNeuralNetworkSettings.loss_func.DerivativeOfLossFunction(lastLayer.Layer_output.Data[index], target_values[index]) * derivateActivationOutputLayer);
                });
            }

            await Task.WhenAll(tasks);

            e.Add(new Tensor(de));

            GeneralNeuralNetworkSettings.loss_func.CalculateLoss(lastLayer.Layer_output.Data, target_values);

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
            return e as List<Tensor>;
        }
    }
}

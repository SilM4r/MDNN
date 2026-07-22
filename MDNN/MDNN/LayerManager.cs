using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;


namespace My_DNN
{
    public class LayerManager
    {
        private List<Layer> layersList = new List<Layer>();
        private int[] number_of_penultimate_output_in_Layer
        {
            get 
            {
                switch (layersList.Count())
                {
                    case 1:
                        return layersList[layersList.Count() - 1].Input_size_and_shape;
                    case 0:
                        return new int[] { -1 };
                    default: 
                        return layersList[layersList.Count() - 2].Output_size_and_shape; ;
                }
            }
        }
        public List<Layer> Layers
        {
            get { return layersList; }
        }
        private NetworkContext context;

        public LayerManager(Layer Output_Layer, NetworkContext context)
        {
            this.context = context;
            Output_Layer.Context = context;
            layersList = new List<Layer> { Output_Layer };
        }
        public LayerManager(List<BaseExportLayer> exportLayers, NetworkContext context)
        {
            this.context = context;
            layersList = new List<Layer>();

            foreach (BaseExportLayer layer in exportLayers)
            {

                if (layer.LayerType == "Dense")
                {
                    layersList.Add(new Dense((ExportDenseLayer)layer));
                }
                else if (layer.LayerType == "RNN")
                {
                    layersList.Add(new RNN((ExportRnnLayer)layer));
                }
                else if (layer.LayerType == "Conv")
                {
                    layersList.Add(new Conv((ExportCNNLayer)layer));
                }
                else if (layer.LayerType == "MaxPool")
                {
                    layersList.Add(new MaxPool((ExportMaxPoolLayer)layer));
                }
                else
                {
                    throw new Exception("not implemetet yet");
                }
            }

            foreach (Layer l in layersList) l.Context = context;
        }
        public void SetInputSizeForFirstLayer(int[]? input_size = null)
        {
            if (input_size != null)
            {
                context.InputShape = input_size;
            }

            foreach (int size in context.InputShape)
            {
                if (size <= 0)
                {
                    throw new Exception("the number of inputs to the first layer must be greater than zero, please");
                }
            }

            // 1. vrstva dostane vstup modelu; každá další se přestaví vstupem = výstup předchozí.
            // Jde zleva doprava, takže Output_size_and_shape předchozí vrstvy je už správně
            // nastavené (nutné hlavně pro Conv/MaxPool, jejichž výstup závisí na vstupu).
            // Bez rozlišení typu — dvě neuronové vrstvy za sebou (Dense→Dense) se JINAK
            // nepřestaví a druhá zůstane s placeholder neurony (0 vah) → IndexOutOfRange.
            layersList[0].LayerAdjustment(null, context.InputShape);
            for (int i = 1; i < layersList.Count; i++)
            {
                layersList[i].LayerAdjustment(null, layersList[i - 1].Output_size_and_shape);
            }
        }
        public void Add(Layer Hidden_Layer)
        {
            Hidden_Layer.Context = context;
            layersList.Insert(layersList.Count() - 1, Hidden_Layer);
            layersList[layersList.Count() - 1].LayerAdjustment(null, number_of_penultimate_output_in_Layer);
        }
        public void Insert(int position, Layer Hidden_Layer)
        {
            Hidden_Layer.Context = context;
            if (position <= layersList.Count())
            {
                int[] New_Layer_Input;

                if (position == 0)
                {
                    New_Layer_Input = layersList[position].Input_size_and_shape;
                }
                else
                {
                    New_Layer_Input = layersList[position - 1].Output_size_and_shape;
                }

                Hidden_Layer.LayerAdjustment(null, New_Layer_Input);
                layersList.Insert(position, Hidden_Layer);

                if (position != (layersList.Count() - 1))
                {
                    layersList[position + 1].LayerAdjustment(null, Hidden_Layer.Output_size_and_shape);
                }
            }
            else
            {
                throw new Exception("Varialbe position must be less or equal than the values ​​of Variable Layers (position <= Layers.Count())");
            }
        }
        public void RemoveAt(int position)
        {
            if (position < layersList.Count())
            {
                int[] New_Layer_Input;

                if (position == 0)
                {
                    New_Layer_Input = layersList[position].Input_size_and_shape;
                }
                else
                {
                    New_Layer_Input = layersList[position - 1].Output_size_and_shape;
                }

                layersList.RemoveAt(position);

                if (layersList.Count() > position)
                {
                    layersList[position].LayerAdjustment(null, New_Layer_Input);
                }
            }
            else
            {
                throw new Exception("Varialbe position must be less than the values ​​of Variable Layers (position < Layers.Count())");
            }
        }
        public void OutputLayerActivationFunc(Activation_func activation_func)
        {
            layersList[layersList.Count() - 1].Activation_Func = activation_func;
        }
        public void ClearAllLayersAndSetNewOutputLayer(Layer Output_Layer)
        {
            Output_Layer.Context = context;
            layersList = new List<Layer> { Output_Layer };
        }
    }
}

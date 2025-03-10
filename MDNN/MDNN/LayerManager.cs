using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Loss_functions;
using My_DNN.Save_neural_network;
using ScottPlot;
using ScottPlot.Palettes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN
{
    public class LayerManager
    {
        private static List<Layer> layersList = new List<Layer>();
        public static int[] number_of_penultimate_output_in_Layer
        {
            get 
            {
                switch (layersList.Count())
                {
                    case 1:
                        return layersList[layersList.Count() - 1].Input_size_and_shape;
                    case 0:
                        if (GeneralNeuralNetworkSettings.modelInputSizeAndShape[0] != 0)
                        {
                            return GeneralNeuralNetworkSettings.modelInputSizeAndShape;
                        }
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
        public LayerManager(Layer Output_Layer)
        {
            layersList = new List<Layer> { Output_Layer };
        }
        public LayerManager(List<BaseExportLayer> exportLayers)
        {
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
        }
        public void SetInputSizeForFirstLayer(int[]? input_size = null)
        {
            if (input_size != null)
            {
                GeneralNeuralNetworkSettings.modelInputSizeAndShape = input_size;
            }

            foreach (int size in GeneralNeuralNetworkSettings.modelInputSizeAndShape)
            {
                if (size <= 0)
                {
                    throw new Exception("the number of inputs to the first layer must be greater than zero, please");
                }
            }

            int[] inputShape = GeneralNeuralNetworkSettings.modelInputSizeAndShape;



            layersList[0].LayerAdjustment(null, inputShape);
            if (layersList.Count() > 1)
            {
                for (int i = 1; i < layersList.Count; i++)
                {
                    if (layersList[i] is not LayerBasedOnNeurons)
                    {
                        inputShape = layersList[i - 1].Output_size_and_shape;
                        layersList[i].LayerAdjustment(null, inputShape);
                    }
                    else if (layersList[i] is LayerBasedOnNeurons && layersList[i-1] is not LayerBasedOnNeurons)
                    {
                        inputShape = layersList[i - 1].Output_size_and_shape;
                        layersList[i].LayerAdjustment(null, inputShape);
                    }
                }
            }
        }
        public void Add(Layer Hidden_Layer)
        {
            layersList.Insert(layersList.Count() - 1, Hidden_Layer);
            layersList[layersList.Count() - 1].LayerAdjustment(null, number_of_penultimate_output_in_Layer);
        }
        public void Insert(int position, Layer Hidden_Layer)
        {
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
            layersList = new List<Layer> { Output_Layer };
        }
    }
}

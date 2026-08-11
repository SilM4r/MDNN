using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;


namespace My_DNN
{
    public class LayerManager
    {
        private List<Layer> layersList = new List<Layer>();

        public List<Layer> Layers
        {
            get { return layersList; }
        }

        // Má některá vrstva zapojený tvar, ale ještě ne parametry? Tohle je signál pro
        // model, že je před forwardem potřeba doběhnout materializaci.
        public bool HasUnmaterializedLayers => layersList.Any(layer => !layer.IsMaterialized);
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

            RewireChain();
            MaterializeAll();
        }

        // Zapojí tvary zleva doprava. Nic nelosuje, takže se to smí volat kolikrát chce —
        // po každé změně struktury. Když vstup modelu ještě neznáme, neudělá nic a tvary
        // se dopočítají až při prvním forwardu.
        public void RewireChain()
        {
            int[] inputShape = EffectiveInputShape();

            if (inputShape.Length == 0 || inputShape[0] <= 0)
            {
                return;
            }

            layersList[0].WireShapes(null, inputShape);

            for (int i = 1; i < layersList.Count; i++)
            {
                layersList[i].WireShapes(null, layersList[i - 1].Output_size_and_shape);
            }
        }

        // Materializace JEDNÍM průchodem zleva doprava. Díky tomu je pořadí losování funkcí
        // architektury, ne historie skládání — dvě stejné sítě se stejným seedem tak vyjdou
        // stejně bez ohledu na to, jestli vznikly přes Add() nebo Insert().
        public void MaterializeAll()
        {
            foreach (Layer layer in layersList)
            {
                layer.MaterializeParameters();
            }
        }

        // Vstupní tvar modelu. Přednost má Context; u NAČTENÉHO modelu ho ale Context nezná,
        // zatímco první vrstva ano (má ho ze souboru) — bez téhle větve by se po Add()
        // do načteného modelu nezapojila nová vrstva a forward spadl na IndexOutOfRange.
        private int[] EffectiveInputShape()
        {
            if (context.InputShape.Length > 0 && context.InputShape[0] > 0)
            {
                return context.InputShape;
            }

            if (layersList.Count > 0)
            {
                int[] firstLayerInput = layersList[0].Input_size_and_shape;
                if (firstLayerInput.Length > 0 && firstLayerInput[0] > 0)
                {
                    return firstLayerInput;
                }
            }

            return new int[] { 0 };
        }

        public void Add(Layer Hidden_Layer)
        {
            Hidden_Layer.Context = context;
            layersList.Insert(layersList.Count() - 1, Hidden_Layer);
            RewireChain();
        }
        public void Insert(int position, Layer Hidden_Layer)
        {
            if (position > layersList.Count())
            {
                throw new Exception("Varialbe position must be less or equal than the values ​​of Variable Layers (position <= Layers.Count())");
            }

            Hidden_Layer.Context = context;
            layersList.Insert(position, Hidden_Layer);
            RewireChain();
        }
        public void RemoveAt(int position)
        {
            if (position >= layersList.Count())
            {
                throw new Exception("Varialbe position must be less than the values ​​of Variable Layers (position < Layers.Count())");
            }

            layersList.RemoveAt(position);
            RewireChain();
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

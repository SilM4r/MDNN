
using My_DNN.Save_neural_network;
using My_DNN.Loss_functions;
using My_DNN.Optimizers;
using My_DNN.Layers.classes;
using My_DNN.Layers;

namespace My_DNN
{
    
    public class MDNN
    {
        private Train train;
        private LayerManager layerManager;
        private NetworkContext context = new();
        private string? schema = null;
        public string Note { get; set; }
        public NetworkContext Context => context;
        public string Schema
        {
            get { return $"[{schema}]"; }
        }
        public Train Train 
        {
            get { return train; }
        }
        public LayerManager Layers
        {
            get { return layerManager; }
        }
        public Loss Loss
        {
            get { return context.Loss; }
        }
        public Optimizer Optimizer
        {
            get { return context.Optimizer; }
        }
        private MDNN(NetworkSaveLoadManager loadModel)
        {

            context.Optimizer = Optimizer.Refactor_optimizer(loadModel.Optimizer);
            GeneralNeuralNetworkSettings.optimizer = context.Optimizer;   // export-konstrukce neuronů klonuje static

            layerManager = new LayerManager(loadModel.Layers, context);
            train = new Train(this, loadModel.Current_epoch, loadModel.Target_epoch, loadModel.Mini_batch);
            context.Loss = Loss.inicialization_Loss_func(loadModel.Loss_functions);
            Note = loadModel.Note;
        }
        public MDNN(Layer Output_Layer, Optimizer? optimizer = null, Loss? loss = null)
        {
            if (optimizer != null)
            {
                context.Optimizer = optimizer;
            }

            if (loss != null)
            {
                context.Loss = loss;
            }

            layerManager = new LayerManager(Output_Layer, context);

            train = new Train(this);
            Note = "";
        }
        public Tensor GetResults(Tensor inputs_values)
        {

            if (Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                context.InputShape = inputs_values.Shape;
                Layers.SetInputSizeForFirstLayer();
            }

            Tensor values = inputs_values;

            foreach (Layer layer in layerManager.Layers) 
            {
                values = layer.FeedForward(values);
            }

            return values;
        }
        public void ResetSequence()
        {
            foreach (Layer layer in Layers.Layers)
            {
                if (layer is RNN)
                {
                    RNN rnn = (RNN)layer;
                    rnn.ResetSequence();
                }
            }
        }
        public async Task<Tensor> GetResultsAsync(Tensor inputs_values)
        {

            if (Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                context.InputShape = inputs_values.Shape;
                Layers.SetInputSizeForFirstLayer();
            }


            Tensor values = inputs_values;

            foreach (Layer layer in layerManager.Layers)
            {
                values = await layer.FeedForwardAsync(values);
            }

            return values;
        }
        public void SaveAsJson(string fileName)
        {
            CreateSchema();
            
            NetworkSaveLoadManager NSLM = new NetworkSaveLoadManager(this);
            NSLM.Save(fileName);
        }
        public static MDNN LoadModel(string fullPath)
        {
            NetworkSaveLoadManager loadModel = NetworkSaveLoadManager.Load(fullPath);
            return new MDNN(loadModel);
        }
        // serializace TOHOTO modelu do JSON stringu (pro in-memory snapshot nejlepšího modelu)
        public string SaveAsJsonString()
        {
            CreateSchema();
            return new NetworkSaveLoadManager(this).SaveToString();
        }

        // In-place načtení uloženého modelu do TÉTO instance: přepíše vrstvy/optimizer/loss
        // stavem ze snapshotu, ALE zachová identitu objektu i jeho Train (a tím i rozdělené
        // datasety). Volající tak drží dál stejný `model`, který je teď ten nejlepší —
        // žádná divergence jako u `model = LoadModel(...)`. context (a jeho InputShape /
        // SequenceTrain) se nechává, takže testovací větev pozná sekvenční režim správně.
        public void LoadWeightsInPlace(string fullPath)
        {
            ApplyLoaded(NetworkSaveLoadManager.Load(fullPath));
        }

        // stejné jako LoadWeightsInPlace, ale z in-memory JSON stringu (snapshot nejlepšího modelu)
        public void LoadWeightsFromString(string json)
        {
            ApplyLoaded(NetworkSaveLoadManager.LoadFromString(json));
        }

        private void ApplyLoaded(NetworkSaveLoadManager loaded)
        {
            context.Optimizer = Optimizer.Refactor_optimizer(loaded.Optimizer);
            GeneralNeuralNetworkSettings.optimizer = context.Optimizer;   // export-konstrukce neuronů klonuje static
            context.Loss = Loss.inicialization_Loss_func(loaded.Loss_functions);

            layerManager = new LayerManager(loaded.Layers, context);
            Note = loaded.Note;
            schema = null;   // zneplatnit cache schématu (přepočítá se z nových vrstev)
        }
        public void info()
        {
            CreateSchema();
            ConsoleControler.ShowModelInfo(this);

        }
        private void CreateSchema()
        {
            if (this.schema != null)
            {
                return;
            }

            string schema = "";

            foreach (Layer layer in layerManager.Layers)
            {
                schema += layer.Name + ",";
            }
            schema = schema.Remove(schema.Length - 1);
            this.schema = schema;
        }
    }
}

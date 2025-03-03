
using mdnn.Save_neural_network;
using System.Collections.Generic;
using mdnn.Activation_functions;
using mdnn.Loss_functions;
using mdnn.Optimizers;
using mdnn.Layers.classes;
using System.Reflection;
using mdnn.Layers;

namespace mdnn
{

    public class MDNN
    {

        /*
            1 async funkce

            povině 

            1 učení do konkrétní chybovosti (aktuálně je jen počet iterací)
            3 ukládání jen gradientů


            3 jiný pooling
            6 BatchNormalization

        */

        private Train train;
        private LayerManager layerManager;
    
   
        private string? schema = null;

        public string Note { get; set; }

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
            get { return GeneralNeuralNetworkSettings.loss_func; }
        }

        public Optimizer Optimizer
        {
            get { return GeneralNeuralNetworkSettings.optimizer; }
        }
        public MDNN(NetworkSaveLoadManager loadModel)
        {

            GeneralNeuralNetworkSettings.optimizer = Optimizer.Refactor_optimizer(loadModel.Optimizer);

            layerManager = new LayerManager(loadModel.Layers);
            train = new Train(this, loadModel.Current_epoch, loadModel.Target_epoch, loadModel.Mini_batch);
            GeneralNeuralNetworkSettings.loss_func = Loss.inicialization_Loss_func(loadModel.Loss_functions);
            Note = loadModel.Note;
        }

        public MDNN(Layer Output_Layer, Optimizer? optimizer = null, Loss? loss = null)
        {
            if (optimizer != null)
            {
                GeneralNeuralNetworkSettings.optimizer = optimizer;
            }

            if (loss != null)
            {
                GeneralNeuralNetworkSettings.loss_func = loss;
            }

            layerManager = new LayerManager(Output_Layer);

            train = new Train(this);
            Note = "";
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

        public Tensor FeedForward(Tensor inputs_values)
        {

            if (Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0 }).Shape;
                Layers.SetInputSizeForFirstLayer();
            }


            Tensor values = inputs_values;

            foreach (Layer layer in layerManager.Layers) 
            {
                values = layer.FeedForward(values);
            }

            return values;
        }

        public async Task<Tensor> FeedForwardAsync(Tensor inputs_values)
        {

            if (Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0 }).Shape;
                Layers.SetInputSizeForFirstLayer();
            }


            Tensor values = inputs_values;

            foreach (Layer layer in layerManager.Layers)
            {
                values = await layer.FeedForwardAsync(values);
            }

            return values;
        }

        public double[] FeedForward(double[] inputs_values)
        {
            Tensor values = new Tensor(inputs_values);
            return (double[])FeedForward(values).GetOriginalData();
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

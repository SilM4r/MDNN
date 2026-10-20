
using MDNN.Serialization;
using MDNN.Losses;
using MDNN.Optimizers;
using MDNN.Layers;

namespace MDNN
{

    /// <summary>
    /// A feed-forward neural network model: its layers, training settings and save/load.
    /// </summary>
    public class NeuralNetwork
    {
        private Train train;
        private LayerManager layerManager;
        private NetworkContext context = new();
        /// <summary>
        /// Free-form note stored with the model file.
        /// </summary>
        public string Note { get; set; }
        /// <summary>
        /// The model's configuration (loss, optimizer, random source, regularization).
        /// </summary>
        public NetworkContext Context => context;
        // Computed from the current layers every time — previously it was cached and
        // Add/Insert/RemoveAt did not invalidate the cache, so a file saved after Info() and a
        // subsequent Add() carried a stale schema.
        /// <summary>
        /// Layer names in order, e.g. <c>"[Dense,Dense]"</c>.
        /// </summary>
        public string Schema => $"[{string.Join(",", layerManager.Layers.Select(layer => layer.Name))}]";
        /// <summary>
        /// Training settings, datasets and training loops of this model.
        /// </summary>
        public Train Train
        {
            get { return train; }
        }
        /// <summary>
        /// The model's layers.
        /// </summary>
        public LayerManager Layers
        {
            get { return layerManager; }
        }
        /// <summary>
        /// The loss function (shortcut for <c>Context.Loss</c>).
        /// </summary>
        public Loss Loss
        {
            get { return context.Loss; }
        }
        /// <summary>
        /// The optimizer (shortcut for <c>Context.Optimizer</c>).
        /// </summary>
        public Optimizer Optimizer
        {
            get { return context.Optimizer; }
        }
        private NeuralNetwork(NetworkSaveLoadManager loadModel, int? seed = null, Optimizer? optimizer = null, Loss? loss = null)
        {
            // Weights come from the file, so the seed does not affect initialization — but it does
            // affect dataset shuffling and sample selection, which matters when continuing training.
            //
            // An explicitly passed seed wins; otherwise the one from the file is used (saved since
            // format v1), so a loaded model knows how it was created. If the file has no seed (old
            // format), it stays null.
            int? effectiveSeed = seed ?? loadModel.Seed;

            if (effectiveSeed != null)
            {
                context.Seed = effectiveSeed;
            }

            // Regularization is part of the training recipe — without restoring it, continued
            // training of a loaded model would silently run without it. Old format lacks the keys → 0 = off.
            context.L2 = loadModel.L2;
            context.Dropout = loadModel.Dropout;

            // Built-in ones are constructed from the file; custom or overriding ones come from the caller.
            context.Optimizer = Optimizer.ResolveForLoad(loadModel.Optimizer, optimizer, ownWins: true);

            layerManager = new LayerManager(loadModel.Layers, context);
            train = new Train(this, loadModel.Current_epoch, loadModel.Target_epoch, loadModel.Mini_batch);
            context.Loss = Loss.ResolveForLoad(loadModel.Loss_functions, loss, ownWins: true);
            Note = loadModel.Note;
        }
        /// <summary>
        /// Creates a model ending with <paramref name="outputLayer"/>; more layers are added via <see cref="Layers"/>.
        /// </summary>
        /// <param name="outputLayer">The output layer.</param>
        /// <param name="optimizer">Optimizer; defaults to SGD.</param>
        /// <param name="loss">Loss function; defaults to MSE.</param>
        /// <param name="seed">
        /// Seed for reproducibility. Affects weight initialization, dataset shuffling and sample selection —
        /// everything random the model does. Two models with the same seed, architecture and data give
        /// bit-identical results. Without a seed the shared global generator is used.
        /// </param>
        public NeuralNetwork(Layer outputLayer, Optimizer? optimizer = null, Loss? loss = null, int? seed = null)
        {
            if (optimizer != null)
            {
                context.Optimizer = optimizer;
            }

            if (loss != null)
            {
                context.Loss = loss;
            }

            if (seed != null)
            {
                context.Seed = seed;
            }

            layerManager = new LayerManager(outputLayer, context);

            train = new Train(this);
            Note = "";
        }
        /// <summary>
        /// Runs a forward pass and returns the network's output for <paramref name="inputs"/>.
        /// </summary>
        public Tensor GetResults(Tensor inputs)
        {
            EnsureReadyForInput(inputs);

            Tensor values = inputs;

            foreach (Layer layer in layerManager.Layers)
            {
                values = layer.FeedForward(values);
            }

            return values;
        }
        // The model is ready when every layer has its shape wired AND its parameters drawn.
        //
        // Previously only `Layers[0].InputShape[0] <= 0` ("input unknown") was checked. That is
        // fragile: in a LOADED model the first layer has its input from the file, so the check failed
        // and a layer added via Add() stayed unwired until an IndexOutOfRange in the forward pass.
        // The second branch covers that case.
        private void EnsureReadyForInput(Tensor inputs)
        {
            if (Layers.Layers[0].InputShape[0] <= 0)
            {
                // the input shape is learned only now, from the first sample
                context.InputShape = inputs.Shape;
                Layers.SetInputShape();
            }
            else if (Layers.HasUnmaterializedLayers)
            {
                // shapes are wired (Add/Insert recomputed them), only parameters are missing
                Layers.MaterializeAll();
            }
        }

        /// <summary>
        /// Resets the hidden state of all recurrent (RNN) layers.
        /// </summary>
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
        /// <summary>
        /// Saves the model to a JSON file.
        /// </summary>
        /// <param name="fileName">Target file path.</param>
        /// <param name="includeOptimizerState">
        /// Also save the optimizer state (Adam's moments and step) so training can resume seamlessly.
        /// With Adam this makes the file roughly 3–4× larger; for a prediction-only model pass false.
        /// </param>
        public void SaveAsJson(string fileName, bool includeOptimizerState = true)
        {
            NetworkSaveLoadManager NSLM = new NetworkSaveLoadManager(this, includeOptimizerState);
            NSLM.Save(fileName);
        }

        /// <summary>
        /// Loads a model saved with <see cref="SaveAsJson"/>.
        /// </summary>
        /// <param name="path">Path to the model file.</param>
        /// <param name="seed">Seed for shuffling/sampling; defaults to the seed stored in the file.</param>
        /// <param name="optimizer">
        /// Own instance for a model saved with a custom optimizer (loading cannot construct it by name);
        /// must have the same name as in the file. Can also override a built-in one, e.g. Adam with a lower
        /// learning rate for fine-tuning; the saved optimizer state is loaded into the passed instance.
        /// </param>
        /// <param name="loss">Own instance for a model saved with a custom loss; must have the same name as in the file.</param>
        public static NeuralNetwork LoadModel(string path, int? seed = null, Optimizer? optimizer = null, Loss? loss = null)
        {
            NetworkSaveLoadManager loadModel = NetworkSaveLoadManager.Load(path);
            return new NeuralNetwork(loadModel, seed, optimizer, loss);
        }
        // Replica for parallel training (see ReplicaPool): ready-made layers, its own context
        // and its own Train, so Fit can be called independently of the original.
        internal NeuralNetwork(NetworkContext context, List<Layer> replicaLayers)
        {
            this.context = context;
            layerManager = LayerManager.ForReplica(replicaLayers, context);
            train = new Train(this);
            Note = "";
        }

        /// <summary>
        /// Serializes this model to a JSON string (e.g. for an in-memory snapshot of the best model).
        /// </summary>
        /// <param name="includeOptimizerState">Also include the optimizer state; see <see cref="SaveAsJson"/>.</param>
        public string SaveAsJsonString(bool includeOptimizerState = true)
        {
            return new NetworkSaveLoadManager(this, includeOptimizerState).SaveToString();
        }

        // Overwrites layers/optimizer/loss with the snapshot BUT keeps the object identity and its Train
        // (and thus the split datasets). The caller keeps the same `model`, which is now the best one —
        // no divergence as with `model = LoadModel(...)`. The context (and its InputShape /
        // SequenceTrain) is kept, so the test branch detects sequence mode correctly.
        /// <summary>
        /// Loads a saved model into THIS instance, keeping the object identity and its <see cref="Train"/>.
        /// </summary>
        /// <param name="path">Path to the model file.</param>
        public void LoadWeightsInPlace(string path)
        {
            ApplyLoaded(NetworkSaveLoadManager.Load(path));
        }

        /// <summary>
        /// Same as <see cref="LoadWeightsInPlace"/>, but from an in-memory JSON string (e.g. a best-model snapshot).
        /// </summary>
        /// <param name="json">JSON produced by <see cref="SaveAsJsonString"/>.</param>
        public void LoadWeightsFromString(string json)
        {
            ApplyLoaded(NetworkSaveLoadManager.LoadFromString(json));
        }

        private void ApplyLoaded(NetworkSaveLoadManager loaded)
        {
            // The layers are rebuilt from the file right away — no point propagating the optimizer
            // change into the current ones.
            context.OptimizerChanged = null;

            // A custom optimizer/loss that cannot be constructed by name is taken from the model's
            // context. Without this, TrainLoop with a custom optimizer or loss crashed at the end when
            // restoring the best model from the snapshot.
            context.Optimizer = Optimizer.ResolveForLoad(loaded.Optimizer, context.Optimizer, ownWins: false);
            context.Loss = Loss.ResolveForLoad(loaded.Loss_functions, context.Loss, ownWins: false);

            layerManager = new LayerManager(loaded.Layers, context);
            Note = loaded.Note;
        }
        /// <summary>
        /// Prints a summary of the model (layers, shapes, parameters) to the console.
        /// </summary>
        public void Info()
        {
            ConsoleController.ShowModelInfo(this);

        }
    }
}

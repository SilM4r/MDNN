using System.Runtime.ExceptionServices;
using MDNN.Layers;

namespace MDNN
{
    // Data-parallel training of one batch across model replicas (Train.Workers > 1).
    //
    // One batch:
    //   1. the original copies into the replicas the parameters not shared by reference (neuron biases)
    //   2. the batch is split into contiguous chunks and each replica computes its own on its own thread
    //      (forward + backward, gradients accumulate in the replica)
    //   3. gradients and loss are added to the original IN A FIXED replica ORDER
    //   4. UpdateParams on the original (done by the caller, single-threaded)
    //
    // Why replicas and not locks or per-thread state: layers keep activations in fields
    // (`output`, `raw_output`, `Neuron.inputs`, …), so only one sample may pass through a layer
    // at a time. A replica gives each thread its own set of those fields without changing the
    // layer internals. Weights are not copied — the replica shares them with the original and only reads them.
    //
    // Determinism: double addition is not associative, so the result depends on how the batch is
    // split. But both the split and the summation order depend only on the number of replicas, not
    // on which thread finishes first — the same seed and the same Workers give a bit-identical
    // result on any machine.
    internal sealed class ReplicaPool
    {
        private readonly NeuralNetwork _original;
        private readonly NeuralNetwork[] _replicas;

        // `training` = the replicas will train (RunBatch). false = evaluate only (Map):
        // forward without dropout, so they need no random generator at all.
        public ReplicaPool(NeuralNetwork original, int workers, bool training = true)
        {
            // Outside sequence training an RNN carries its hidden state from one sample to the NEXT
            // (nothing resets it between samples). Sequentially the state flows in batch order; in
            // parallel each replica would carry it along its own chunk and the result would silently differ.
            // In sequence mode this is fine — a replica takes a whole sequence and the state is reset.
            if (!original.Context.SequenceTrain && original.Layers.Layers.Any(l => l is Layers.RNN))
            {
                throw new NotSupportedException(
                    "Outside sequence training an RNN layer carries its hidden state from one sample to the next, so it cannot " +
                    "be trained in parallel (Train.Workers > 1). Use sequence training " +
                    "(TrainLoop(..., isSequence: true)), or Train.Workers = 1.");
            }

            _original = original;
            _replicas = new NeuralNetwork[workers];

            // A replica needs randomness only for dropout masks (parameters are already drawn).
            // Without dropout the original's generator is NOT drawn from — every draw would advance
            // its state and the next epoch would shuffle data differently than with Workers = 1, so
            // enabling parallelism would change training by more than rounding.
            // During evaluation dropout is off, so nothing is ever drawn — otherwise validation between
            // epochs would advance the generator and change the data order of the next epoch.
            bool needsRandom = training
                               && original.Layers.Layers.Any(l => l is Layers.Dense d && d.UsesDropout);

            for (int w = 0; w < workers; w++)
            {
                var layers = new List<Layer>();
                foreach (Layer layer in original.Layers.Layers)
                {
                    layers.Add(layer.CreateReplica() ?? throw new NotSupportedException(
                        $"Layer {layer.GetType().Name} does not support parallel training " +
                        "(Train.Workers > 1). Set Train.Workers = 1."));
                }

                // The replica's seed is drawn from the original's generator: it remains the model's
                // single source of randomness, so a seeded run is reproducible. Without dropout the
                // replica gets a fixed seed — still its own instance, because a context without a
                // generator would fall back to the SHARED global one, which is not thread-safe.
                int seed = needsRandom ? original.Context.Random.Next() : 0;
                NetworkContext context = original.Context.CreateReplica(seed);
                _replicas[w] = new NeuralNetwork(context, layers);
            }
        }

        // Calls `run(replica, index)` for each index 0..count-1 and returns the results IN INDEX
        // ORDER, not in completion order. Read-only use of the model (forward for validation and test):
        // nothing is added to the original, so the result does not depend on how work is split.
        // The caller then processes the results sequentially, exactly as with Workers = 1.
        public T[] Map<T>(int count, Func<NeuralNetwork, int, T> run)
        {
            var results = new T[count];
            int used = Math.Min(_replicas.Length, count);
            if (used == 0) return results;

            SyncReplicas(used);

            RunOnReplicas(used, w =>
            {
                int from = count * w / used;
                int to = count * (w + 1) / used;

                for (int k = from; k < to; k++)
                {
                    results[k] = run(_replicas[w], k);
                }
            });

            return results;
        }

        // Before work: copy into the replicas the parameters not shared by reference (neuron biases).
        private void SyncReplicas(int used)
        {
            IReadOnlyList<Layer> originalLayers = _original.Layers.Layers;

            for (int w = 0; w < used; w++)
            {
                IReadOnlyList<Layer> replicaLayers = _replicas[w].Layers.Layers;
                for (int i = 0; i < originalLayers.Count; i++)
                {
                    originalLayers[i].SyncReplica(replicaLayers[i]);
                }
            }
        }

        private static void RunOnReplicas(int used, Action<int> body)
        {
            try
            {
                Parallel.For(0, used, new ParallelOptions { MaxDegreeOfParallelism = used }, body);
            }
            catch (AggregateException e)
            {
                // The caller should get the same exception as on the sequential path (e.g. an
                // ArgumentException about a mismatched target count), not the Parallel.For wrapper.
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
                throw;
            }
        }

        // Computes samples order[start..end) and accumulates their gradients into the original.
        // `fitOne` gets the replica's Train and the sample index — the same code that the
        // sequential path calls on the original.
        public void RunBatch(int[] order, int start, int end, Action<Train, int> fitOne)
        {
            int count = end - start;
            int used = Math.Min(_replicas.Length, count);
            IReadOnlyList<Layer> originalLayers = _original.Layers.Layers;

            SyncReplicas(used);

            RunOnReplicas(used, w =>
            {
                int from = start + count * w / used;
                int to = start + count * (w + 1) / used;

                for (int k = from; k < to; k++)
                {
                    fitOne(_replicas[w].Train, order[k]);
                }
            });

            // Fixed order 0, 1, 2, … — not by who finishes first.
            for (int w = 0; w < used; w++)
            {
                IReadOnlyList<Layer> replicaLayers = _replicas[w].Layers.Layers;
                for (int i = 0; i < originalLayers.Count; i++)
                {
                    originalLayers[i].TakeGradientsFrom(replicaLayers[i]);
                }

                _original.Context.Loss.TakeAccumulatedLossFrom(_replicas[w].Context.Loss);
            }
        }
    }
}

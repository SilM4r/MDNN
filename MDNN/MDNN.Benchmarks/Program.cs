using System.Diagnostics;
using MDNN;
using MDNN.Activations;
using MDNN.Layers;
using MDNN.Losses;
using MDNN.Optimizers;

namespace MDNN.Benchmarks
{
    // Profile of a REAL training step, not of isolated methods.
    //
    // Goal: find out where time and memory actually go before optimizing anything.
    // Specifically, whether conv is the bottleneck, and how much is eaten by things that
    // have nothing to do with parallelization (Tensor allocations, boxing in Convert.ToDouble,
    // copying activations).
    //
    // Both time and ALLOCATIONS are measured. Allocations matter as much as time here: the
    // per-neuron model and the Tensor returned from every getter are the prime suspects, and
    // they show up in time only indirectly, through GC pressure.
    internal static class Program
    {
        // ==================================================================================
        //  BASELINE — what the current run is compared against
        // ==================================================================================
        //
        //  Measured 2026-08-26, Apple Silicon / 10 cores, .NET 9, Release, workstation GC.
        //  Median of three runs. The profile below prints, for every row, how far time and
        //  allocations moved relative to these numbers.
        //
        //  Absolute values depend on the machine — on another computer the whole
        //  "vs base" column will be shifted and it means nothing. It only makes sense when
        //  comparing on the SAME machine. When the scenario changes (different architecture,
        //  different sample count), the baseline must be re-measured, otherwise you compare
        //  apples to oranges.
        //
        //  ⚠️ ALLOCATIONS are deterministic (three runs give bit-identical results), TIMES vary
        //  between runs by up to 16 %. The "vs base" time column has a 5 % noise threshold, so
        //  you will still see swings that mean nothing. Trust allocations and large numbers more.
        //
        // ----------------------------------------------------------------------------------
        //  HISTORY — what moved where
        // ----------------------------------------------------------------------------------
        //
        //  Previous baseline: 2026-08-12, commit b0168bb, same machine. That was the state
        //  BEFORE the data-layer optimizations and before convention B. Main finding back then:
        //  the bottleneck is not the math but the data layer — Tensor.FlattenArray boxed
        //  every double (foreach over System.Array) into a List<double>, so a 6 KB payload
        //  cost ~40 KB of allocations.
        //
        //   MLP 784-128-10, 32 samples           2026-08-12  →  2026-08-26
        //     phase/1. forward                73.5 ms /  1.5 MB  →  76.7 ms / 240 KB
        //     phase/2. gradient chain          2.1 ms / 184 KB   ┐ merged by convention B into
        //     phase/3. weight accumulation    70.8 ms / 198 KB   ┘ "2. backward" 73.3 ms / 312 KB
        //     phase/4. parameter update        3.4 ms /     0 B  →   3.4 ms /     0 B
        //     forward/Dense(128)              72.4 ms /  1.3 MB  →  75.7 ms / 232 KB
        //     forward/Dense(10)                1.1 ms / 208 KB   →   1.0 ms /   9 KB
        //     accumulation/Dense(128)         69.8 ms / 198 KB   →  72.2 ms / 269 KB
        //     accumulation/Dense(10)           0.9 ms /     0 B  →   1.2 ms /  43 KB
        //
        //   CNN 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10, 8 samples
        //     phase/1. forward                27.9 ms / 31.6 MB  →  10.0 ms / 1.2 MB
        //     phase/2. gradient chain         29.7 ms / 32.2 MB  ┐ merged into
        //     phase/3. weight accumulation     7.5 ms /     0 B  ┘ "2. backward" 19.8 ms / 2.8 MB
        //     phase/4. parameter update        3.6 ms /    80 B  →   3.7 ms /    80 B
        //     forward/Conv(16x3)              13.6 ms / 19.2 MB  →   2.8 ms / 726 KB
        //     forward/Conv(32x3)               9.2 ms /  6.3 MB  →   5.7 ms / 243 KB
        //     forward/MaxPool                  3.9 ms /  5.8 MB  →   0.7 ms / 221 KB
        //     forward/Dense(64)                1.1 ms / 332 KB   →   0.9 ms / 4.8 KB
        //     forward/Dense(10)                0.0 ms /  28 KB   →   0.0 ms / 2.2 KB
        //     accumulation/Conv(32x3)          4.8 ms /     0 B  →  13.9 ms / 384 KB
        //     accumulation/Conv(16x3)          1.6 ms /     0 B  →   3.6 ms / 302 KB
        //     accumulation/Dense(64)           1.0 ms /     0 B  →   1.1 ms /  60 KB
        //     accumulation/Dense(10)           0.0 ms /     0 B  →   0.0 ms / 6.8 KB
        //
        //   Data-layer overhead, 20 000 calls
        //     GetSubTensor([i])            464.0 ms / 963.6 MB →   6.0 ms / 122.1 MB
        //     Tensor.Data (flatten)          409.0 ms / 800.5 MB →   6.8 ms / 124.4 MB
        //     Output + .Data           105.0 ms / 125.3 MB →   1.7 ms /  21.4 MB
        //     Output (getter)            1.6 ms /   4.3 MB →   1.9 ms /  21.4 MB
        //     new Tensor(double[784])          1.5 ms /   4.3 MB →   1.5 ms /   4.3 MB
        //
        //  How to read the two rows that look like regressions:
        //
        //  1. "Output (getter)" got more expensive because the output copy MOVED into it.
        //     It used to return a window into the layer's working array and the copy was made
        //     only in .Data — but by then the result could already be overwritten by the next
        //     pass. Read the pair together: getter + .Data dropped from 105 ms to 1.7 ms.
        //  2. "accumulation/Conv" and "accumulation/Dense" now have non-zero allocations and
        //     higher times because convention B moved the ∂L/∂input computation from the
        //     "gradient chain" phase into the layer's backward. No work was added, it is just
        //     accounted elsewhere — the former "gradient chain" row (29.7 ms / 32.2 MB for CNN)
        //     no longer exists in the table.
        //
        //  Overall vs. August: CNN forward -64 % time and -96 % allocations, data-layer
        //  overhead -98 % / -85 %. MLP times did not move — that scenario spends 98 % of its
        //  time in the per-neuron Dense loops, which only a planned FastDense will address.
        // ==================================================================================
        private static readonly Dictionary<string, Cost> Baseline = new()
        {
            // MLP 784-128-10, 32 samples
            ["MLP/phase/1. forward"] = new(76.7, 246_272),
            ["MLP/phase/2. backward"] = new(73.3, 319_744),
            ["MLP/phase/4. parameter update"] = new(3.4, 0),
            ["MLP/forward/Dense(128)"] = new(75.7, 237_312),
            ["MLP/forward/Dense(10)"] = new(1.0, 8_960),
            ["MLP/accumulation/Dense(128)"] = new(72.2, 275_712),
            ["MLP/accumulation/Dense(10)"] = new(1.2, 44_032),

            // CNN, 8 samples
            ["CNN/phase/1. forward"] = new(10.0, 1_225_088),
            ["CNN/phase/2. backward"] = new(19.8, 2_918_656),
            ["CNN/phase/4. parameter update"] = new(3.7, 80),
            ["CNN/forward/Conv(16x3)"] = new(2.8, 743_424),
            ["CNN/forward/Conv(32x3)"] = new(5.7, 248_640),
            ["CNN/forward/MaxPool"] = new(0.7, 225_920),
            ["CNN/forward/Dense(64)"] = new(0.9, 4_864),
            ["CNN/forward/Dense(10)"] = new(0.0, 2_240),
            ["CNN/accumulation/Conv(32x3)"] = new(13.9, 393_728),
            ["CNN/accumulation/Conv(16x3)"] = new(3.6, 309_248),
            ["CNN/accumulation/MaxPool"] = new(0.9, 2_147_008),
            ["CNN/accumulation/Dense(64)"] = new(1.1, 61_760),
            ["CNN/accumulation/Dense(10)"] = new(0.0, 6_912),

            // data-layer overhead, 20 000 calls
            ["overhead/GetSubTensor([i])"] = new(6.0, 128_000_000),
            ["overhead/Tensor.Data (flatten)"] = new(6.8, 130_400_000),
            ["overhead/Output + .Data"] = new(1.7, 22_400_000),
            ["overhead/Output (getter)"] = new(1.9, 22_400_000),
            ["overhead/new Tensor(double[784])"] = new(1.5, 4_480_000),
        };

        // One measurement: time + how many bytes were allocated meanwhile.
        private readonly record struct Cost(double Milliseconds, long Bytes)
        {
            public static Cost operator +(Cost a, Cost b)
                => new(a.Milliseconds + b.Milliseconds, a.Bytes + b.Bytes);
        }

        // ∂L/∂output of the output layer — the input to the Backward chain. Same thing that
        // Gradient.RunBackward computes internally; done separately here so that the backward
        // alone can be measured.
        private static Tensor OutputGradient(NeuralNetwork model, Tensor target)
        {
            double[] output = model.Layers.Layers[^1].Output!.Data;
            double[] d = new double[output.Length];

            for (int i = 0; i < d.Length; i++)
                d[i] = model.Context.Loss.DerivativeOfLossFunction(output[i], target.Data[i]);

            return new Tensor(d);
        }

        private static Cost Measure(Action action)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();

            action();

            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;

            return new Cost(ms, bytes);
        }

        private static void Add(Dictionary<string, Cost> into, string key, Cost cost)
            => into[key] = into.TryGetValue(key, out Cost existing) ? existing + cost : cost;

        private static string Bytes(long b)
            => b >= 1024L * 1024 * 1024 ? $"{b / (1024.0 * 1024 * 1024):F2} GB"
             : b >= 1024L * 1024 ? $"{b / (1024.0 * 1024):F1} MB"
             : b >= 1024 ? $"{b / 1024.0:F1} KB"
             : $"{b} B";

        // Change vs. baseline. Positive = regression, negative = improvement.
        // Marked as new when the row has no baseline (new scenario or new layer).
        private static string VsBaseline(double now, double before)
        {
            if (before <= 0)
            {
                return now <= 0 ? "     ." : "    new";
            }

            double change = (now - before) / before * 100;

            // Below 5 % it is noise at this level of measurement, not a result.
            return Math.Abs(change) < 5 ? "     ~" : $"{change,+6:F0}%";
        }

        private static void Table(string title, Dictionary<string, Cost> rows, string baselinePrefix)
        {
            double totalMs = rows.Values.Sum(c => c.Milliseconds);
            long totalBytes = rows.Values.Sum(c => c.Bytes);

            Console.WriteLine();
            Console.WriteLine(title);
            Console.WriteLine($"  {"",-26} {"time",13} {"share",7} {"vs base",8} "
                            + $"{"allocated",11} {"share",7} {"vs base",8}");

            double baselineMs = 0;
            long baselineBytes = 0;

            foreach ((string name, Cost cost) in rows.OrderByDescending(r => r.Value.Milliseconds))
            {
                double timeShare = totalMs > 0 ? cost.Milliseconds / totalMs * 100 : 0;
                double allocShare = totalBytes > 0 ? (double)cost.Bytes / totalBytes * 100 : 0;

                Baseline.TryGetValue(baselinePrefix + name, out Cost old);
                baselineMs += old.Milliseconds;
                baselineBytes += old.Bytes;

                Console.WriteLine($"  {name,-26} {cost.Milliseconds,10:F1} ms {timeShare,5:F1} % "
                                + $"{VsBaseline(cost.Milliseconds, old.Milliseconds),8} "
                                + $"{Bytes(cost.Bytes),11} {allocShare,5:F1} % "
                                + $"{VsBaseline(cost.Bytes, old.Bytes),8}");
            }

            Console.WriteLine($"  {"TOTAL",-26} {totalMs,10:F1} ms {"",7} "
                            + $"{VsBaseline(totalMs, baselineMs),8} "
                            + $"{Bytes(totalBytes),11} {"",7} "
                            + $"{VsBaseline(totalBytes, baselineBytes),8}");
        }

        // Layer name for the summary — for Dense/RNN including the neuron count, so it is
        // possible to tell which layer is expensive.
        private static string LayerKey(Layer layer)
            => layer switch
            {
                Dense d => $"Dense({d.Neurons.Count})",
                RNN r => $"RNN({r.Neurons.Count})",
                Conv c => $"Conv({c.Kernels.Length}x{c.Kernels[0].Length})",
                _ => layer.Name,
            };

        // Runs `samples` samples through one training step and breaks time and allocations
        // down by phase and by layer.
        private static void ProfileTrainingStep(string title, string baselinePrefix, NeuralNetwork model, double[][] inputs, double[][] targets)
        {
            IReadOnlyList<Layer> layers = model.Layers.Layers;

            // Warm-up: the first pass builds the layers and lets the JIT compile the code paths.
            model.GetResults(new Tensor(inputs[0]));
            model.Train.BackPropagation(new Tensor(targets[0]));
            model.Train.UpdateParams();

            var byPhase = new Dictionary<string, Cost>();
            var forwardByLayer = new Dictionary<string, Cost>();
            var backwardByLayer = new Dictionary<string, Cost>();

            for (int s = 0; s < inputs.Length; s++)
            {
                Tensor value = new Tensor(inputs[s]);

                // --- forward, layer by layer ---
                foreach (Layer layer in layers)
                {
                    Layer captured = layer;
                    Tensor input = value;
                    Tensor? produced = null;

                    Cost cost = Measure(() => produced = captured.FeedForward(input));

                    Add(byPhase, "1. forward", cost);
                    Add(forwardByLayer, LayerKey(layer), cost);
                    value = produced!;
                }

                // --- backward, layer by layer ---
                // Convention B: a single pass. Previously two phases were measured (gradient
                // chain + weight accumulation) because backward was split into two methods. Now
                // each layer does both at once, so it is one phase per layer.
                Tensor grad = OutputGradient(model, new Tensor(targets[s]));

                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    Layer layer = layers[i];
                    bool fused = i == layers.Count - 1
                                 && model.Context.Loss.RequiresSoftmax
                                 && layer.Activation.AppliesToLayer;

                    Tensor produced = grad;
                    Cost cost = Measure(() => produced = layer.Backward(grad, fused));
                    grad = produced;

                    Add(byPhase, "2. backward", cost);
                    Add(backwardByLayer, LayerKey(layer), cost);
                }
            }

            // --- update once per batch ---
            foreach (Layer layer in layers)
            {
                Add(byPhase, "4. parameter update", Measure(() => layer.UpdateParams()));
            }

            Console.WriteLine();
            Console.WriteLine(new string('=', 78));
            Console.WriteLine(title);
            Console.WriteLine(new string('=', 78));

            // The actual topology, not just the label in the title. MDNN has no input layer
            // object — the input size is inferred from the data and only shows up in the weight
            // count of the first layer, so it cannot be read from the model code.
            Console.WriteLine($"  input: {inputs[0].Length} values, {inputs.Length} samples");
            foreach (Layer layer in layers)
            {
                string shape = layer switch
                {
                    Dense d => $"{d.Neurons.Count} neurons, {d.Neurons[0].Weights.Length} weights per neuron",
                    RNN r => $"{r.Neurons.Count} neurons, {r.Neurons[0].Weights.Length} weights per neuron (incl. recurrent)",
                    Conv c => $"{c.Kernels.Length} filters {c.Kernels[0].Length}x{c.Kernels[0][0].Length}x{c.Kernels[0][0][0].Length}"
                              + $", output [{string.Join(",", c.OutputShape)}]",
                    _ => $"output [{string.Join(",", layer.OutputShape)}]",
                };
                Console.WriteLine($"  {layer.Name,-8} {shape}");
            }
            Console.WriteLine($"  trainable parameters: {ConsoleController.CountTrainableParams(model):N0}");

            Table("BY PHASE", byPhase, baselinePrefix + "/phase/");
            Table("FORWARD by layer", forwardByLayer, baselinePrefix + "/forward/");
            Table("WEIGHT ACCUMULATION by layer", backwardByLayer, baselinePrefix + "/accumulation/");
        }

        // Suspects unrelated to parallelization — pure data-layer overhead.
        private static void ProfileOverhead()
        {
            const int repeats = 20_000;

            var model = new NeuralNetwork(new Dense(10, new Linear()), new SGD(0.01), new MSE(), seed: 1);
            model.Layers.Add(new Dense(128, new ReLU()));
            model.GetResults(new Tensor(new double[784]));

            Layer hidden = model.Layers.Layers[0];

            double[] flat = new double[784];
            var jagged = new double[64][];
            for (int i = 0; i < jagged.Length; i++) jagged[i] = new double[784];
            Tensor batch = new Tensor(jagged);

            var rows = new Dictionary<string, Cost>();

            // warm-up
            _ = hidden.Output!.Data;
            _ = batch.GetSubTensor([0]);

            Add(rows, "Output (getter)", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor? t = hidden.Output; }
            }));

            Add(rows, "Output + .Data", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { double[] d = hidden.Output!.Data; }
            }));

            Add(rows, "new Tensor(double[784])", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor t = new Tensor(flat); }
            }));

            Add(rows, "Tensor.Data (flatten)", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { double[] d = new Tensor(flat).Data; }
            }));

            Add(rows, "GetSubTensor([i])", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor t = batch.GetSubTensor([i % 64]); }
            }));

            Console.WriteLine();
            Console.WriteLine(new string('=', 78));
            Console.WriteLine($"DATA-LAYER OVERHEAD ({repeats:N0}x, excluding the actual computation)");
            Console.WriteLine(new string('=', 78));
            Table($"cost of {repeats:N0} calls", rows, "overhead/");

            Console.WriteLine();
            Console.WriteLine("  Note: TrainLoop calls GetSubTensor([i]) for EVERY sample of every epoch,");
            Console.WriteLine("        Gradient reads Output on every backward step.");
        }

        private static (double[][] inputs, double[][] targets) MakeData(int samples, int inputSize, int outputSize, int seed)
        {
            var rnd = new Random(seed);
            var inputs = new double[samples][];
            var targets = new double[samples][];

            for (int i = 0; i < samples; i++)
            {
                inputs[i] = new double[inputSize];
                for (int j = 0; j < inputSize; j++) inputs[i][j] = rnd.NextDouble();

                targets[i] = new double[outputSize];
                targets[i][rnd.Next(outputSize)] = 1;
            }

            return (inputs, targets);
        }

        private static void Main()
        {
            Console.WriteLine($"NeuralNetwork profile — {Environment.ProcessorCount} cores, "
                            + $"{(Environment.Is64BitProcess ? "64" : "32")}bit, "
                            + $"server GC: {System.Runtime.GCSettings.IsServerGC}");
#if DEBUG
            Console.WriteLine("!! DEBUG build — the numbers are useless, run with -c Release");
#endif

            // --- MLP: 784 -> 128 -> 10 ---
            {
                var model = new NeuralNetwork(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy(), seed: 1);
                model.Layers.Add(new Dense(128, new ReLU()));

                var (inputs, targets) = MakeData(32, 784, 10, seed: 1);
                ProfileTrainingStep("MLP 784 -> 128 -> 10   (32 samples, one batch)", "MLP", model, inputs, targets);
            }

            // --- CNN as in mdnn_test: 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10 ---
            {
                var model = new NeuralNetwork(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy(), seed: 1);
                model.Layers.Add(new Conv(16, 3, new ReLU(), "valid"));
                model.Layers.Add(new MaxPool(2));
                model.Layers.Add(new Conv(32, 3, new ReLU(), "valid"));
                model.Layers.Add(new MaxPool(2));
                model.Layers.Add(new Dense(64, new ReLU()));

                var (inputs, targets) = MakeData(8, 784, 10, seed: 2);
                ProfileTrainingStep("CNN 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10   (8 samples)",
                    "CNN", model, inputs, targets);
            }

            ProfileOverhead();
        }
    }
}

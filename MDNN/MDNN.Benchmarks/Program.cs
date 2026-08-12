using System.Diagnostics;
using My_DNN;
using My_DNN.Activation_functions;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Loss_functions;
using My_DNN.Optimizers;

namespace MDNN.Benchmarks
{
    // Profil REÁLNÉHO tréninkového kroku, ne izolovaných metod.
    //
    // Cíl (viz ROADMAP, Fáze 5): zjistit, kam se čas a paměť doopravdy podějí, než se
    // začne cokoli optimalizovat. Konkrétně jestli je úzkým hrdlem conv, a kolik sežerou
    // věci, které s paralelizací nesouvisí vůbec (alokace Tensorů, boxing v Convert.ToDouble,
    // kopírování aktivací).
    //
    // Měří se čas i ALOKACE. Alokace jsou tady stejně důležité jako čas: per-neuronový model
    // a Tensor vracený z každého getteru jsou hlavní podezřelí a v čase se projeví až nepřímo,
    // přes tlak na GC.
    internal static class Program
    {
        // Jedno měření: čas + kolik bajtů se při tom naalokovalo.
        private readonly record struct Cost(double Milliseconds, long Bytes)
        {
            public static Cost operator +(Cost a, Cost b)
                => new(a.Milliseconds + b.Milliseconds, a.Bytes + b.Bytes);
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

        private static void Table(string title, Dictionary<string, Cost> rows)
        {
            double totalMs = rows.Values.Sum(c => c.Milliseconds);
            long totalBytes = rows.Values.Sum(c => c.Bytes);

            Console.WriteLine();
            Console.WriteLine(title);
            Console.WriteLine($"  {"",-26} {"čas",13} {"podíl",7} {"alokace",11} {"podíl",7}");

            foreach ((string name, Cost cost) in rows.OrderByDescending(r => r.Value.Milliseconds))
            {
                double timeShare = totalMs > 0 ? cost.Milliseconds / totalMs * 100 : 0;
                double allocShare = totalBytes > 0 ? (double)cost.Bytes / totalBytes * 100 : 0;

                Console.WriteLine($"  {name,-26} {cost.Milliseconds,10:F1} ms {timeShare,5:F1} % "
                                + $"{Bytes(cost.Bytes),11} {allocShare,5:F1} %");
            }

            Console.WriteLine($"  {"CELKEM",-26} {totalMs,10:F1} ms {"",7} {Bytes(totalBytes),11}");
        }

        // Jméno vrstvy pro souhrn — u Dense/RNN i s počtem neuronů, ať jde rozlišit,
        // která vrstva je drahá.
        private static string LayerKey(Layer layer)
            => layer switch
            {
                Dense d => $"Dense({d.Neurons.Count})",
                RNN r => $"RNN({r.Neurons.Count})",
                Conv c => $"Conv({c.Kernel.Length}x{c.Kernel[0].Length})",
                _ => layer.Name,
            };

        // Projede `samples` vzorků jedním krokem tréninku a rozpadne čas i alokace
        // podle fáze a podle vrstvy.
        private static void ProfileTrainingStep(string title, My_DNN.MDNN model, double[][] inputs, double[][] targets)
        {
            List<Layer> layers = model.Layers.Layers;

            // Zahřátí: první průchod postaví vrstvy a nechá JIT zkompilovat cesty.
            model.GetResults(new Tensor(inputs[0]));
            model.Train.BackPropagation(new Tensor(targets[0]));
            model.Train.UpdateParams();

            var byPhase = new Dictionary<string, Cost>();
            var forwardByLayer = new Dictionary<string, Cost>();
            var backwardByLayer = new Dictionary<string, Cost>();

            for (int s = 0; s < inputs.Length; s++)
            {
                Tensor value = new Tensor(inputs[s]);

                // --- forward, vrstvu po vrstvě ---
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

                // --- řetěz gradientů (loss + CalculateLayerGradients přes vrstvy) ---
                Tensor[] gradients = Array.Empty<Tensor>();
                Add(byPhase, "2. řetěz gradientů",
                    Measure(() => gradients = Gradient.GetGradients(new Tensor(targets[s]), model)));

                // --- akumulace gradientů vah, vrstvu po vrstvě ---
                for (int i = 0; i < layers.Count; i++)
                {
                    Layer layer = layers[i];
                    Tensor gradient = gradients[i];

                    Cost cost = Measure(() => layer.BackPropagation(gradient));

                    Add(byPhase, "3. akumulace vah", cost);
                    Add(backwardByLayer, LayerKey(layer), cost);
                }
            }

            // --- update jednou za dávku ---
            foreach (Layer layer in layers)
            {
                Add(byPhase, "4. update parametrů", Measure(() => layer.UpdateParams()));
            }

            Console.WriteLine();
            Console.WriteLine(new string('=', 78));
            Console.WriteLine(title);
            Console.WriteLine(new string('=', 78));

            // Skutečná topologie, ne jen popisek v nadpisu. MDNN nemá vstupní vrstvu jako
            // objekt — velikost vstupu se odvodí z dat a projeví se až v počtu vah první
            // vrstvy, takže z kódu modelu ji přečíst nejde.
            Console.WriteLine($"  vstup: {inputs[0].Length} hodnot, {inputs.Length} vzorků");
            foreach (Layer layer in layers)
            {
                string shape = layer switch
                {
                    Dense d => $"{d.Neurons.Count} neuronů, {d.Neurons[0].Weights.Length} vah na neuron",
                    RNN r => $"{r.Neurons.Count} neuronů, {r.Neurons[0].Weights.Length} vah na neuron (vč. rekurentní)",
                    Conv c => $"{c.Kernel.Length} filtrů {c.Kernel[0].Length}x{c.Kernel[0][0].Length}x{c.Kernel[0][0][0].Length}"
                              + $", výstup [{string.Join(",", c.Output_size_and_shape)}]",
                    _ => $"výstup [{string.Join(",", layer.Output_size_and_shape)}]",
                };
                Console.WriteLine($"  {layer.Name,-8} {shape}");
            }
            Console.WriteLine($"  trénovatelných parametrů: {ConsoleControler.CountTrainableParams(model):N0}");

            Table("PODLE FÁZE", byPhase);
            Table("FORWARD podle vrstvy", forwardByLayer);
            Table("AKUMULACE VAH podle vrstvy", backwardByLayer);
        }

        // Podezřelí, kteří s paralelizací nesouvisí — čistá režie datové vrstvy.
        private static void ProfileOverhead()
        {
            const int repeats = 20_000;

            var model = new My_DNN.MDNN(new Dense(10, new Linear()), new SGD(0.01), new MSE(), seed: 1);
            model.Layers.Add(new Dense(128, new ReLu()));
            model.GetResults(new Tensor(new double[784]));

            Layer hidden = model.Layers.Layers[0];

            double[] flat = new double[784];
            var jagged = new double[64][];
            for (int i = 0; i < jagged.Length; i++) jagged[i] = new double[784];
            Tensor batch = Tensor.ConvertArrayToTensor(jagged);

            var rows = new Dictionary<string, Cost>();

            // zahřátí
            _ = hidden.Layer_output!.Data;
            _ = batch.GetTensorValue([0]);

            Add(rows, "Layer_output (getter)", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor? t = hidden.Layer_output; }
            }));

            Add(rows, "Layer_output + .Data", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { double[] d = hidden.Layer_output!.Data; }
            }));

            Add(rows, "new Tensor(double[784])", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor t = new Tensor(flat); }
            }));

            Add(rows, "Tensor.Data (flatten)", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { double[] d = new Tensor(flat).Data; }
            }));

            Add(rows, "GetTensorValue([i])", Measure(() =>
            {
                for (int i = 0; i < repeats; i++) { Tensor t = batch.GetTensorValue([i % 64]); }
            }));

            Console.WriteLine();
            Console.WriteLine(new string('=', 78));
            Console.WriteLine($"REŽIE DATOVÉ VRSTVY ({repeats:N0}x, mimo vlastní výpočet)");
            Console.WriteLine(new string('=', 78));
            Table($"náklady na {repeats:N0} volání", rows);

            Console.WriteLine();
            Console.WriteLine("  Pozn.: GetTensorValue([i]) volá TrainLoop na KAŽDÝ vzorek každé epochy,");
            Console.WriteLine("         Layer_output čte Gradient na každý krok backwardu.");
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
            Console.WriteLine($"MDNN profil — {Environment.ProcessorCount} jader, "
                            + $"{(Environment.Is64BitProcess ? "64" : "32")}bit, "
                            + $"server GC: {System.Runtime.GCSettings.IsServerGC}");
#if DEBUG
            Console.WriteLine("!! DEBUG build — čísla jsou k ničemu, pusť s -c Release");
#endif

            // --- MLP: 784 -> 128 -> 10 ---
            {
                var model = new My_DNN.MDNN(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy(), seed: 1);
                model.Layers.Add(new Dense(128, new ReLu()));

                var (inputs, targets) = MakeData(32, 784, 10, seed: 1);
                ProfileTrainingStep("MLP 784 -> 128 -> 10   (32 vzorků, jedna dávka)", model, inputs, targets);
            }

            // --- CNN jako v mdnn_test: 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10 ---
            {
                var model = new My_DNN.MDNN(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy(), seed: 1);
                model.Layers.Add(new Conv(16, 3, new ReLu(), "valid"));
                model.Layers.Add(new MaxPool(2));
                model.Layers.Add(new Conv(32, 3, new ReLu(), "valid"));
                model.Layers.Add(new MaxPool(2));
                model.Layers.Add(new Dense(64, new ReLu()));

                var (inputs, targets) = MakeData(8, 784, 10, seed: 2);
                ProfileTrainingStep("CNN 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10   (8 vzorků)",
                    model, inputs, targets);
            }

            ProfileOverhead();
        }
    }
}

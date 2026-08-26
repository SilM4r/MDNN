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
        // ==================================================================================
        //  ZÁKLAD (baseline) — proti čemu se dnešní běh porovnává
        // ==================================================================================
        //
        //  Naměřeno 2026-08-26, Apple Silicon / 10 jader, .NET 9, Release, workstation GC.
        //  Medián ze tří běhů. Profil níž tiskne u každého řádku, o kolik se čas a alokace
        //  posunuly proti těmhle číslům.
        //
        //  Absolutní hodnoty závisí na stroji — na jiném počítači bude celý sloupec
        //  "vs základ" posunutý a nic to neznamená. Smysl to dává jen při porovnání
        //  na TOMTÉŽ stroji. Když se změní scénář (jiná architektura, jiný počet vzorků),
        //  je potřeba základ přeměřit, jinak se porovnává neporovnatelné.
        //
        //  ⚠️ ALOKACE jsou deterministické (tři běhy dají bit po bitu totéž), ČASY kolísají
        //  mezi běhy až o 16 %. Sloupec "vs základ" u času má práh šumu 5 %, takže na něm
        //  uvidíš i výkyvy, které nic neznamenají. Věř spíš alokacím a velkým číslům.
        //
        // ----------------------------------------------------------------------------------
        //  HISTORIE — odkud se to kam posunulo
        // ----------------------------------------------------------------------------------
        //
        //  Předchozí základ: 2026-08-12, commit b0168bb, tentýž stroj. Byl to stav PŘED
        //  optimalizacemi datové vrstvy (Fáze 5) a před konvencí B. Hlavní zjištění tehdy:
        //  úzkým hrdlem není matematika, ale datová vrstva — Tensor.FlattenArray boxoval
        //  každý double (foreach nad System.Array) do List<double>, takže payload 6 KB
        //  stál ~40 KB alokací.
        //
        //   MLP 784-128-10, 32 vzorků            2026-08-12  →  2026-08-26
        //     fáze/1. forward                 73.5 ms /  1.5 MB  →  76.7 ms / 240 KB
        //     fáze/2. řetěz gradientů          2.1 ms / 184 KB   ┐ sloučeno konvencí B do
        //     fáze/3. akumulace vah           70.8 ms / 198 KB   ┘ "2. backward" 73.3 ms / 312 KB
        //     fáze/4. update parametrů         3.4 ms /     0 B  →   3.4 ms /     0 B
        //     forward/Dense(128)              72.4 ms /  1.3 MB  →  75.7 ms / 232 KB
        //     forward/Dense(10)                1.1 ms / 208 KB   →   1.0 ms /   9 KB
        //     akumulace/Dense(128)            69.8 ms / 198 KB   →  72.2 ms / 269 KB
        //     akumulace/Dense(10)              0.9 ms /     0 B  →   1.2 ms /  43 KB
        //
        //   CNN 28x28 -> conv16 -> pool -> conv32 -> pool -> dense64 -> dense10, 8 vzorků
        //     fáze/1. forward                 27.9 ms / 31.6 MB  →  10.0 ms / 1.2 MB
        //     fáze/2. řetěz gradientů         29.7 ms / 32.2 MB  ┐ sloučeno do
        //     fáze/3. akumulace vah            7.5 ms /     0 B  ┘ "2. backward" 19.8 ms / 2.8 MB
        //     fáze/4. update parametrů         3.6 ms /    80 B  →   3.7 ms /    80 B
        //     forward/Conv(16x3)              13.6 ms / 19.2 MB  →   2.8 ms / 726 KB
        //     forward/Conv(32x3)               9.2 ms /  6.3 MB  →   5.7 ms / 243 KB
        //     forward/MaxPool                  3.9 ms /  5.8 MB  →   0.7 ms / 221 KB
        //     forward/Dense(64)                1.1 ms / 332 KB   →   0.9 ms / 4.8 KB
        //     forward/Dense(10)                0.0 ms /  28 KB   →   0.0 ms / 2.2 KB
        //     akumulace/Conv(32x3)             4.8 ms /     0 B  →  13.9 ms / 384 KB
        //     akumulace/Conv(16x3)             1.6 ms /     0 B  →   3.6 ms / 302 KB
        //     akumulace/Dense(64)              1.0 ms /     0 B  →   1.1 ms /  60 KB
        //     akumulace/Dense(10)              0.0 ms /     0 B  →   0.0 ms / 6.8 KB
        //
        //   Režie datové vrstvy, 20 000 volání
        //     GetTensorValue([i])            464.0 ms / 963.6 MB →   6.0 ms / 122.1 MB
        //     Tensor.Data (flatten)          409.0 ms / 800.5 MB →   6.8 ms / 124.4 MB
        //     Layer_output + .Data           105.0 ms / 125.3 MB →   1.7 ms /  21.4 MB
        //     Layer_output (getter)            1.6 ms /   4.3 MB →   1.9 ms /  21.4 MB
        //     new Tensor(double[784])          1.5 ms /   4.3 MB →   1.5 ms /   4.3 MB
        //
        //  Jak číst dva řádky, které vypadají jako zhoršení:
        //
        //  1. "Layer_output (getter)" zdražil, protože se do něj PŘESUNULA kopie výstupu.
        //     Dřív vracel okno do pracovního pole vrstvy a kopie se dělala až ve .Data —
        //     jenže výsledek se do té doby stihl přepsat dalším průchodem. Dvojici je
        //     potřeba číst dohromady: getter + .Data spadl ze 105 ms na 1.7 ms.
        //  2. "akumulace/Conv" a "akumulace/Dense" mají nově nenulové alokace i vyšší časy,
        //     protože konvence B přesunula výpočet ∂L/∂input z fáze "řetěz gradientů"
        //     do backwardu vrstvy. Práce nepřibyla, jen se účtuje jinam — dřívější
        //     "řetěz gradientů" (29.7 ms / 32.2 MB u CNN) v tabulce už neexistuje.
        //
        //  Celkově proti srpnu: CNN forward -64 % času a -96 % alokací, režie datové
        //  vrstvy -98 % / -85 %. MLP časy stojí — ten scénář je z 98 % v per-neuronových
        //  smyčkách Dense, na které míří teprve FastDense (viz ROADMAP).
        // ==================================================================================
        private static readonly Dictionary<string, Cost> Baseline = new()
        {
            // MLP 784-128-10, 32 vzorků
            ["MLP/fáze/1. forward"] = new(76.7, 246_272),
            ["MLP/fáze/2. backward"] = new(73.3, 319_744),
            ["MLP/fáze/4. update parametrů"] = new(3.4, 0),
            ["MLP/forward/Dense(128)"] = new(75.7, 237_312),
            ["MLP/forward/Dense(10)"] = new(1.0, 8_960),
            ["MLP/akumulace/Dense(128)"] = new(72.2, 275_712),
            ["MLP/akumulace/Dense(10)"] = new(1.2, 44_032),

            // CNN, 8 vzorků
            ["CNN/fáze/1. forward"] = new(10.0, 1_225_088),
            ["CNN/fáze/2. backward"] = new(19.8, 2_918_656),
            ["CNN/fáze/4. update parametrů"] = new(3.7, 80),
            ["CNN/forward/Conv(16x3)"] = new(2.8, 743_424),
            ["CNN/forward/Conv(32x3)"] = new(5.7, 248_640),
            ["CNN/forward/MaxPool"] = new(0.7, 225_920),
            ["CNN/forward/Dense(64)"] = new(0.9, 4_864),
            ["CNN/forward/Dense(10)"] = new(0.0, 2_240),
            ["CNN/akumulace/Conv(32x3)"] = new(13.9, 393_728),
            ["CNN/akumulace/Conv(16x3)"] = new(3.6, 309_248),
            ["CNN/akumulace/MaxPool"] = new(0.9, 2_147_008),
            ["CNN/akumulace/Dense(64)"] = new(1.1, 61_760),
            ["CNN/akumulace/Dense(10)"] = new(0.0, 6_912),

            // režie datové vrstvy, 20 000 volání
            ["režie/GetTensorValue([i])"] = new(6.0, 128_000_000),
            ["režie/Tensor.Data (flatten)"] = new(6.8, 130_400_000),
            ["režie/Layer_output + .Data"] = new(1.7, 22_400_000),
            ["režie/Layer_output (getter)"] = new(1.9, 22_400_000),
            ["režie/new Tensor(double[784])"] = new(1.5, 4_480_000),
        };

        // Jedno měření: čas + kolik bajtů se při tom naalokovalo.
        private readonly record struct Cost(double Milliseconds, long Bytes)
        {
            public static Cost operator +(Cost a, Cost b)
                => new(a.Milliseconds + b.Milliseconds, a.Bytes + b.Bytes);
        }

        // ∂L/∂output výstupní vrstvy — vstup do Backward řetězu. Totéž, co si uvnitř
        // počítá Gradient.RunBackward; tady zvlášť, aby šel změřit jen samotný backward.
        private static Tensor OutputGradient(My_DNN.MDNN model, Tensor target)
        {
            double[] output = model.Layers.Layers[^1].Layer_output!.Data;
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

        // Změna proti základu. Kladné číslo = zhoršení, záporné = zlepšení.
        // Prázdno, když pro daný řádek základ není (nový scénář nebo nová vrstva).
        private static string VsBaseline(double now, double before)
        {
            if (before <= 0)
            {
                return now <= 0 ? "     ." : "   nový";
            }

            double change = (now - before) / before * 100;

            // Pod 5 % už je to na téhle úrovni měření šum, ne výsledek.
            return Math.Abs(change) < 5 ? "     ~" : $"{change,+6:F0}%";
        }

        private static void Table(string title, Dictionary<string, Cost> rows, string baselinePrefix)
        {
            double totalMs = rows.Values.Sum(c => c.Milliseconds);
            long totalBytes = rows.Values.Sum(c => c.Bytes);

            Console.WriteLine();
            Console.WriteLine(title);
            Console.WriteLine($"  {"",-26} {"čas",13} {"podíl",7} {"vs zákl",8} "
                            + $"{"alokace",11} {"podíl",7} {"vs zákl",8}");

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

            Console.WriteLine($"  {"CELKEM",-26} {totalMs,10:F1} ms {"",7} "
                            + $"{VsBaseline(totalMs, baselineMs),8} "
                            + $"{Bytes(totalBytes),11} {"",7} "
                            + $"{VsBaseline(totalBytes, baselineBytes),8}");
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
        private static void ProfileTrainingStep(string title, string baselinePrefix, My_DNN.MDNN model, double[][] inputs, double[][] targets)
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

                // --- backward, vrstvu po vrstvě ---
                // Konvence B: jeden průchod. Dřív se měřily dvě fáze (řetěz gradientů +
                // akumulace vah), protože backward byl rozdělený na dvě metody. Teď dělá
                // každá vrstva obojí naráz, takže je to jedna fáze na vrstvu.
                Tensor grad = OutputGradient(model, new Tensor(targets[s]));

                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    Layer layer = layers[i];
                    bool fused = i == layers.Count - 1
                                 && model.Context.Loss.RequiresSoftmax
                                 && layer.Activation_Func.Apply_to_layer;

                    Tensor produced = grad;
                    Cost cost = Measure(() => produced = layer.Backward(grad, fused));
                    grad = produced;

                    Add(byPhase, "2. backward", cost);
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

            Table("PODLE FÁZE", byPhase, baselinePrefix + "/fáze/");
            Table("FORWARD podle vrstvy", forwardByLayer, baselinePrefix + "/forward/");
            Table("AKUMULACE VAH podle vrstvy", backwardByLayer, baselinePrefix + "/akumulace/");
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
            Table($"náklady na {repeats:N0} volání", rows, "režie/");

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
                ProfileTrainingStep("MLP 784 -> 128 -> 10   (32 vzorků, jedna dávka)", "MLP", model, inputs, targets);
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
                    "CNN", model, inputs, targets);
            }

            ProfileOverhead();
        }
    }
}

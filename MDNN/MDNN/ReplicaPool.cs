using System.Runtime.ExceptionServices;
using My_DNN.Layers.classes;

namespace My_DNN
{
    // Data-paralelní trénink jedné dávky přes repliky modelu (Train.Workers > 1).
    //
    // Jedna dávka:
    //   1. originál přenese do replik parametry, které nesdílí odkazem (biasy neuronů)
    //   2. dávka se rozdělí na souvislé úseky a každá replika spočítá svůj na vlastním vlákně
    //      (forward + backward, gradienty se sčítají v replice)
    //   3. gradienty a loss se přičtou k originálu V PEVNÉM POŘADÍ replik
    //   4. UpdateParams originálu (to už dělá volající, jedním vláknem jako dřív)
    //
    // Proč repliky a ne zámky nebo stav po vláknech: vrstvy drží aktivace v polích
    // (`output`, `raw_output`, `Neuron.inputs`, …), takže vrstvou smí v jednu chvíli
    // procházet jen jeden vzorek. Replika dá každému vláknu vlastní sadu těch polí,
    // aniž by se vnitřek vrstev musel měnit. Váhy se přitom nekopírují — replika je
    // sdílí s originálem a jen je čte.
    //
    // Determinismus: sčítání double není asociativní, takže výsledek závisí na tom, jak
    // se dávka rozdělí. Rozdělení i pořadí sčítání jsou ale dané jen počtem replik, ne
    // tím, které vlákno doběhne dřív — stejný seed a stejné Workers dají bit po bitu
    // stejný výsledek, na jakémkoli stroji.
    internal sealed class ReplicaPool
    {
        private readonly MDNN _original;
        private readonly MDNN[] _replicas;

        public ReplicaPool(MDNN original, int workers)
        {
            if (original.Context.CalculationViaGpu)
            {
                throw new NotSupportedException(
                    "Paralelní trénink (Train.Workers > 1) zatím nejde kombinovat s výpočtem na GPU. " +
                    "Nastav Train.Workers = 1, nebo vypni CalculationViaGpu.");
            }

            // RNN mimo sekvenční trénink nese skrytý stav z jednoho vzorku do DALŠÍHO
            // (nic ho mezi vzorky nenuluje). Sekvenčně teče stav v pořadí dávky; paralelně
            // by si ho každá replika nesla po svém kusu a výsledek by se tiše lišil.
            // V sekvenčním režimu to nevadí — replika bere celou sekvenci a stav se nuluje.
            if (!original.Context.SequenceTrain && original.Layers.Layers.Any(l => l is Layers.RNN))
            {
                throw new NotSupportedException(
                    "RNN vrstva mimo sekvenční trénink nese skrytý stav mezi vzorky, takže ji nejde " +
                    "trénovat paralelně (Train.Workers > 1). Použij sekvenční trénink " +
                    "(TrainLoop(..., isSequence: true)), nebo Train.Workers = 1.");
            }

            _original = original;
            _replicas = new MDNN[workers];

            // Náhodu replika potřebuje jen na masky dropoutu (parametry už jsou vylosované).
            // Bez dropoutu se z generátoru originálu NEČERPÁ — každé čerpání by posunulo
            // jeho stav a další epocha by zamíchala data jinak než při Workers = 1. Zapnout
            // paralelismus by pak změnilo trénink víc než o zaokrouhlení.
            bool needsRandom = original.Layers.Layers.Any(l => l is Layers.Dense d && d.UsesDropout);

            for (int w = 0; w < workers; w++)
            {
                var layers = new List<Layer>();
                foreach (Layer layer in original.Layers.Layers)
                {
                    layers.Add(layer.CreateReplica() ?? throw new NotSupportedException(
                        $"Vrstva {layer.GetType().Name} nepodporuje paralelní trénink " +
                        "(Train.Workers > 1). Nastav Train.Workers = 1."));
                }

                // Seed pro generátor repliky se bere z generátoru originálu: je to pořád
                // jediný zdroj náhody modelu, takže se seedem reprodukovatelné. Bez dropoutu
                // dostane replika pevný seed — vlastní instance tak jako tak, protože
                // kontext bez generátoru by sáhl po SDÍLENÉM globálním, a ten thread-safe není.
                int seed = needsRandom ? original.Context.Random.Next() : 0;
                NetworkContext context = original.Context.CreateReplica(seed);
                _replicas[w] = new MDNN(context, layers);
            }
        }

        // Spočítá vzorky order[start..end) a nasčítá jejich gradienty do originálu.
        // `fitOne` dostane Train repliky a index vzorku — tentýž kód, jaký se volá
        // v sekvenční cestě nad originálem.
        public void RunBatch(int[] order, int start, int end, Action<Train, int> fitOne)
        {
            int count = end - start;
            int used = Math.Min(_replicas.Length, count);
            List<Layer> originalLayers = _original.Layers.Layers;

            for (int w = 0; w < used; w++)
            {
                List<Layer> replicaLayers = _replicas[w].Layers.Layers;
                for (int i = 0; i < originalLayers.Count; i++)
                {
                    originalLayers[i].SyncReplica(replicaLayers[i]);
                }
            }

            try
            {
                Parallel.For(0, used, new ParallelOptions { MaxDegreeOfParallelism = used }, w =>
                {
                    int from = start + count * w / used;
                    int to = start + count * (w + 1) / used;

                    for (int k = from; k < to; k++)
                    {
                        fitOne(_replicas[w].Train, order[k]);
                    }
                });
            }
            catch (AggregateException e)
            {
                // Volající má dostat tutéž výjimku jako v sekvenční cestě (třeba
                // ArgumentException o nesedícím počtu cílů), ne obal z Parallel.For.
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
                throw;
            }

            // Pevné pořadí 0, 1, 2, … — ne podle toho, kdo doběhne dřív.
            for (int w = 0; w < used; w++)
            {
                List<Layer> replicaLayers = _replicas[w].Layers.Layers;
                for (int i = 0; i < originalLayers.Count; i++)
                {
                    originalLayers[i].TakeGradientsFrom(replicaLayers[i]);
                }

                _original.Context.Loss.TakeAccumulatedLossFrom(_replicas[w].Context.Loss);
            }
        }
    }
}

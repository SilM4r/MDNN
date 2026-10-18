using System;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// Stav optimizeru (momenty Adamu, rychlost Momenta) patří k vahám — dokud váhy přežijí,
/// musí přežít i on.
///
/// Dřív `MaterializeParameters` klonovalo optimizer z Contextu VŽDYCKY, i když se neurony
/// vůbec nepřestavovaly. A `Clone_optimizer` staví optimizer znovu ze jména a hyperparametrů,
/// takže `m`, `v`, `t` šly do koše. Spouštěče byly dva:
///
///   1. `SetInputSizeForFirstLayer()` — a `SimpleTrainLoop` ho volá při KAŽDÉM běhu
///   2. `Add`/`Insert`/`RemoveAt` — `MaterializeAll()` projde všechny vrstvy, takže se stav
///      ztratil i vrstvám, kterých se přepojení vůbec netýkalo
///
/// Druhý bod bolí u AutoML, které architekturu upravuje inkrementálně: každý krok tiše
/// zahodil rozjeté momenty celého modelu.
/// </summary>
public class OptimizerStateTests
{
    private static readonly Tensor X = new Tensor(new double[] { 1.0, 0.5 });
    private static readonly Tensor T = new Tensor(new double[] { 1.0 });

    private static void Step(My_DNN.MDNN model)
    {
        model.Train.Fit(X, T);
        model.Train.UpdateParams();
    }

    /// <summary>Model s rozjetým Adamem — po pěti krocích už momenty nejsou triviální.</summary>
    private static My_DNN.MDNN Warm(int hidden = 0)
    {
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        if (hidden > 0) model.Layers.Add(new Dense(hidden, new ReLu()));

        for (int i = 0; i < 5; i++) Step(model);
        return model;
    }

    private static double Weight(My_DNN.MDNN model, int layer = 0)
        => ((Dense)model.Layers.Layers[layer]).Neurons[0].Weights[0];

    private static Optimizer OptimizerOf(My_DNN.MDNN model, int layer)
        => ((Dense)model.Layers.Layers[layer]).Neurons[0].optimizer;

    // ------------------------------------------------------------ přepojení, které nic nemění

    [Fact]
    public void Rewiring_that_changes_nothing_keeps_optimizer_state()
    {
        var reference = Warm();
        Step(reference);

        var rewired = Warm();
        rewired.Layers.SetInputSizeForFirstLayer();   // tvar se nemění, jen se „přepojí"
        Step(rewired);

        Assert.Equal(Weight(reference), Weight(rewired));
    }

    [Fact]
    public void Rewiring_that_changes_nothing_keeps_the_same_optimizer_instance()
    {
        var model = Warm();
        Optimizer before = OptimizerOf(model, 0);

        model.Layers.SetInputSizeForFirstLayer();

        Assert.Same(before, OptimizerOf(model, 0));
    }

    // ------------------------------------------------------------ změna architektury

    [Fact]
    public void Adding_a_layer_keeps_the_optimizer_of_untouched_layers()
    {
        // Vrstvy: [Dense(3) skrytá, Dense(1) výstupní]. Po Add(Dense(4)) se vloží doprostřed,
        // takže VÝSTUPNÍ vrstva dostane jiný počet vstupů (3 → 4) a musí se přestavět,
        // kdežto ta první se nezmění vůbec.
        var model = Warm(hidden: 3);
        Optimizer untouched = OptimizerOf(model, 0);
        Optimizer rebuilt = OptimizerOf(model, 1);

        model.Layers.Add(new Dense(4, new ReLu()));
        model.GetResults(X);   // spustí MaterializeAll

        Assert.Same(untouched, OptimizerOf(model, 0));
        Assert.NotSame(rebuilt, OptimizerOf(model, 2));   // výstupní vrstva je teď index 2
    }

    [Fact]
    public void Removing_a_layer_keeps_the_optimizer_of_untouched_layers()
    {
        var model = Warm(hidden: 3);
        model.Layers.Add(new Dense(4, new ReLu()));
        model.GetResults(X);

        Optimizer untouched = OptimizerOf(model, 0);

        model.Layers.RemoveAt(1);
        model.GetResults(X);

        Assert.Same(untouched, OptimizerOf(model, 0));
    }

    // ------------------------------------------------------------ celý trénink

    [Fact]
    public void SimpleTrainLoop_in_two_halves_matches_one_run()
    {
        // `SimpleTrainLoop` volá SetInputSizeForFirstLayer() bezpodmínečně při každém běhu,
        // takže druhá půlka dřív startovala s vynulovaným Adamem.
        double[][] x = Enumerable.Range(1, 8).Select(i => new[] { i / 10.0 }).ToArray();
        double[][] y = x.Select(r => new[] { r[0] * 2 }).ToArray();

        My_DNN.MDNN Build()
        {
            var m = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.05), new MSE(), seed: 3);
            m.Train.AutoSaveInTrainLoop = false;
            return m;
        }

        var oneGo = Build();
        var inHalves = Build();

        var original = Console.Out;
        Console.SetOut(System.IO.TextWriter.Null);
        try
        {
            oneGo.Train.SimpleTrainLoop(x, y, 10, 4);

            inHalves.Train.SimpleTrainLoop(x, y, 5, 4);
            inHalves.Train.SimpleTrainLoop(x, y, 5, 4);
        }
        finally
        {
            Console.SetOut(original);
        }

        Assert.Equal(10u, oneGo.Train.CurrentEpoch);
        Assert.Equal(10u, inHalves.Train.CurrentEpoch);
        Assert.Equal(Weight(oneGo), Weight(inHalves));
    }

    // ------------------------------------------------------------ pojistky, ať se to nepřežene

    [Fact]
    public void Freshly_built_neurons_use_the_models_optimizer_not_the_global_default()
    {
        // Neuron si v konstruktoru bere klon GLOBÁLNÍHO optimizeru (SGD s lr 0.0001).
        // Materializace ho musí přepsat tím z Contextu — jinak by model s Adamem
        // trénoval krokem 0.0001 a vypadalo by to jako „model se neučí".
        // Skrytá vrstva má Tanh, ne ReLu: při téhle inicializaci vydá ReLu samé nuly,
        // takže by gradient výstupních vah byl nulový a Adam by se nehnul ani správně.
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        model.Layers.Add(new Dense(3, new Tanh()));
        model.GetResults(X);                // až tady se vrstvy materializují

        double before = Weight(model, 1);   // výstupní vrstva
        Step(model);
        double moved = Math.Abs(Weight(model, 1) - before);

        // První krok Adamu je zhruba lr (m̂/(√v̂+ε) ≈ ±1); globální SGD by dal ~1e-4.
        Assert.True(moved > 0.05, $"váha se posunula jen o {moved:G6} — vypadá to na globální SGD, ne na Adam z modelu");
    }

    [Fact]
    public void Two_models_with_adam_do_not_share_momentum()
    {
        // Nezávislost modelů z Fáze 3: každý neuron musí mít VLASTNÍ instanci optimizeru.
        var a = Warm();
        var b = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        b.GetResults(X);   // materializace

        Assert.NotSame(OptimizerOf(a, 0), OptimizerOf(b, 0));

        // `b` je čerstvý, takže jeho první krok musí být plný krok Adamu — nezávisle
        // na tom, že `a` má za sebou pět kroků.
        double before = Weight(b);
        Step(b);

        Assert.True(Math.Abs(Weight(b) - before) > 0.05);
    }

    // ------------------------------------------------------------ EnsureCapacity

    /// <summary>Délka pole se stavem optimizeru (Adam.m, Momentum.velocity).</summary>
    /// Reflexe na privátní pole je záměr: velikost stavu není chování, které by šlo
    /// pozorovat zvenku, a přidávat kvůli testu veřejnou vlastnost nechci.
    private static int StateLength(Optimizer optimizer)
    {
        string field = optimizer is Adam ? "m" : "velocity";
        var info = optimizer.GetType().GetField(field,
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return ((double[])info.GetValue(optimizer)!).Length;
    }

    public static TheoryData<Optimizer> StatefulOptimizers()
        => new() { new Adam(0.1), new Momentum(0.1, 0.9) };

    [Theory]
    [MemberData(nameof(StatefulOptimizers))]
    public void EnsureCapacity_keeps_state_that_is_already_accumulated(Optimizer optimizer)
    {
        // Zvětšení pole uprostřed tréninku nesmí vynulovat momenty už rozjetých parametrů.
        Optimizer reference = Optimizer.Clone_optimizer(optimizer);

        double a = 1.0, b = 1.0;
        for (int step = 0; step < 3; step++)
        {
            a = optimizer.Update(a, 0.5, 0);
            b = reference.Update(b, 0.5, 0);
        }

        optimizer.EnsureCapacity(100);
        optimizer.EnsureCapacity(1);   // menší požadavek nesmí pole zmenšit

        Assert.Equal(reference.Update(b, 0.5, 0), optimizer.Update(a, 0.5, 0));
        Assert.Equal(100, StateLength(optimizer));
    }

    [Fact]
    public void Dense_sizes_optimizer_state_exactly_to_weights_plus_bias()
    {
        // Hlídá, aby se nevrátilo zdvojnásobování kapacity (commit f6b8e05): 785 vah
        // nafouklo stav na 1024 prvků a ~3 % navíc stál každý UpdateParams.
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.1), new MSE(), seed: 11);
        Step(model);

        Assert.Equal(X.Data.Length + 1, StateLength(OptimizerOf(model, 0)));
    }

    [Fact]
    public void Conv_sizes_optimizer_state_exactly_to_kernels_plus_biases()
    {
        var model = new My_DNN.MDNN(new Conv(2, 2, new Linear(), "valid"), new Adam(0.1), new MSE(), seed: 1);
        var image = new double[4, 4, 1];
        for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            image[i, j, 0] = 0.1 * (i + j);

        model.Train.Fit(new Tensor(image), new Tensor(new double[3 * 3 * 2]));
        model.Train.UpdateParams();

        var conv = model.Layers.Layers[0];
        var optimizer = (Optimizer)conv.GetType().GetField("optimizer",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(conv)!;

        // 2 filtry × (kernel 2×2×1 + bias)
        Assert.Equal(2 * (2 * 2 * 1 + 1), StateLength(optimizer));
    }
}

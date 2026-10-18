using System;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;

namespace MDNN.Tests;

/// <summary>
/// Vlastní optimizer zděděný z `Optimizer` musí jít použít, jak slibuje README.
///
/// Dřív se optimizer pro každý neuron klonoval podle jména a `switch` znal jen SGD,
/// Adam a Momentum — vlastní optimizer spadl na „this optimizer does not exist" hned
/// při sestavení první vrstvy. A výjimka nechala vrstvu napůl přestavěnou: neurony už
/// byly nové, s GLOBÁLNÍM SGD(0.0001) z konstruktoru, takže další volání prošlo a model
/// tiše trénoval úplně jiným optimizerem.
/// </summary>
public class CustomOptimizerTests
{
    /// <summary>Krok o pevnou délku proti znaménku gradientu — jiný výsledek než SGD/Adam.</summary>
    private sealed class SignStep : Optimizer
    {
        private readonly double step;
        public SignStep(double step) { this.step = step; }

        public override string Name => "SignStep";
        public override double[] Hyperparameters => new[] { step };
        public override Optimizer CreateFresh() => new SignStep(step);
        public override double Update(double w, double gradient, int paramIndex) => w - step * Math.Sign(gradient);
    }

    /// <summary>Optimizer, jehož klonování selže — simuluje chybu při sestavení vrstvy.</summary>
    private sealed class BrokenClone : Optimizer
    {
        public override string Name => "BrokenClone";
        public override double[] Hyperparameters => Array.Empty<double>();
        public override Optimizer CreateFresh() => throw new InvalidOperationException("clone failed");
        public override double Update(double w, double gradient, int paramIndex) => w;
    }

    private static readonly Tensor X = new Tensor(new double[] { 1.0, 0.5 });
    private static readonly Tensor T = new Tensor(new double[] { 1.0 });

    private static double[,,] Image()
    {
        var img = new double[4, 4, 1];
        for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            img[i, j, 0] = 0.1 * (i + j);
        return img;
    }

    private static Optimizer ConvOptimizer(Layer conv)
        => (Optimizer)conv.GetType().GetField("optimizer",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(conv)!;

    [Fact]
    public void Dense_trains_with_a_custom_optimizer()
    {
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SignStep(0.25), new MSE(), seed: 3);
        model.GetResults(X);
        var neuron = ((Dense)model.Layers.Layers[0]).Neurons[0];
        double before = neuron.Weights[0];

        model.Train.Fit(X, T);
        model.Train.UpdateParams();

        Assert.IsType<SignStep>(neuron.Optimizer);
        Assert.Equal(0.25, Math.Abs(neuron.Weights[0] - before), 12);   // krok přesně 0.25, ne SGD
    }

    [Fact]
    public void Each_neuron_gets_its_own_instance_of_the_custom_optimizer()
    {
        var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SignStep(0.25), new MSE(), seed: 3);
        model.GetResults(X);
        var neurons = ((Dense)model.Layers.Layers[0]).Neurons;

        Assert.NotSame(neurons[0].Optimizer, neurons[1].Optimizer);
        Assert.NotSame(model.Optimizer, neurons[0].Optimizer);
    }

    [Fact]
    public void Conv_trains_with_a_custom_optimizer()
    {
        var model = new My_DNN.MDNN(new Conv(2, 2, new Linear(), "valid"), new SignStep(0.25), new MSE(), seed: 1);

        model.Train.Fit(new Tensor(Image()), new Tensor(new double[3 * 3 * 2]));
        model.Train.UpdateParams();

        Assert.IsType<SignStep>(ConvOptimizer(model.Layers.Layers[0]));
    }

    public static TheoryData<Optimizer> BuiltIn()
        => new() { new SGD(0.1), new Adam(0.1, 0.8, 0.95), new Momentum(0.1, 0.7) };

    [Theory]
    [MemberData(nameof(BuiltIn))]
    public void CreateFresh_keeps_hyperparameters_and_drops_state(Optimizer optimizer)
    {
        Optimizer pristine = optimizer.CreateFresh();
        for (int step = 0; step < 3; step++) optimizer.Update(1.0, 0.5, 0);   // rozjet stav

        Optimizer fresh = optimizer.CreateFresh();

        Assert.NotSame(optimizer, fresh);
        Assert.Equal(optimizer.GetType(), fresh.GetType());
        Assert.Equal(optimizer.Hyperparameters, fresh.Hyperparameters);
        // první krok čerstvého = první krok nedotčeného, tedy žádný zděděný moment
        Assert.Equal(pristine.Update(1.0, 0.5, 0), fresh.Update(1.0, 0.5, 0));
    }

    [Fact]
    public void Failed_dense_build_does_not_leave_the_layer_half_built()
    {
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new BrokenClone(), new MSE(), seed: 3);

        Assert.Throws<InvalidOperationException>(() => model.GetResults(X));
        // Dřív tady druhé volání prošlo a model trénoval globálním SGD(0.0001).
        Assert.Throws<InvalidOperationException>(() => model.Train.Fit(X, T));
        Assert.False(model.Layers.Layers[0].IsMaterialized);   // zůstaly zástupné neurony z konstruktoru
    }

    [Fact]
    public void Failed_conv_build_does_not_leave_the_layer_half_built()
    {
        var model = new My_DNN.MDNN(new Conv(2, 2, new Linear(), "valid"), new BrokenClone(), new MSE(), seed: 1);

        Assert.Throws<InvalidOperationException>(() => model.GetResults(new Tensor(Image())));
        Assert.Throws<InvalidOperationException>(() => model.GetResults(new Tensor(Image())));
    }
}

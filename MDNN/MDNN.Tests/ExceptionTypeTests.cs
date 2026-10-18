using System;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// Chybové stavy hlásí konkrétní typ výjimky, ne obecnou <see cref="Exception"/>. Volající
/// tak může chytit jen to, co umí ošetřit (špatný argument × špatný stav × nepodporované),
/// a kód, který chytá <c>Exception</c>, funguje beze změny dál.
/// </summary>
public class ExceptionTypeTests
{
    private static My_DNN.MDNN Model()
    {
        var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.1), new MSE(), seed: 1);
        model.Train.ShowLossChartInTrainLoop = false;
        model.Train.ShowModelInfoIntrainLoop = false;
        model.Train.AutoSaveInTrainLoop = false;
        return model;
    }

    [Fact]
    public void Layer_position_out_of_range()
    {
        var model = Model();
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.Insert(5, new Dense(2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Layers.RemoveAt(3));
    }

    [Fact]
    public void Inputs_and_targets_with_different_sample_counts()
    {
        // Pozn.: tenhle případ odmítne už LabeledData (vlastní ArgumentException), takže se
        // k převedené kontrole v Train.CheckTensorShapes nedostane — test hlídá výsledný typ
        // výjimky, ne ten konkrétní řádek. Prošel by i před převodem.
        var model = Model();
        Assert.Throws<ArgumentException>(() =>
            model.Train.TrainLoop(new double[,] { { 1 }, { 2 }, { 3 } }, new double[,] { { 1 }, { 2 } }, 1, 1));
    }

    [Fact]
    public void Invalid_split_ratio_is_a_configuration_error()
    {
        var model = Model();
        model.Train.TrainSplitRatio = 1.5;
        var x = new double[20, 1];
        var y = new double[20, 1];

        Assert.Throws<InvalidOperationException>(() => model.Train.TrainLoop(x, y, 1, 1));
    }

    private sealed class Identity : LayerWithUntrainedParameters
    {
        private int[] shape = { 0 };
        public override string Name => "Identity";
        public override int[] Input_size_and_shape => shape;
        public override int[] Output_size_and_shape => shape;
        public override Tensor? Layer_output => null;
        public override Tensor? Layer_raw_output => null;
        public override Tensor FeedForward(Tensor input_values) => input_values;
        public override void WireShapes(int? n = null, int[]? input = null) { if (input != null) shape = input; }
        public override void MaterializeParameters() { }
        public override bool IsMaterialized => true;
        public override Tensor Backward(Tensor g, bool p = false, bool c = true) => g;
    }

    [Fact]
    public void Saving_a_custom_layer_is_not_supported()
    {
        var model = Model();
        model.Layers.Insert(0, new Identity());
        model.GetResults(new Tensor(new double[] { 1.0 }));

        var ex = Assert.Throws<NotSupportedException>(() => model.SaveAsJsonString());
        Assert.Contains("Identity", ex.Message);
    }
}

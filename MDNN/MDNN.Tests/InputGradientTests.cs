using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Activation_functions;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;

namespace MDNN.Tests;

/// <summary>
/// <c>Backward(gradOutput) → gradInput</c> — že vrstva vrátí správný gradient vůči svému
/// VSTUPU, ověřeno numericky proti perturbaci vstupu.
///
/// Je to jediné místo, kde se <c>Backward</c> řetězí RUČNĚ, mimo <c>Gradient.RunBackward</c>.
/// Vzhledem k tomu, že <c>Layer.Backward</c> je veřejné API, má smysl to držet.
///
/// ---------------------------------------------------------------------------------------
/// Co tenhle soubor NAHRAZUJE — poučení, které stojí za zapamatování
/// ---------------------------------------------------------------------------------------
/// Byl to `BackwardConventionTests` a měl 16 testů „bit-identita staré a nové cesty".
/// Ty vznikly kvůli přechodu na konvenci B a svou práci odvedly: hlídaly, že nový
/// `Backward` dá BITOVĚ stejné gradienty jako tehdejší trojice CalculateLayerGradients
/// + SeedOutputGradient + BackPropagation.
///
/// Jenže krok 3 toho refaktoru (b2b02fb) starou cestu SMAZAL — a tím se z těch testů
/// stalo porovnávání kódu se sebou samým: `oldModel.Train.Fit(...)` i ruční řetěz volaly
/// odteď tentýž `Backward`. Testy zůstaly zelené, jméno i dokumentace dál slibovaly
/// pojistku, a nikdo si toho nevšiml.
///
/// Doloženo mutační zkouškou (2026-08-26) — do backwardu se zanesla chyba a sledovalo se,
/// kdo ji chytí:
///
///   mutace                                  BackwardConventionTests   zbytek sady
///   Dense: dropout maska v backwardu pryč            2 padly            2 padly
///   fúze softmax+CE vypnutá                          1 padl             3 padly
///   Conv: chybí act' v backwardu                     0                 20 padlo
///   RNN: act' navíc v akumulaci                      0                  3 padly
///   Dense: chybí SkipSample u dropoutu               0                  1 padl
///
/// Nechytaly tedy NIC navíc a tři chyby z pěti propásly — protože mutace zasáhla obě
/// strany porovnání stejně. To je horší než nulová hodnota: sada, která se tváří jako
/// pojistka a neochrání nic.
///
/// **Poučení:** přechodové testy „staré vs. nové" mají smysl jen dokud stará cesta žije.
/// Až se bude stejným třífázovým postupem (přidat vedle → přepnout → smazat) dělat
/// activation context, musí se s tou starou cestou smazat i ony.
/// </summary>
public class InputGradientTests
{
    /// <summary>∂L/∂output výstupní vrstvy — totéž, co počítá Gradient.RunBackward
    /// před předáním do prvního Backwardu.</summary>
    private static double[] OutputGradient(My_DNN.MDNN model, Tensor output, Tensor target)
    {
        double[] d = new double[output.Data.Length];
        for (int i = 0; i < d.Length; i++)
            d[i] = model.Context.Loss.DerivativeOfLossFunction(output.Data[i], target.Data[i]);
        return d;
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]      // POVINNĚ i s dropoutem: jinak není žádná delta přesně nulová
    [InlineData(0.8)]      // a větev pro vypnutý neuron se v gradInput nikdy neprojde
    public void Backward_returns_gradient_with_respect_to_input(double rate)
    {
        // ∂L/∂input je to, co konvence B navíc VRACÍ — dřív si ho vrstva pod Dense
        // musela dopočítat sama z cizích vah. Ověřeno numericky proti perturbaci vstupu.
        //
        // Proč i s dropoutem: negativní kontrola ukázala, že varianta jen bez dropoutu
        // NEODHALÍ chybu ve zpracování nulových delt (záměna continue/break v akumulaci
        // gradInput prošla). Bez vypnutých neuronů se ta větev prostě nikdy nevykoná.
        var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
        var target = new Tensor(new double[] { 0.2, -0.4, 0.7 });

        var model = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE(), seed: 21);
        model.Layers.Add(new Dense(9, new Tanh(), dropout: rate));

        // Zmrazená maska: numerická derivace musí měřit pokaždé TUTÉŽ funkci.
        model.Context.IsTraining = true;
        model.GetResults(x);
        foreach (Dense d in model.Layers.Layers.Cast<Dense>())
            d.FreezeDropoutMaskForTests = true;

        if (rate > 0)
        {
            double[] mask = ((Dense)model.Layers.Layers[0]).DropoutMaskForTests!;
            Assert.Contains(mask, m => m == 0.0);   // pojistka: opravdu je co vypnout
        }

        Tensor output = model.GetResults(x);

        var layers = model.Layers.Layers;
        var grad = new Tensor(OutputGradient(model, output, target));
        for (int i = layers.Count - 1; i >= 0; i--)
            grad = layers[i].Backward(grad);

        double[] analytic = grad.Data;

        for (int j = 0; j < x.Data.Length; j++)
        {
            double orig = x.Data[j];
            const double eps = 1e-6;

            x.Data[j] = orig + eps;
            double lossPlus = model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);
            x.Data[j] = orig - eps;
            double lossMinus = model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);
            x.Data[j] = orig;

            double numeric = (lossPlus - lossMinus) / (2 * eps);
            Assert.True(Numeric.RelErr(analytic[j], numeric) < 1e-5,
                $"∂L/∂input[{j}]: analytic {analytic[j]:G17}, numeric {numeric:G17}");
        }
    }
}

// POZOR na namespace: soubor leží v Exceptions/, ale namespace je `My_DNN` (root), ne
// `My_DNN.Exceptions`. Ve zbytku repa složka = namespace, tady je to schválně jinak:
//   - Složka je pro NAŠI organizaci (bude jich víc), namespace je API pro volajícího.
//   - Kdo chytá `TrainingDivergedException`, má už `using My_DNN;` (jinak by neměl MDNN).
//     Nutit ho k druhému using je tření navíc bez užitku.
//   - Odpovídá to i BCL: `FileNotFoundException` je v `System.IO`, ne `System.IO.Exceptions`
//     — .NET dává výjimky tam, kam patří významem, ne do vlastního jmenného prostoru.
// Když to chceš sjednotit se zbytkem repa, je to jednořádková změna.
namespace My_DNN
{
    // Trénink zdivergoval — loss je NaN nebo nekonečno, takže další kroky by jen šířily
    // nesmysly a váhy jsou nepoužitelné.
    //
    // Dřív se na tomhle místě volalo `Environment.Exit(0)`: knihovna zabila hostitelský
    // proces, a ještě s návratovým kódem 0 = „úspěch". Volající neměl šanci zareagovat
    // a nadřazený nástroj (AutoML runner, CI skript) se ani nedozvěděl, že něco selhalo.
    // Výjimka to obrací — kdo chce, chytí ji a jede dál s dalším kandidátem; kdo ne,
    // dostane normální stack trace.
    public class TrainingDivergedException : Exception
    {
        // Epocha, ve které se divergence poprvé zjistila.
        public uint Epoch { get; }

        // Hodnota loss, která divergenci odhalila (NaN nebo ±∞).
        public double Loss { get; }

        public TrainingDivergedException(uint epoch, double loss)
            : base($"Trénink zdivergoval v epoše {epoch}: loss = {loss}. " +
                   "Výstup sítě je příliš velký nebo malý — zkus nižší learning rate, " +
                   "jinou inicializaci vah nebo normalizaci vstupů.")
        {
            Epoch = epoch;
            Loss = loss;
        }
    }
}

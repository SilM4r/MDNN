// Namespace `My_DNN` (root), i když soubor leží v Exceptions/ — stejný důvod jako
// u TrainingDivergedException: složka je pro naši organizaci, namespace je API volajícího.
namespace My_DNN
{
    // Soubor s modelem neodpovídá svému kontrolnímu součtu, nebo je jinak nepoužitelný.
    //
    // ⚠️ POZOR na význam: checksum dokazuje NEPORUŠENOST, ne PRAVOST. Detekuje useknutý
    // nebo poškozený soubor, chybu přenosu, ruční překlep v JSONu. NEdetekuje cílenou
    // manipulaci — kdo soubor upraví, přepočítá si i hash, algoritmus je veřejný.
    // Na to by byl potřeba HMAC s tajným klíčem nebo podpis, což má smysl jen tehdy,
    // když klíč není dosažitelný tomu, proti komu se model chrání.
    public class ModelFileCorruptedException : Exception
    {
        // Kontrolní součet zapsaný v souboru.
        public string? ExpectedChecksum { get; }

        // Kontrolní součet spočítaný z dat, která v souboru doopravdy jsou.
        public string? ActualChecksum { get; }

        public ModelFileCorruptedException(string message)
            : base(message)
        {
        }

        public ModelFileCorruptedException(string? expected, string? actual)
            : base($"The model checksum does not match — the file is damaged or has been modified. " +
                   $"In the file: {expected ?? "(missing)"}, computed from the data: {actual ?? "(n/a)"}. " +
                   "Note: the checksum proves integrity, not authenticity — it does not protect " +
                   "against deliberate tampering.")
        {
            ExpectedChecksum = expected;
            ActualChecksum = actual;
        }
    }
}

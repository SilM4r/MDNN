namespace My_DNN.Layers.classes
{
    /// <summary>
    /// Jak se z tvaru vstupu stane <c>[výška, šířka, kanály]</c> — společné pro vrstvy,
    /// které pracují v prostoru (<see cref="My_DNN.Layers.Conv"/>, <see cref="My_DNN.Layers.MaxPool"/>).
    /// </summary>
    ///
    /// Proč to má vlastní místo: tahle logika byla napsaná DVAKRÁT v každé z těch vrstev —
    /// zvlášť ve `WireShapes` (zapojení řetězu) a zvlášť ve `FeedForward` (každý průchod) —
    /// a ty dvě kopie se neshodovaly. Zapojení umělo rank 1 a 3, forward navíc rank 2.
    /// Model s Conv nebo MaxPool jako první vrstvou a vstupem `double[h,w]` proto nešel
    /// vůbec sestavit: spadl na `throw new Exception()` úplně bez textu v hlášce.
    ///
    /// Dokud je normalizace na jednom místě, nemají se ty dvě cesty jak rozejít — a to je
    /// přesně vzorec, který v týhle knihovně kousl už čtyřikrát (async backward bez dropoutu,
    /// FitAsync bez IsTraining, Train.FeedForward bez IsTraining).
    internal static class SpatialShape
    {
        /// <summary>Tvar na <c>[h,w,c]</c>. Vrací VŽDY nové pole — vrstva si nesmí držet
        /// referenci na tvar souseda, ten se pod ní může přepočítat.</summary>
        public static int[] Normalize(int[] shape, string layerName)
        {
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException($"{layerName}: the input shape is empty.");
            }

            switch (shape.Length)
            {
                case 3:
                    return new int[] { shape[0], shape[1], shape[2] };

                case 2:
                    // 2D obrázek bez kanálové dimenze → jeden kanál
                    return new int[] { shape[0], shape[1], 1 };

                case 1:
                    (int rows, int cols) = Factorize(shape[0], layerName);
                    return new int[] { rows, cols, 1 };

                default:
                    throw new ArgumentException(
                        $"{layerName} accepts input as [height,width,channels], [height,width] or a flat " +
                        $"vector, but got a {shape.Length}-dimensional shape [{Describe(shape)}].");
            }
        }

        /// <summary>Plochý vektor na 2D.</summary>
        ///
        /// ⚠️ Je to ODHAD, ne informace: pro 784 vyjde 28×28, ale pro 600 vyjde 25×24,
        /// přestože uživatel mohl mít na mysli 20×30, a RGB 28×28×3 (2352) se rozloží
        /// jako 49×48. Chování se tu ZÁMĚRNĚ zachovává beze změny — je to samostatná
        /// položka v roadmapě (chce to API na zadání tvaru, ne chytřejší hádání).
        public static (int rows, int cols) Factorize(int count, string layerName)
        {
            if (count <= 0)
            {
                throw new ArgumentException(
                    $"{layerName}: a flat input must have at least one element, got {count}.");
            }

            int rows = (int)Math.Ceiling(Math.Sqrt(count));

            while (count % rows != 0)
            {
                rows++;
            }

            return (rows, count / rows);
        }

        /// <summary>Ověří, že příchozí vstup sedí na tvar, na který je vrstva ZAPOJENÁ,
        /// a vrátí ho znormalizovaný.</summary>
        ///
        /// Tvary se zapojují jednou — prvním průchodem, zleva doprava celým řetězem — a od
        /// té chvíle jsou dané. Vrstva si podle nich předalokovala vstupní buffer, jenže
        /// kopírovací smyčka ve `FeedForward` se řídila tvarem PŘÍCHOZÍHO tenzoru. Menší
        /// vstup tedy přepsal jen svůj roh a zbytek bufferu zůstal z minulého vzorku:
        /// vrstva zapojená na 4×4 dostala samé nuly a vrátila čísla předchozího obrázku.
        /// Bez výjimky, bez varování. Větší vstup naopak spadl na `IndexOutOfRange`
        /// z hloubi smyčky, bez zmínky o tvarech.
        public static int[] RequireWiredShape(int[] incoming, int[] wired, string layerName)
        {
            if (wired.Length != 3 || wired[0] <= 0)
            {
                throw new InvalidOperationException(
                    $"{layerName} is not wired to any input shape yet. Add the layer to a model " +
                    "and let the first pass wire it, or call LayerAdjustment(...) yourself.");
            }

            int[] normalized = Normalize(incoming, layerName);

            if (!SameAs(normalized, wired))
            {
                string from = SameAs(normalized, incoming)
                    ? ""
                    : $" (z tvaru [{Describe(incoming)}])";

                throw new ArgumentException(
                    $"{layerName} is wired to input {Describe(wired)}, but got " +
                    $"{Describe(normalized)}{from}. The input shape must not change once the model is built — " +
                    "layers are wired once, on the first pass.");
            }

            return normalized;
        }

        public static bool SameAs(int[] a, int[] b)
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Tvar do hlášky: <c>4×4×1</c>.</summary>
        public static string Describe(int[] shape) => string.Join("×", shape);
    }
}

using My_DNN.Layers;
using My_DNN.Layers.classes;
using System.Text.Json;
using System.Text.Json.Serialization;
namespace My_DNN.Save_neural_network
{
    public class NetworkSaveLoadManager
    {
        public string Note { get; set; }
        public double Valid_Loss { get; set; }
        public string Schema { get; set; }
        public uint Current_epoch { get; set; }
        public uint Target_epoch { get; set; }
        public uint Mini_batch { get; set; }

        public string Loss_functions { get; set; }

        // Provenience: čím a kdy model vznikl.
        // Seed se dosud NEUKLÁDAL, takže z uloženého modelu nešlo zjistit, jak ho zopakovat.
        // Pozn.: seed sám o sobě NEumožní bezešvé navázání tréninku — k tomu by byla potřeba
        // i pozice v generátoru a stav optimizeru (momenty Adamu). Je to údaj o původu.
        public int? Seed { get; set; }

        // UTC + ISO-8601 ("O") přes JsonSerializer, ať to nezávisí na lokalizaci stroje.
        public DateTime SavedAtUtc { get; set; }

        public ExportOptimizer Optimizer { get; set; }

        [JsonConverter(typeof(LayerListConverter))]
        public List<BaseExportLayer> Layers { get; set; }

    
        public NetworkSaveLoadManager(MDNN model)
        {
            Schema = model.Schema;
            Current_epoch = model.Train.CurrentEpoch;
            Target_epoch = model.Train.TotalEpoch;
            Mini_batch = model.Train.MiniBatch;
            Loss_functions = model.Loss.Name;
            Valid_Loss = model.Loss.GetAverageLossPerIteration();
            Note = model.Note;
            Seed = model.Context.Seed;
            SavedAtUtc = DateTime.UtcNow;
            Optimizer = new ExportOptimizer(model.Optimizer);
            Layers = new List<BaseExportLayer>();

            foreach (Layer layer in model.Layers.Layers)
            {
                if (layer is Dense)
                {
                    Layers.Add(new ExportDenseLayer((Dense)layer));
                }
                else if (layer is RNN)
                {
                    Layers.Add(new ExportRnnLayer((RNN)layer));
                }
                else if (layer is Conv)
                {
                    Layers.Add(new ExportCNNLayer((Conv)layer));
                }
                else if (layer is MaxPool)
                {
                    Layers.Add(new ExportMaxPoolLayer((MaxPool)layer));
                }
                else
                {
                    throw new Exception("not implemetet yet");
                }
                    
            }
        }

        [JsonConstructor]
        public NetworkSaveLoadManager() 
        {
        }

        // Aktuální verze formátu. Soubor BEZ tohoto pole = starý (v0) plochý formát.
        public const int CurrentFormatVersion = 1;

        private static JsonSerializerOptions SerializerOptions() => new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
        };

        // Obálka: { FormatVersion, Checksum, Model: { ...vlastní model... } }
        //
        // Proč obálka a ne jen další pole v modelu: hash se počítá nad SUROVÝM textem
        // podsekce `Model`, takže se při ověřování nemusí nic reserializovat. Kdyby se
        // porovnávalo „přeserializuj a zahashuj znovu", výsledek by závisel na tom, že
        // formátování doubles a pořadí vlastností zůstane napříč verzemi .NET identické —
        // což není nic, na co by se dalo spolehnout.
        private sealed class ModelEnvelope
        {
            public int FormatVersion { get; set; }
            public string? Checksum { get; set; }
            public JsonElement Model { get; set; }
        }

        // Kanonický tvar: JSON bez odsazení. Hashuje se VŽDYCKY tohle, nikdy text ze souboru
        // tak, jak je — jinak by součet záležel na odsazení. Uvnitř obálky je model odsazený
        // o úroveň hlouběji, než když se serializuje sám, takže by se dva zápisy téhož modelu
        // lišily. (Vedlejší efekt, který se hodí: přeformátování souboru se nepočítá jako
        // poškození, protože kanonický tvar zůstane stejný.)
        //
        // Obě strany — ukládání i ověřování — jdou přes JsonElement do kompaktního zápisu,
        // takže je to bit po bitu stejný kód a nemá se kde rozejít.
        private static string Canonical(JsonElement element)
        {
            using var buffer = new MemoryStream();
            using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = false }))
            {
                element.WriteTo(writer);
            }
            return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
        }

        // SHA-256 nad UTF-8 textem. Prefix drží algoritmus u hodnoty, aby šel někdy vyměnit
        // (např. za HMAC s tajným klíčem) bez změny tvaru formátu.
        private static string ComputeChecksum(string payload)
        {
            byte[] hash = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(payload));
            return "sha256:" + Convert.ToHexString(hash).ToLowerInvariant();
        }

        // jádro serializace (bez souboru) — reuse pro disk i pro in-memory snapshot
        public string SaveToString()
        {
            JsonSerializerOptions options = SerializerOptions();

            string modelJson = System.Text.Json.JsonSerializer.Serialize(this, options);
            JsonElement modelElement = JsonDocument.Parse(modelJson).RootElement.Clone();

            var envelope = new ModelEnvelope
            {
                FormatVersion = CurrentFormatVersion,
                Checksum = ComputeChecksum(Canonical(modelElement)),
                Model = modelElement
            };

            return System.Text.Json.JsonSerializer.Serialize(envelope, options);
        }

        // Přípona .json se doplní, jen když tam ještě není. Dřív se přidávala vždycky,
        // takže Save("model.json") vyrobil "model.json.json".
        private const string JsonExtension = ".json";

        public static string EnsureJsonExtension(string path)
        {
            return path.EndsWith(JsonExtension, StringComparison.OrdinalIgnoreCase)
                ? path
                : path + JsonExtension;
        }

        public void Save(string fileName)
        {
            File.WriteAllText(EnsureJsonExtension(fileName), SaveToString());
        }

        public static NetworkSaveLoadManager LoadFromString(string json)
        {
            JsonSerializerOptions options = new JsonSerializerOptions
            {
                Converters = { new LayerConverter() }  // Přidání konvertoru pro vrstvy
            };

            string modelJson = UnwrapAndVerify(json);

            NetworkSaveLoadManager? model = System.Text.Json.JsonSerializer.Deserialize<NetworkSaveLoadManager>(modelJson, options);

            if (model != null)
            {
                return model;
            }

            else
            {
                throw new ArgumentException("Bad format of file");
            }
        }

        // Vrátí JSON vlastního modelu. Nový formát rozbalí a ověří kontrolní součet,
        // starý (bez FormatVersion) propustí beze změny — jinak by přestaly jít načíst
        // dřív uložené modely.
        private static string UnwrapAndVerify(string json)
        {
            JsonElement root;

            try
            {
                root = JsonDocument.Parse(json).RootElement;
            }
            catch (JsonException ex)
            {
                throw new ModelFileCorruptedException($"Soubor s modelem není platný JSON: {ex.Message}");
            }

            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("FormatVersion", out JsonElement versionElement))
            {
                return json;   // starý plochý formát (v0) — bez obálky i bez součtu
            }

            int version = versionElement.GetInt32();
            if (version > CurrentFormatVersion)
            {
                throw new ModelFileCorruptedException(
                    $"Model je uložený ve formátu verze {version}, tahle verze knihovny umí nejvýš " +
                    $"{CurrentFormatVersion}. Aktualizuj knihovnu.");
            }

            if (!root.TryGetProperty("Model", out JsonElement modelElement))
            {
                throw new ModelFileCorruptedException("Obálka modelu neobsahuje sekci 'Model'.");
            }

            // GetRawText() vrátí přesně ty znaky, které v souboru jsou → hash se počítá nad
            // původními daty, ne nad jejich novou serializací
            string modelJson = modelElement.GetRawText();

            string? expected = root.TryGetProperty("Checksum", out JsonElement checksumElement)
                ? checksumElement.GetString()
                : null;

            if (expected != null)
            {
                string actual = ComputeChecksum(Canonical(modelElement));
                if (!string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase))
                {
                    throw new ModelFileCorruptedException(expected, actual);
                }
            }

            return modelJson;
        }

        // Tolerantní protějšek k Save: `SaveAsJson("model")` zapíše "model.json", takže
        // `LoadModel("model")` musí fungovat taky. Cesta se bere přednostně tak, jak ji
        // uživatel zadal; teprve když takový soubor není, zkusí se s doplněnou příponou.
        public static NetworkSaveLoadManager Load(string fullPath)
        {
            string path = File.Exists(fullPath) ? fullPath : EnsureJsonExtension(fullPath);

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Model se nepodařilo najít ani jako '{fullPath}', ani jako '{EnsureJsonExtension(fullPath)}'.",
                    fullPath);
            }

            return LoadFromString(File.ReadAllText(path));
        }


    }
}

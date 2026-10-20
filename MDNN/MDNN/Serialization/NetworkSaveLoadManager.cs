using MDNN.Layers;
using System.Text.Json;
using System.Text.Json.Serialization;
namespace MDNN.Serialization
{
    // Property names (including the snake_case ones) are the JSON keys of the model file
    // format — do NOT rename them, or previously saved models will stop loading.
    internal class NetworkSaveLoadManager
    {
        public string Note { get; set; } = "";
        public double Valid_Loss { get; set; }
        public string Schema { get; set; } = "";
        public uint Current_epoch { get; set; }
        public uint Target_epoch { get; set; }
        public uint Mini_batch { get; set; }

        // A missing name is rejected later by Loss.ResolveForLoad (empty string = "missing").
        public string Loss_functions { get; set; } = "";

        // Provenance: how and when the model was created.
        // Previously the seed was NOT saved, so a saved model gave no way to reproduce it.
        // Note: the seed alone does NOT allow seamlessly resuming training — that also needs the
        // optimizer state (saved with neurons and Conv layers, see ExportNeuron) and the generator
        // position (not saved). It is provenance information.
        public int? Seed { get; set; }

        // UTC + ISO-8601 ("O") via JsonSerializer, so it does not depend on the machine's locale.
        public DateTime SavedAtUtc { get; set; }

        // Regularization is part of the training RECIPE, not of the trained weights — without saving
        // it, a loaded model would continue training without it, invisibly.
        // Older files lack these keys → they load as 0, i.e. off.
        public double L2 { get; set; }
        public double Dropout { get; set; }

        // `null!` only until deserialization fills the field — LoadFromString then verifies that it
        // really was in the file (RequireComplete); otherwise a damaged file would surface only as a
        // NullReferenceException while building the model.
        public ExportOptimizer Optimizer { get; set; } = null!;

        [JsonConverter(typeof(LayerListConverter))]
        public List<BaseExportLayer> Layers { get; set; } = null!;

    
        public NetworkSaveLoadManager(NeuralNetwork model, bool includeOptimizerState = true)
        {
            Schema = model.Schema;
            Current_epoch = model.Train.CurrentEpoch;
            Target_epoch = model.Train.TotalEpoch;
            Mini_batch = model.Train.MiniBatchSize;
            Loss_functions = model.Loss.Name;
            Valid_Loss = model.Loss.GetAverageLossPerIteration();
            Note = model.Note;
            Seed = model.Context.Seed;
            L2 = model.Context.L2;
            Dropout = model.Context.Dropout;
            SavedAtUtc = DateTime.UtcNow;
            Optimizer = new ExportOptimizer(model.Optimizer);
            Layers = new List<BaseExportLayer>();

            foreach (Layer layer in model.Layers.Layers)
            {
                if (layer is Dense)
                {
                    Layers.Add(new ExportDenseLayer((Dense)layer, includeOptimizerState));
                }
                else if (layer is RNN)
                {
                    Layers.Add(new ExportRnnLayer((RNN)layer, includeOptimizerState));
                }
                else if (layer is Conv)
                {
                    Layers.Add(new ExportCNNLayer((Conv)layer, includeOptimizerState));
                }
                else if (layer is MaxPool)
                {
                    Layers.Add(new ExportMaxPoolLayer((MaxPool)layer));
                }
                else
                {
                    throw new NotSupportedException($"Saving of layer type '{layer.GetType().Name}' is not supported yet — only Dense, RNN, Conv and MaxPool can be saved.");
                }
                    
            }
        }

        [JsonConstructor]
        public NetworkSaveLoadManager() 
        {
        }

        // Current format version. A file WITHOUT this field = the old (v0) flat format.
        public const int CurrentFormatVersion = 1;

        private static JsonSerializerOptions SerializerOptions() => new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new LayerConverter() }  // converter for layers
        };

        // Envelope: { FormatVersion, Checksum, Model: { ...the model itself... } }
        //
        // Why an envelope and not just another field in the model: the hash is computed over the
        // RAW text of the `Model` subsection, so verification needs no re-serialization. Comparing via
        // "re-serialize and hash again" would rely on double formatting and property order staying
        // identical across .NET versions — nothing to depend on.
        private sealed class ModelEnvelope
        {
            public int FormatVersion { get; set; }
            public string? Checksum { get; set; }
            public JsonElement Model { get; set; }
        }

        // Canonical form: JSON without indentation. This is ALWAYS what gets hashed, never the file
        // text as is — otherwise the checksum would depend on indentation. Inside the envelope the
        // model is indented one level deeper than when serialized alone, so two writes of the same
        // model would differ. (Useful side effect: reformatting the file does not count as damage,
        // because the canonical form stays the same.)
        //
        // Both sides — saving and verification — go through JsonElement to the compact form, so it
        // is the very same code and cannot diverge.
        private static string Canonical(JsonElement element)
        {
            using var buffer = new MemoryStream();
            using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = false }))
            {
                element.WriteTo(writer);
            }
            return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
        }

        // SHA-256 over the UTF-8 text. The prefix keeps the algorithm next to the value so it can be
        // swapped later (e.g. for an HMAC with a secret key) without changing the format shape.
        private static string ComputeChecksum(string payload)
        {
            byte[] hash = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(payload));
            return "sha256:" + Convert.ToHexString(hash).ToLowerInvariant();
        }

        // core serialization (no file) — shared by saving to disk and the in-memory snapshot
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

        // The .json extension is added only when not already present. Previously it was always
        // added, so Save("model.json") produced "model.json.json".
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
                Converters = { new LayerConverter() }  // converter for layers
            };

            string modelJson = UnwrapAndVerify(json);

            NetworkSaveLoadManager? model = System.Text.Json.JsonSerializer.Deserialize<NetworkSaveLoadManager>(modelJson, options);

            if (model != null)
            {
                model.RequireComplete();
                return model;
            }

            else
            {
                throw new ArgumentException("Bad format of file");
            }
        }

        // Fields without which the model cannot be built (see the Optimizer/Layers declarations).
        private void RequireComplete()
        {
            if ((object?)Optimizer == null)
                throw new ModelFileCorruptedException("The model file has no 'Optimizer' section.");
            if ((object?)Layers == null)
                throw new ModelFileCorruptedException("The model file has no 'Layers' section.");
        }

        // Returns the JSON of the model itself. The new format is unwrapped and its checksum verified;
        // the old one (without FormatVersion) passes through unchanged — otherwise previously saved
        // models would no longer load.
        private static string UnwrapAndVerify(string json)
        {
            JsonElement root;

            try
            {
                root = JsonDocument.Parse(json).RootElement;
            }
            catch (JsonException ex)
            {
                throw new ModelFileCorruptedException($"The model file is not valid JSON: {ex.Message}");
            }

            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("FormatVersion", out JsonElement versionElement))
            {
                return json;   // old flat format (v0) — no envelope, no checksum
            }

            int version = versionElement.GetInt32();
            if (version > CurrentFormatVersion)
            {
                throw new ModelFileCorruptedException(
                    $"The model is saved in format version {version}; this version of the library supports at most " +
                    $"{CurrentFormatVersion}. Update the library.");
            }

            if (!root.TryGetProperty("Model", out JsonElement modelElement))
            {
                throw new ModelFileCorruptedException("The model envelope has no 'Model' section.");
            }

            // GetRawText() returns exactly the characters in the file → the hash is computed over
            // the original data, not over a re-serialization
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

        // Lenient counterpart to Save: `SaveAsJson("model")` writes "model.json", so
        // `LoadModel("model")` must work too. The path is first tried as given; only if no such
        // file exists is the extension appended.
        public static NetworkSaveLoadManager Load(string fullPath)
        {
            string path = File.Exists(fullPath) ? fullPath : EnsureJsonExtension(fullPath);

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"The model was found neither as '{fullPath}' nor as '{EnsureJsonExtension(fullPath)}'.",
                    fullPath);
            }

            return LoadFromString(File.ReadAllText(path));
        }


    }
}

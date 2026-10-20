// Namespace `MDNN` (root) even though the file lives in Exceptions/ — same reason as for
// TrainingDivergedException: the folder is for our organization, the namespace is the caller's API.
namespace MDNN
{
    // NOTE on meaning: the checksum proves INTEGRITY, not AUTHENTICITY. It detects a truncated
    // or damaged file, a transfer error, a manual typo in the JSON. It does NOT detect deliberate
    // tampering — whoever edits the file can recompute the hash, the algorithm is public.
    // That would need an HMAC with a secret key or a signature, which only makes sense when the
    // key is out of reach of whoever the model is being protected from.
    /// <summary>
    /// Thrown when a model file does not match its checksum or is otherwise unusable.
    /// </summary>
    /// <remarks>
    /// The checksum proves integrity, not authenticity: it does not protect against deliberate tampering.
    /// </remarks>
    public class ModelFileCorruptedException : Exception
    {
        /// <summary>
        /// The checksum stored in the file.
        /// </summary>
        public string? ExpectedChecksum { get; }

        /// <summary>
        /// The checksum computed from the data actually present in the file.
        /// </summary>
        public string? ActualChecksum { get; }

        /// <summary>
        /// Creates the exception with a custom message.
        /// </summary>
        public ModelFileCorruptedException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Creates the exception for a checksum mismatch.
        /// </summary>
        /// <param name="expected">Checksum stored in the file.</param>
        /// <param name="actual">Checksum computed from the data.</param>
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

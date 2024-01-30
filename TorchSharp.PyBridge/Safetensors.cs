using ICSharpCode.SharpZipLib.Zip.Compression.Streams;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TorchSharp.PyBridge {
    static class Safetensors {

        public static Dictionary<string, torch.Tensor> LoadStateDict(string path, List<string>? keysToKeep = null) {
            using var stream = File.OpenRead(path);
            return LoadStateDict(stream, keysToKeep: keysToKeep);
        }

        public static Dictionary<string, torch.Tensor> LoadStateDict(Stream stream, bool leaveOpen = false, List<string>? keysToKeep = null) {

            // Start by loading in the index of all the tensors
            var index = LoadIndex(stream);

            long offset = stream.Position;
            // Each entry in the index contains all the info for reconstructing the tensors
            var ret = new Dictionary<string, torch.Tensor>();
            foreach (var kvp in index) {
                if (kvp.Key == "__metadata__") continue;
                if (keysToKeep is not null && !keysToKeep.Contains(kvp.Key)) continue;

                var tensor = torch.empty(kvp.Value.Shape, dtype: ConvertToTorchDType(kvp.Value.DataType));

                // Make sure the length isn't > int.MaxValue, since .NET has the 2GB limit
                long length = kvp.Value.Offsets[1] - kvp.Value.Offsets[0];
                if (length > int.MaxValue)
                    throw new NotImplementedException("Loading tensors larger than 2GB");

                stream.Position = offset + kvp.Value.Offsets[0];
                tensor.bytes = stream.ReadBytes((int)length);

                ret.Add(kvp.Key, tensor);
            }


            if (!leaveOpen)
                stream.Close();

            return ret;
        }

        public static void SaveStateDict(string path, Dictionary<string, torch.Tensor> stateDict) {
            using var stream = File.OpenWrite(path);
            SaveStateDict(path, stateDict);
        }

        public static void SaveStateDict(Stream stream, Dictionary<string, torch.Tensor> stateDict, bool leaveOpen = false) {
            // We want to first build the index and then write out the tensors themselves. Therefore, first convert
            // the state dict to an ordered collection, and build the index. 
            var orderedState = stateDict.ToList();
            var index = new Dictionary<string, SafetensorsEntry>();
            long offset = 0;
            foreach (var kvp in orderedState) {
                long length = kvp.Value.NumberOfElements * kvp.Value.ElementSize;

                index.Add(kvp.Key, new SafetensorsEntry() {
                    DataType = ConvertToSafeTensorsDType(kvp.Value.dtype),
                    Shape = kvp.Value.shape,
                    Offsets = new[] { offset, offset + length }
                });
                offset += length;
            }// next key

            string indexJson = JsonSerializer.Serialize(index);

            // Write out the JSON followed by the bytes of the tensors
            var br = new BinaryWriter(stream);
            br.Write((ulong)indexJson.Length);
            br.Write(Encoding.UTF8.GetBytes(indexJson));
            foreach (var kvp in orderedState)
                br.Write(kvp.Value.bytes);

            if (!leaveOpen)
                br.Close();
        }


        public static Dictionary<string, SafetensorsEntry> LoadIndex(string path) {
            using var stream = File.OpenRead(path);
            return LoadIndex(stream);
        }

        public static Dictionary<string, SafetensorsEntry> LoadIndex(Stream stream) {
            // First 8 bytes represent the length of the JSON in UTF8.
            ulong length = BitConverter.ToUInt64(stream.ReadBytes(8));
            if (length > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(length), "Length of JSON exceeded int.MaxValue, not supported yet");
            
            // Read the rest of the JSON, and deserialize it
            var jsonBytes = stream.ReadBytes((int)length);
            return JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntry>>(Encoding.UTF8.GetString(jsonBytes)) ?? throw new NotImplementedException("Loaded header string failed to deserialize into the correct format.");
        }

        private static torch.ScalarType ConvertToTorchDType(string dataType) {
            return dataType switch {
                "F64" => torch.ScalarType.Float64,
                "F32" => torch.ScalarType.Float32,
                "F16" => torch.ScalarType.Float16,
                "BF16" => torch.ScalarType.BFloat16,
                "I64" => torch.ScalarType.Int64,
                "I32" => torch.ScalarType.Int32,
                "I16" => torch.ScalarType.Int16,
                "I8" => torch.ScalarType.Int8,
                "U8" => torch.ScalarType.Byte,
                "BOOL" => torch.ScalarType.Bool,
                _ => throw new NotImplementedException($"Unrecognized data type listed: {dataType}")
            };
        }

        private static string ConvertToSafeTensorsDType(torch.ScalarType dtype) {
            return dtype switch {
                torch.ScalarType.Float64 => "F64",
                torch.ScalarType.Float32 => "F32",
                torch.ScalarType.Float16 => "F16",
                torch.ScalarType.BFloat16 => "BF16",
                torch.ScalarType.Int64 => "I64",
                torch.ScalarType.Int32 => "I32",
                torch.ScalarType.Int16 => "I16",
                torch.ScalarType.Int8 => "I8",
                torch.ScalarType.Byte => "U8",
                torch.ScalarType.Bool => "BOOL",
                _ => throw new NotImplementedException($"Unrecognized data type listed: {dtype}")
            };
        }
    }
    internal class SafetensorsEntry {
        [JsonPropertyName("dtype")]
        public string DataType { get; set; }

        [JsonPropertyName("shape")]
        public long[] Shape { get; set; }

        [JsonPropertyName("data_offsets")]
        public long[] Offsets { get; set; }
    }
}

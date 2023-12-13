using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;

namespace TorchSharp.PyBridge.Tests {
    internal static class SaveUtils {
        public static bool SaveAndCompare(OptimizerHelper optim, string file) {
            // Save the model to a memory stream
            var ms = new MemoryStream();
            optim.save_py(ms, leaveOpen: true);
            ms.Position = 0;

            // Compare the bytes to pyload_test
            return CompareSavedModules(ms, File.OpenRead(file));
        }
        public static bool CompareSavedModules(Stream baseFile, Stream targetFile) {
            // One catch: They are zip files, and therefore the timestamp is embedded 
            // in the bytes. Therefore, we are going to extract every entry in the archive
            // and compare the raw bytes then.
            byte[] compBytes = new ZipArchive(baseFile).ExtractAllContentBytes();
            byte[] goldBytes = new ZipArchive(targetFile).ExtractAllContentBytes();

            return Enumerable.SequenceEqual(goldBytes, compBytes);
        }
    }
    static class ZipArchiveExtensions {
        public static byte[] ExtractAllContentBytes(this ZipArchive archive) {
            var ms = new MemoryStream();
            foreach (var entry in archive.Entries.OrderBy(e => e.FullName))
                entry.Open().CopyTo(ms);

            return ms.ToArray();
        }
    }
}

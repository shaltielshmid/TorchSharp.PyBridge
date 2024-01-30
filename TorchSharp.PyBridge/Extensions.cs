using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.PyBridge {
    static class Extensions {
        public static byte[] ReadBytes(this Stream stream, int count) {
            var ret = new byte[count];
            stream.Read(ret, 0, count);
            return ret;
        }

        public static void RemoveKeys<K,V>(this IDictionary<K, V> dict, IList<K> keys) {
            foreach (var key in keys) dict.Remove(key);
        }
    }
}

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp.PyBridge {
    internal static class OptimizerUtils {
        internal static void AssignFieldsAndPropsToTargetTable<T>(T obj, IDictionary targetTable) where T : notnull {
            // Go through all the fields
            foreach (var field in obj.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public)) {
                object? value = field.GetValue(obj);

                SetValueInTargetTable(field.Name, value, targetTable);
            }// next property

            // Go through all the properties
            foreach (var property in obj.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public)) {
                object? value = property.GetValue(obj);

                SetValueInTargetTable(property.Name, value, targetTable);
            }// next property
        }

        internal static void AssignFieldsAndPropsFromReferenceTable<T>(T obj, IDictionary referenceTable) where T : notnull {
            // Go through all the fields
            foreach (var field in obj.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)) {
                object? value = GetValueFromReferenceTable(field.Name, field.FieldType, referenceTable);

                // Set the value!
                // If it's a tensor - first dispose the old tensor, and then set the new value
                if (field.FieldType == typeof(torch.Tensor)) {
                    var orig = (torch.Tensor?)field.GetValue(obj);
                    if (orig is not null)
                        orig?.Dispose();
                }
                field.SetValue(obj, Convert.ChangeType(value, Nullable.GetUnderlyingType(field.FieldType) ?? field.FieldType));
            }// next property

            // Go through all the properties
            foreach (var property in obj.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)) {
                object? value = GetValueFromReferenceTable(property.Name, property.PropertyType, referenceTable);

                // Set the value!
                // If it's a tensor - first dispose the old tensor, and then set the new value
                if (property.PropertyType == typeof(torch.Tensor)) {
                    var orig = (torch.Tensor?)property.GetValue(obj);
                    if (orig is not null)
                        orig?.Dispose();
                }
                property.SetValue(obj, Convert.ChangeType(value, Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType));
            }// next property
        }

        private static object? GetValueFromReferenceTable(string name, Type type, IDictionary referenceTable) {
            // Special handling for lrs/betas/eta/step_sizes/step:
            return name switch {
                "LearningRate" or "InitialLearningRate" => referenceTable["lr"],
                "beta1" or "beta2" => ((object[])referenceTable["betas"]!)[name == "beta1" ? 0 : 1],
                "etaminus" or "etaplus" => ((object[])referenceTable["etas"]!)[name == "etaminus" ? 0 : 1],
                "min_step" or "max_step" => ((object[])referenceTable["step_sizes"]!)[name == "min_step" ? 0 : 1],
                _ when type == typeof(torch.Tensor) => referenceTable[name!],
                _ => GetValueFromMaybeTensor(referenceTable[name]!, type),
            };
        }

        private static object? GetValueFromMaybeTensor(object obj, Type type) {
            if (obj is null || obj is not torch.Tensor)
                return obj;
            // Stored as a tensor, so return it as a float and dispose the tensor
            using var tensor = (torch.Tensor)obj;
            return tensor.dtype switch {
                ScalarType.Byte => tensor.ToByte(),
                ScalarType.Int8 => tensor.@short().ToInt16(),
                ScalarType.Int16 => tensor.ToInt16(),
                ScalarType.Int32 => tensor.ToInt32(),
                ScalarType.Int64 => tensor.ToInt64(),
                ScalarType.Float16 => tensor.ToHalf(),
                ScalarType.Float32 => tensor.ToSingle(),
                ScalarType.Float64 => tensor.ToDouble(),
                ScalarType.ComplexFloat32 => tensor.ToComplexFloat32(),
                ScalarType.ComplexFloat64 => tensor.ToComplexFloat64(),
                ScalarType.Bool => tensor.ToBoolean(),
                ScalarType.BFloat16 => tensor.@half().ToHalf(),
                _ => throw new ArgumentException($"Loaded tensor of type unknown to `TorchSharp.PyBridge`: {tensor.dtype}. Please open an issue in the repository.")
            };
        }

        private static void SetValueInTargetTable(string name, object? value, IDictionary targetTable) {
            // Special lrs/handling for betas/eta/step_sizes/step:
            switch (name) {
                case "InitialLearningRate": break;
                case "LearningRate": targetTable["lr"] = value; break;
                case "beta1": case "beta2": Set2ItemTupleValue(targetTable, "betas", value, name == "beta1" ? 0 : 1); break;
                case "etaminus": case "etaplus": Set2ItemTupleValue(targetTable, "etas", value, name == "etaminus" ? 0 : 1); break;
                case "min_step": case "max_step": Set2ItemTupleValue(targetTable, "step_sizes", value, name == "min_step" ? 0 : 1); break;
                case "step": targetTable[name] = torch.tensor((long)value!); break;
                default: targetTable[name] = value; break;
            }
        }

        private static void Set2ItemTupleValue(IDictionary targetTable, string key, object? value, int idx) {
            if (targetTable[key] is null)
                targetTable[key] = new object?[2];
            ((object?[])targetTable[key]!)[idx] = value;
        }
    }
}

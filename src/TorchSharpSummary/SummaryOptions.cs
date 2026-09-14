using TorchSharp;

namespace TorchSharpSummary;

/// <summary>Display unit used when rendering memory footprints in a <see cref="ModelSummary"/> table.</summary>
public enum MemoryUnit
{
    /// <summary>Raw bytes.</summary>
    Bytes,

    /// <summary>Kilobytes (1,024 bytes).</summary>
    KB,

    /// <summary>Megabytes (1,024² bytes).</summary>
    MB,

    /// <summary>Gigabytes (1,024³ bytes).</summary>
    GB,
}

/// <summary>
/// Configures how <see cref="ModuleExtensions.Summary(torch.nn.Module, long[][])"/> executes the dry-run
/// pass and how the resulting <see cref="ModelSummary"/> is formatted.
/// </summary>
public sealed class SummaryOptions
{
    /// <summary>
    /// Maximum module-tree depth to display in the rendered table. Depth 0 is the top-level model's
    /// direct children. Defaults to <see cref="int.MaxValue"/> (show every nesting level). Every layer
    /// is still visited and included in the computed totals regardless of this setting — it only
    /// affects which rows are printed.
    /// </summary>
    public int MaxDepth { get; init; } = int.MaxValue;

    /// <summary>Unit used to display memory footprints (default: <see cref="MemoryUnit.MB"/>).</summary>
    public MemoryUnit MemoryUnit { get; init; } = MemoryUnit.MB;

    /// <summary>Whether to include the "Input Shape" column. Defaults to <see langword="true"/>.</summary>
    public bool ShowInputShape { get; init; } = true;

    /// <summary>Whether to include the "Output Shape" column. Defaults to <see langword="true"/>.</summary>
    public bool ShowOutputShape { get; init; } = true;

    /// <summary>Whether to include the "Param #" column. Defaults to <see langword="true"/>.</summary>
    public bool ShowParamCount { get; init; } = true;

    /// <summary>Whether to include the "Mult-Adds" (MACs) column. Defaults to <see langword="true"/>.</summary>
    public bool ShowMacs { get; init; } = true;

    /// <summary>Whether to print the memory footprint section (input/forward-pass/params/total size). Defaults to <see langword="true"/>.</summary>
    public bool ShowMemory { get; init; } = true;

    /// <summary>Width, in characters, reserved for the "Layer (type)" column. Defaults to 40.</summary>
    public int NameColumnWidth { get; init; } = 40;

    /// <summary>Element type used for the dummy input tensors created during the dry run. Defaults to <see cref="torch.ScalarType.Float32"/>.</summary>
    public torch.ScalarType InputDType { get; init; } = torch.ScalarType.Float32;

    /// <summary>Device the dummy input tensors are created on. Defaults to <see langword="null"/> (the library's default device, normally CPU).</summary>
    public torch.Device? Device { get; init; }
}

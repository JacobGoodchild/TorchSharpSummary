using System.Text;

namespace TorchSharpSummary;

/// <summary>
/// The result of running <see cref="ModuleExtensions.Summary(TorchSharp.torch.nn.Module, long[][])"/>
/// on a model: a per-layer breakdown plus aggregate parameter counts, MAC estimates, and memory
/// footprint, along with a formatted, torchinfo-style tabular representation via <see cref="ToString"/>.
/// </summary>
public sealed class ModelSummary
{
    /// <summary>The top-level model's CLR type name.</summary>
    public required string ModelName { get; init; }

    /// <summary>Every module that was actually invoked during the dry-run forward pass, in execution order.</summary>
    public required IReadOnlyList<LayerInfo> Layers { get; init; }

    /// <summary>Total number of parameters across the whole model with <c>requires_grad == true</c>.</summary>
    public required long TrainableParams { get; init; }

    /// <summary>Total number of parameters across the whole model with <c>requires_grad == false</c>.</summary>
    public required long NonTrainableParams { get; init; }

    /// <summary>Total parameters across the whole model (<see cref="TrainableParams"/> + <see cref="NonTrainableParams"/>).</summary>
    public long TotalParams => TrainableParams + NonTrainableParams;

    /// <summary>Estimated native memory, in bytes, occupied by all of the model's parameters.</summary>
    public required long ParamBytes { get; init; }

    /// <summary>Estimated native memory, in bytes, occupied by the dry-run input tensor(s).</summary>
    public required long InputBytes { get; init; }

    /// <summary>Estimated native memory, in bytes, occupied by every intermediate activation produced during the forward pass.</summary>
    public required long ActivationBytes { get; init; }

    /// <summary>Rough estimate of total native memory required for one forward pass: <see cref="ParamBytes"/> + <see cref="InputBytes"/> + <see cref="ActivationBytes"/>.</summary>
    public long EstimatedTotalBytes => ParamBytes + InputBytes + ActivationBytes;

    /// <summary>Total estimated multiply-accumulate operations (MACs) across all layers. See <see cref="LayerInfo.Macs"/> for per-layer coverage.</summary>
    public required long TotalMacs { get; init; }

    /// <summary>The options this summary was rendered with.</summary>
    public required SummaryOptions Options { get; init; }

    /// <summary>Writes the formatted summary table to the console (equivalent to <c>Console.WriteLine(summary.ToString())</c>).</summary>
    public void Print() => Console.WriteLine(ToString());

    /// <inheritdoc />
    public override string ToString()
    {
        var columns = BuildColumns();
        int totalWidth = columns.Sum(c => c.Width) + (columns.Count - 1);
        string majorRule = new string('=', totalWidth);
        string minorRule = new string('-', totalWidth);

        var sb = new StringBuilder();
        sb.Append("Model: ").Append(ModelName).Append('\n');
        sb.Append(minorRule).Append('\n');
        sb.Append(string.Join(" ", columns.Select(c => c.Header.PadRight(c.Width)))).Append('\n');
        sb.Append(majorRule).Append('\n');

        // Layers are stored in actual execution order (children fire before the parent that
        // contains them), but that reads oddly as a table: show the root model first, as a
        // header row, followed by its children in the order they actually ran.
        var displayOrder = Layers.Where(l => l.Depth == -1).Concat(Layers.Where(l => l.Depth != -1));
        foreach (var layer in displayOrder.Where(l => l.Depth <= Options.MaxDepth))
        {
            sb.Append(string.Join(" ", columns.Select(c => FitToWidth(c.Select(layer), c.Width))))
              .Append('\n');
        }

        sb.Append(majorRule).Append('\n');
        sb.Append("Total params: ").Append(TotalParams.ToString("N0")).Append('\n');
        sb.Append("Trainable params: ").Append(TrainableParams.ToString("N0")).Append('\n');
        sb.Append("Non-trainable params: ").Append(NonTrainableParams.ToString("N0")).Append('\n');
        if (Options.ShowMacs)
        {
            sb.Append("Total mult-adds (MACs): ").Append(TotalMacs.ToString("N0")).Append('\n');
        }

        if (Options.ShowMemory)
        {
            sb.Append(minorRule).Append('\n');
            string unit = UnitLabel(Options.MemoryUnit);
            sb.Append("Input size (").Append(unit).Append("): ").Append(ConvertBytes(InputBytes).ToString("F2")).Append('\n');
            sb.Append("Forward pass size (").Append(unit).Append("): ").Append(ConvertBytes(ActivationBytes).ToString("F2")).Append('\n');
            sb.Append("Params size (").Append(unit).Append("): ").Append(ConvertBytes(ParamBytes).ToString("F2")).Append('\n');
            sb.Append("Estimated Total Size (").Append(unit).Append("): ").Append(ConvertBytes(EstimatedTotalBytes).ToString("F2")).Append('\n');
        }

        sb.Append(majorRule);
        return sb.ToString();
    }

    private List<(string Header, int Width, Func<LayerInfo, string> Select)> BuildColumns()
    {
        var columns = new List<(string, int, Func<LayerInfo, string>)>
        {
            ("Layer (type)", Options.NameColumnWidth, l => new string(' ', Math.Max(0, l.Depth) * 2) + $"{l.Name} ({l.LayerType})"),
        };

        if (Options.ShowInputShape)
            columns.Add(("Input Shape", 22, l => FormatShapes(l.InputShapes)));
        if (Options.ShowOutputShape)
            columns.Add(("Output Shape", 22, l => l.OutputShape is null ? "--" : FormatShape(l.OutputShape)));
        if (Options.ShowParamCount)
            columns.Add(("Param #", 16, l => l.TotalParams == 0 ? "--" : l.TotalParams.ToString("N0")));
        if (Options.ShowMacs)
            columns.Add(("Mult-Adds", 16, l => l.Macs == 0 ? "--" : l.Macs.ToString("N0")));

        return columns;
    }

    private double ConvertBytes(long bytes) => Options.MemoryUnit switch
    {
        MemoryUnit.Bytes => bytes,
        MemoryUnit.KB => bytes / 1024.0,
        MemoryUnit.MB => bytes / (1024.0 * 1024.0),
        MemoryUnit.GB => bytes / (1024.0 * 1024.0 * 1024.0),
        _ => bytes,
    };

    private static string UnitLabel(MemoryUnit unit) => unit switch
    {
        MemoryUnit.Bytes => "B",
        MemoryUnit.KB => "KB",
        MemoryUnit.MB => "MB",
        MemoryUnit.GB => "GB",
        _ => "B",
    };

    internal static string FormatShape(long[] shape) => "[" + string.Join(", ", shape) + "]";

    internal static string FormatShapes(IReadOnlyList<long[]> shapes) =>
        shapes.Count == 0 ? "--" : string.Join(", ", shapes.Select(FormatShape));

    private static string FitToWidth(string value, int width)
    {
        if (value.Length > width)
            value = width <= 1 ? value[..width] : value[..(width - 1)] + "…";
        return value.PadRight(width);
    }
}

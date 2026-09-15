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
        var rows = BuildTreeRows();
        var columns = BuildColumns(rows);
        int totalWidth = columns.Sum(c => c.Width) + (columns.Count - 1);
        string majorRule = new string('=', totalWidth);
        string minorRule = new string('-', totalWidth);

        var sb = new StringBuilder();
        sb.Append("Model: ").Append(ModelName).Append('\n');
        sb.Append(minorRule).Append('\n');
        sb.Append(string.Join(" ", columns.Select(c => c.Header.PadRight(c.Width)))).Append('\n');
        sb.Append(majorRule).Append('\n');

        foreach (var row in rows)
        {
            sb.Append(string.Join(" ", columns.Select(c => FitToWidth(c.Select(row), c.Width))))
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

    /// <summary>One line of the rendered table: a layer plus the tree-branch prefix ("├─ ", "│  └─ ", ...) in front of its own name.</summary>
    private readonly record struct TreeRow(LayerInfo Layer, string TreeLabel);

    /// <summary>
    /// Reconstructs the actual call tree from <see cref="Layers"/> and walks it into display
    /// order: the root model first (as a header row), then each child immediately followed by
    /// its own children, each row prefixed with box-drawing branch connectors so nesting reads
    /// as a tree rather than a flat, same-indent list. Rows deeper than
    /// <see cref="SummaryOptions.MaxDepth"/> are pruned (along with their descendants).
    /// </summary>
    /// <remarks>
    /// <see cref="Layers"/> is stored in actual execution order, which for a forward pass is a
    /// post-order traversal of the call tree: every descendant of a call finishes (and fires its
    /// hook) before that call's own hook fires. Parentage is reconstructed from that order plus
    /// each row's <see cref="LayerInfo.Depth"/> — deliberately <i>not</i> from matching dotted
    /// names, because a module invoked more than once in one pass (weight sharing, a loop) shows
    /// up as multiple distinct rows that share the same name; grouping by name would smear every
    /// occurrence's children across all of them. Grouping by reference identity instead gives
    /// each call its own, correctly-scoped set of children.
    /// </remarks>
    private List<TreeRow> BuildTreeRows()
    {
        LayerInfo? root = null;
        // pendingChildren[d] accumulates rows at depth d that haven't yet been claimed by an
        // enclosing call at depth d-1. Because the input is post-order, a call's own children
        // are always exactly whatever is sitting in pendingChildren[call.Depth + 1] at the
        // moment the call itself is reached.
        var pendingChildren = new Dictionary<int, List<LayerInfo>>();
        var childrenOf = new Dictionary<LayerInfo, List<LayerInfo>>();
        var occurrenceIndexOf = new Dictionary<LayerInfo, int>();
        var occurrencesSeenByName = new Dictionary<string, int>();

        foreach (var layer in Layers)
        {
            if (layer.Depth == -1) { root = layer; continue; }

            childrenOf[layer] = pendingChildren.TryGetValue(layer.Depth + 1, out var claimed)
                ? claimed
                : new List<LayerInfo>();
            pendingChildren.Remove(layer.Depth + 1);

            if (!pendingChildren.TryGetValue(layer.Depth, out var siblings))
                pendingChildren[layer.Depth] = siblings = new List<LayerInfo>();
            siblings.Add(layer);

            occurrenceIndexOf[layer] = occurrencesSeenByName.GetValueOrDefault(layer.Name);
            occurrencesSeenByName[layer.Name] = occurrenceIndexOf[layer] + 1;
        }

        var rows = new List<TreeRow>();
        if (root is not null)
            rows.Add(new TreeRow(root, root.LayerType));

        // Whatever never got claimed by an enclosing call is, by construction, the top level.
        var topLevel = pendingChildren.GetValueOrDefault(0, new List<LayerInfo>());

        void Walk(LayerInfo layer, string continuationPrefix, bool isLast)
        {
            if (layer.Depth > Options.MaxDepth) return;

            string leafName = LeafName(layer.Name);
            // A module invoked more than once in a single pass reuses the same weights, so only
            // its first occurrence carries parameter counts (see AttachParameterCounts) — later
            // occurrences are flagged "(recursive)" here, matching torchinfo's convention, so the
            // repeat reads as intentional rather than as a rendering bug.
            string recursiveTag = occurrenceIndexOf[layer] > 0 ? " (recursive)" : "";
            rows.Add(new TreeRow(layer, $"{continuationPrefix}{(isLast ? "└─ " : "├─ ")}{leafName} ({layer.LayerType}){recursiveTag}"));

            var children = childrenOf.GetValueOrDefault(layer, new List<LayerInfo>());
            string childPrefix = continuationPrefix + (isLast ? "   " : "│  ");
            for (int i = 0; i < children.Count; i++)
                Walk(children[i], childPrefix, i == children.Count - 1);
        }

        for (int i = 0; i < topLevel.Count; i++)
            Walk(topLevel[i], string.Empty, i == topLevel.Count - 1);

        return rows;
    }

    private static string LeafName(string dottedName)
    {
        int lastDot = dottedName.LastIndexOf('.');
        return lastDot < 0 ? dottedName : dottedName[(lastDot + 1)..];
    }

    private List<(string Header, int Width, Func<TreeRow, string> Select)> BuildColumns(List<TreeRow> rows)
    {
        // The name column grows to fit the deepest/longest entry actually being rendered
        // (tree prefixes get wider with nesting), instead of silently truncating everything
        // past a fixed guess. Options.NameColumnWidth acts as a floor, not a ceiling.
        int nameWidth = Math.Max(Options.NameColumnWidth, rows.Count == 0 ? 0 : rows.Max(r => r.TreeLabel.Length));

        var columns = new List<(string, int, Func<TreeRow, string>)>
        {
            ("Layer (type)", nameWidth, r => r.TreeLabel),
        };

        if (Options.ShowInputShape)
            columns.Add(("Input Shape", 22, r => FormatShapes(r.Layer.InputShapes)));
        if (Options.ShowOutputShape)
            columns.Add(("Output Shape", 22, r => r.Layer.OutputShape is null ? "--" : FormatShape(r.Layer.OutputShape)));
        if (Options.ShowParamCount)
        {
            columns.Add(("Param #", 16, r => r.Layer.TotalParams == 0 ? "--" : r.Layer.TotalParams.ToString("N0")));
            columns.Add(("Trainable", 9, r => TrainableLabel(r.Layer)));
        }
        if (Options.ShowMacs)
            columns.Add(("Mult-Adds", 16, r => r.Layer.Macs == 0 ? "--" : r.Layer.Macs.ToString("N0")));

        return columns;
    }

    private static string TrainableLabel(LayerInfo layer) => (layer.TrainableParams, layer.NonTrainableParams) switch
    {
        (0, 0) => "--",
        (> 0, 0) => "Yes",
        (0, > 0) => "No",
        _ => "Mixed",
    };

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

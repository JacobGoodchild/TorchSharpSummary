namespace TorchSharpSummary;

/// <summary>
/// Diagnostic information captured for a single <c>torch.nn.Module</c> instance
/// while it was executed during a dry-run forward pass: its position in the
/// module tree, the tensor shapes that flowed through it, and its parameter /
/// memory footprint.
/// </summary>
/// <remarks>
/// Instances are produced internally by <see cref="ModuleExtensions"/> while
/// walking a leak-free dry-run pass wrapped in a native dispose scope. Only
/// plain values (strings, arrays of <see cref="long"/>, numeric counters) are
/// retained here — no live <c>Tensor</c> handles — so a <see cref="LayerInfo"/>
/// remains safe to inspect long after the underlying native tensors have been
/// disposed.
/// </remarks>
public sealed class LayerInfo
{
    /// <summary>The module's name within its parent (e.g. "conv1"), or its dotted path for nested modules (e.g. "block.conv1").</summary>
    public required string Name { get; init; }

    /// <summary>The module's CLR type name (e.g. "Conv2d", "Linear", "ReLU").</summary>
    public required string LayerType { get; init; }

    /// <summary>Nesting depth within the module tree. The top-level model's direct children are at depth 0.</summary>
    public required int Depth { get; init; }

    /// <summary>The order in which this module was actually invoked during the forward pass (0-based).</summary>
    public required int ExecutionOrder { get; init; }

    /// <summary>The shape of each tensor argument passed into this module's forward call.</summary>
    public IReadOnlyList<long[]> InputShapes { get; init; } = Array.Empty<long[]>();

    /// <summary>The shape of the tensor this module returned, or <c>null</c> if it could not be captured.</summary>
    public long[]? OutputShape { get; init; }

    /// <summary>Number of parameters directly owned by this module (not its children) with <c>requires_grad == true</c>.</summary>
    public long TrainableParams { get; internal set; }

    /// <summary>Number of parameters directly owned by this module (not its children) with <c>requires_grad == false</c>.</summary>
    public long NonTrainableParams { get; internal set; }

    /// <summary>Total parameters directly owned by this module (<see cref="TrainableParams"/> + <see cref="NonTrainableParams"/>).</summary>
    public long TotalParams => TrainableParams + NonTrainableParams;

    /// <summary>Estimated native memory, in bytes, occupied by this module's own parameters.</summary>
    public long ParamBytes { get; internal set; }

    /// <summary>Estimated native memory, in bytes, occupied by the output tensor this module produced.</summary>
    public long OutputBytes { get; internal set; }

    /// <summary>
    /// Estimated multiply-accumulate operations (MACs) performed by this module for the dry-run input.
    /// Precisely computed for <c>Linear</c> and convolution layers; <c>0</c> for layer types this
    /// library does not yet model (e.g. activations, normalization, pooling).
    /// </summary>
    public long Macs { get; internal set; }
}

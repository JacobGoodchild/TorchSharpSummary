using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpSummary.Tests;

public class SummaryTests
{
    // ---------------------------------------------------------------------
    // Sequential CNN model shape resolution
    // ---------------------------------------------------------------------

    [Fact]
    public void SequentialCnn_ResolvesLayerShapes_InExecutionOrder()
    {
        using var model = Sequential(
            ("conv1", Conv2d(1, 8, kernel_size: 3, stride: 1, padding: 1)),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(kernel_size: 2)),
            ("flatten", Flatten()),
            ("fc1", Linear(8 * 14 * 14, 10)));

        long[] inputShape = { 1, 1, 28, 28 };

        // Ground truth: call the model directly and see what it actually produces.
        using (var scope = torch.NewDisposeScope())
        {
            using var x = zeros(inputShape);
            using var expected = model.call(x);
            Assert.Equal(new long[] { 1, 10 }, expected.shape);
        }

        var summary = model.Summary(inputShape);

        // The root Sequential itself gets one row too (depth -1); the five children below it
        // are what this test is about, so pull them out in their actual execution order.
        var childLayers = summary.Layers.Where(l => l.Depth >= 0).ToList();

        Assert.Equal(6, summary.Layers.Count);
        Assert.Equal(5, childLayers.Count);
        Assert.Equal(new[] { "Conv2d", "ReLU", "MaxPool2d", "Flatten", "Linear" },
            childLayers.Select(l => l.LayerType).ToArray());
        Assert.Equal(new[] { "conv1", "relu1", "pool1", "flatten", "fc1" },
            childLayers.Select(l => l.Name).ToArray());

        // conv1: same spatial size (padding=1, stride=1, kernel=3) but 8 channels.
        Assert.Equal(new long[] { 1, 8, 28, 28 }, childLayers[0].OutputShape);
        // pool1: kernel=2 halves the spatial dims.
        Assert.Equal(new long[] { 1, 8, 14, 14 }, childLayers[2].OutputShape);
        // flatten: collapses everything after the batch dim.
        Assert.Equal(new long[] { 1, 8 * 14 * 14 }, childLayers[3].OutputShape);
        // fc1: final classifier output, matches the ground-truth forward pass above.
        Assert.Equal(new long[] { 1, 10 }, childLayers[4].OutputShape);

        // conv1's MACs should be computed precisely: out_elements * kernel_area * in_channels.
        Assert.Equal(1L * 8 * 28 * 28 * (3 * 3) * 1, childLayers[0].Macs);
        // fc1's MACs: out_elements * in_features.
        Assert.Equal(1L * 10 * (8 * 14 * 14), childLayers[4].Macs);
    }

    [Fact]
    public void SequentialCnn_HonorsMaxDepth_WhenRenderingTheTable()
    {
        using var block = Sequential(("conv", Conv2d(1, 4, kernel_size: 3, padding: 1)), ("relu", ReLU()));
        using var model = Sequential(("block", block), ("flatten", Flatten()), ("fc", Linear(4 * 8 * 8, 2)));

        var summary = model.Summary(new SummaryOptions { MaxDepth = 0 }, new long[] { 1, 1, 8, 8 });

        // The root plus all five submodules still executed and are counted in the totals...
        Assert.Equal(6, summary.Layers.Count);
        // ...but only depth<=0 rows ("block", "flatten", "fc" and the root) show up in the rendered table.
        string rendered = summary.ToString();
        Assert.Contains("block (Sequential)", rendered);
        Assert.Contains("flatten (Flatten)", rendered);
        Assert.Contains("fc (Linear)", rendered);
        Assert.DoesNotContain("conv (Conv2d)", rendered);
        Assert.DoesNotContain("relu (ReLU)", rendered);
    }

    // ---------------------------------------------------------------------
    // Custom multi-input / branching modules
    // ---------------------------------------------------------------------

    [Fact]
    public void BranchingModule_ResolvesBothInputShapes()
    {
        using var model = new TwoBranchSum(inFeatures: 4, outFeatures: 6);

        var summary = model.Summary(new long[] { 2, 4 }, new long[] { 2, 4 });

        var rootRow = Assert.Single(summary.Layers, l => l.Depth == -1 && l.LayerType == nameof(TwoBranchSum));

        // Both branch layers ran and captured their single input shape correctly.
        var branchA = summary.Layers.Single(l => l.Name == "branchA");
        var branchB = summary.Layers.Single(l => l.Name == "branchB");
        Assert.Equal(new long[] { 2, 4 }, branchA.InputShapes.Single());
        Assert.Equal(new long[] { 2, 6 }, branchA.OutputShape);
        Assert.Equal(new long[] { 2, 4 }, branchB.InputShapes.Single());
        Assert.Equal(new long[] { 2, 6 }, branchB.OutputShape);

        // The root module itself was invoked with two inputs.
        Assert.Equal(2, rootRow.InputShapes.Count);
        Assert.Equal(new long[] { 2, 4 }, rootRow.InputShapes[0]);
        Assert.Equal(new long[] { 2, 4 }, rootRow.InputShapes[1]);
        Assert.Equal(new long[] { 2, 6 }, rootRow.OutputShape);
    }

    // ---------------------------------------------------------------------
    // Parameter count accuracy
    // ---------------------------------------------------------------------

    [Fact]
    public void ParamCounts_MatchDirectParameterEnumeration()
    {
        using var model = Sequential(
            ("fc1", Linear(10, 20)),
            ("fc2", Linear(20, 5)));

        long expectedTotal = model.parameters().Sum(p => p.numel());
        long expectedTrainable = model.parameters().Where(p => p.requires_grad).Sum(p => p.numel());

        var summary = model.Summary(new long[] { 1, 10 });

        Assert.Equal(expectedTotal, summary.TotalParams);
        Assert.Equal(expectedTrainable, summary.TrainableParams);
        Assert.Equal(0, summary.NonTrainableParams);

        // Per-layer breakdown sums back to the same total (no double counting, nothing dropped).
        Assert.Equal(expectedTotal, summary.Layers.Sum(l => l.TotalParams));
    }

    [Fact]
    public void ParamCounts_DistinguishFrozenLayers_AsNonTrainable()
    {
        using var model = Sequential(
            ("fc1", Linear(10, 20)),
            ("fc2", Linear(20, 5)));

        // Freeze fc1: it should be reported as non-trainable.
        foreach (var p in ((Linear)model[0]).parameters())
            p.requires_grad_(false);

        long expectedFrozen = ((Linear)model[0]).parameters().Sum(p => p.numel());
        long expectedTrainable = ((Linear)model[1]).parameters().Sum(p => p.numel());

        var summary = model.Summary(new long[] { 1, 10 });

        Assert.Equal(expectedFrozen, summary.NonTrainableParams);
        Assert.Equal(expectedTrainable, summary.TrainableParams);

        var fc1Row = summary.Layers.Single(l => l.Name == "fc1");
        Assert.Equal(0, fc1Row.TrainableParams);
        Assert.Equal(expectedFrozen, fc1Row.NonTrainableParams);
    }

    // ---------------------------------------------------------------------
    // Deterministic memory cleanup (no native/unmanaged leaks)
    // ---------------------------------------------------------------------

    [Fact]
    public void Summary_LeavesNoLiveTensorsBehind_AfterItReturns()
    {
        using var model = Sequential(
            ("conv1", Conv2d(3, 16, kernel_size: 3, padding: 1)),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(kernel_size: 2)),
            ("flatten", Flatten()),
            ("fc1", Linear(16 * 16 * 16, 10)));

        // Force a clean baseline so earlier tests' tensors don't muddy the count.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long liveBefore = DisposeScopeManager.Statistics.ThreadTotalLiveCount;

        _ = model.Summary(new long[] { 1, 3, 32, 32 });

        GC.Collect();
        GC.WaitForPendingFinalizers();
        long liveAfter = DisposeScopeManager.Statistics.ThreadTotalLiveCount;

        // Every dummy input and every intermediate activation created during the dry run
        // must have been disposed; only the model's own (pre-existing) parameters remain live.
        Assert.Equal(liveBefore, liveAfter);
    }

    [Fact]
    public void Summary_RestoresOriginalTrainingMode_EvenWhenItThrows()
    {
        using var model = Linear(4, 4);
        model.train(true);

        Assert.Throws<ArgumentException>(() => model.Summary(Array.Empty<long[]>()));

        Assert.True(model.training);
    }

    [Fact]
    public void Summary_DoesNotPermanentlyMutate_TrainEvalMode()
    {
        using var model = Sequential(("fc1", Linear(4, 4)));
        model.train(true);
        Assert.True(model.training);

        model.Summary(new long[] { 1, 4 });

        Assert.True(model.training);

        model.eval();
        model.Summary(new long[] { 1, 4 });
        Assert.False(model.training);
    }

    // ---------------------------------------------------------------------
    // Input validation and overload resolution
    // ---------------------------------------------------------------------

    [Fact]
    public void Summary_Throws_WhenModuleIsNull()
    {
        torch.nn.Module? model = null;
        Assert.Throws<ArgumentNullException>(() => ModuleExtensions.Summary(model!, new long[] { 1 }));
    }

    [Fact]
    public void Summary_Throws_WhenNoInputShapesProvided()
    {
        using var model = Linear(4, 4);
        Assert.Throws<ArgumentException>(() => model.Summary());
    }

    [Fact]
    public void Summary_NonGenericOverload_WorksThroughBaseModuleReference()
    {
        using var model = Linear(4, 4);
        torch.nn.Module baseReference = model;

        var summary = baseReference.Summary(new long[] { 1, 4 });

        Assert.Equal(new long[] { 1, 4 }, summary.Layers.Single().OutputShape);
    }

    [Fact]
    public void ToString_ProducesATableWithTotals()
    {
        using var model = Sequential(("fc1", Linear(4, 8)), ("fc2", Linear(8, 2)));

        var summary = model.Summary(new long[] { 1, 4 });
        string rendered = summary.ToString();

        Assert.Contains("Model: Sequential", rendered);
        Assert.Contains("fc1 (Linear)", rendered);
        Assert.Contains("fc2 (Linear)", rendered);
        Assert.Contains($"Total params: {summary.TotalParams:N0}", rendered);
        Assert.Contains("Estimated Total Size", rendered);
    }
}

/// <summary>
/// A minimal branching module used to exercise the two-tensor-input hook path:
/// two independent Linear branches whose outputs are summed together.
/// </summary>
internal sealed class TwoBranchSum : torch.nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Linear branchA;
    private readonly Linear branchB;

    public TwoBranchSum(long inFeatures, long outFeatures) : base(nameof(TwoBranchSum))
    {
        branchA = Linear(inFeatures, outFeatures);
        branchB = Linear(inFeatures, outFeatures);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x1, Tensor x2) => branchA.call(x1) + branchB.call(x2);
}

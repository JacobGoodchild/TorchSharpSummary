# TorchSharpSummary

A lightweight, leak-free model summary and diagnostic utility for [TorchSharp](https://github.com/dotnet/TorchSharp) — the .NET equivalent of Python's [`torchinfo`](https://github.com/TylerYep/torchinfo) (formerly `pytorch-summary`).

Point it at any `torch.nn.Module` and a set of input shapes, and it runs a safe dry-run forward pass, then prints a layer-by-layer breakdown of output shapes, parameter counts (trainable vs. non-trainable), estimated memory footprint, and MAC/FLOP estimates — without leaking any native libtorch memory and without touching your model's weights or its train/eval mode.

```
Model: Sequential
----------------------------------------------------------------------------------------------------------------------------------
Layer (type)                             Input Shape            Output Shape           Param #          Trainable Mult-Adds
====================================================================================================================================
Sequential                               [1, 1, 28, 28]         [1, 10]                --               --        --
├─ conv1 (Conv2d)                        [1, 1, 28, 28]         [1, 8, 28, 28]         80               Yes       56,448
├─ relu1 (ReLU)                          [1, 8, 28, 28]         [1, 8, 28, 28]         --               --        --
├─ pool1 (MaxPool2d)                     [1, 8, 28, 28]         [1, 8, 14, 14]         --               --        --
├─ flatten (Flatten)                     [1, 8, 14, 14]         [1, 1568]              --               --        --
└─ fc1 (Linear)                          [1, 1568]              [1, 10]                15,690           Yes       15,680
====================================================================================================================================
Total params: 15,770
Trainable params: 15,770
Non-trainable params: 0
Total mult-adds (MACs): 72,128
------------------------------------------------------------------------------------------------------------------------------------
Input size (MB): 0.00
Forward pass size (MB): 0.06
Params size (MB): 0.06
Estimated Total Size (MB): 0.12
====================================================================================================================================
```

Nested modules render as an actual tree (not just a flat, same-indent list), and a frozen sub-module (`requires_grad_(false)`) shows up clearly in the **Trainable** column:

```
Sequential
├─ block (Sequential)
│  ├─ conv (Conv2d)                      [1, 1, 28, 28]         [1, 8, 28, 28]         80               No        56,448
│  ├─ bn (BatchNorm2d)                   [1, 8, 28, 28]         [1, 8, 28, 28]         16               No        6,272
│  └─ relu (ReLU)                        [1, 8, 28, 28]         [1, 8, 28, 28]         --               --        --
├─ pool (MaxPool2d)                      [1, 8, 28, 28]         [1, 8, 14, 14]         --               --        --
└─ fc1 (Linear)                          [1, 1568]              [1, 64]                100,416          Yes       100,352
```

## Why

TorchSharp wraps native (C++) libtorch tensors and modules through unmanaged handles. Running an exploratory forward pass just to see what shape comes out the other end is trickier than it should be in .NET: it's easy to leak native memory if the intermediate tensors aren't disposed deterministically, and there's no built-in way to see a model's shape/parameter breakdown the way `torchinfo.summary()` gives you in Python. TorchSharpSummary fills that gap.

## Install

```bash
dotnet add package TorchSharpSummary
```

You'll also need TorchSharp itself and a native backend, if your project doesn't already reference them:

```bash
dotnet add package TorchSharp
dotnet add package TorchSharp-cpu   # or TorchSharp-cuda-* for a GPU backend
```

## Usage

```csharp
using TorchSharp;
using TorchSharpSummary;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using var model = Sequential(
    ("conv1", Conv2d(1, 8, kernel_size: 3, stride: 1, padding: 1)),
    ("relu1", ReLU()),
    ("pool1", MaxPool2d(kernel_size: 2)),
    ("flatten", Flatten()),
    ("fc1", Linear(8 * 14 * 14, 10)));

// One shape per input tensor forward() expects — most models take just one.
var summary = model.Summary(new long[] { 1, 1, 28, 28 });

summary.Print();                    // writes the table above to the console
Console.WriteLine(summary.TotalParams);
Console.WriteLine(summary.TrainableParams);
Console.WriteLine(summary.EstimatedTotalBytes);

foreach (var layer in summary.Layers)
    Console.WriteLine($"{layer.Name}: {layer.LayerType} -> {string.Join(",", layer.OutputShape!)}");
```

`Summary()` is an extension method with both a generic and non-generic form, so it works whether you're holding a strongly-typed model or a plain `torch.nn.Module` reference:

```csharp
public static ModelSummary Summary(this torch.nn.Module module, params long[][] inputShapes);
public static ModelSummary Summary<T>(this T module, params long[][] inputShapes) where T : torch.nn.Module;
```

### Models with more than one input

A branching/merge module whose `forward` takes two (or three) tensors just gets two (or three) shapes:

```csharp
public class TwoBranchSum : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Linear branchA, branchB;
    public TwoBranchSum(long inFeatures, long outFeatures) : base(nameof(TwoBranchSum))
    {
        branchA = Linear(inFeatures, outFeatures);
        branchB = Linear(inFeatures, outFeatures);
        RegisterComponents();
    }
    public override Tensor forward(Tensor x1, Tensor x2) => branchA.call(x1) + branchB.call(x2);
}

using var model = new TwoBranchSum(inFeatures: 4, outFeatures: 6);
var summary = model.Summary(new long[] { 2, 4 }, new long[] { 2, 4 });
```

### Configuring the output

Pass a `SummaryOptions` to control depth, memory units, and which columns are shown:

```csharp
var options = new SummaryOptions
{
    MaxDepth = 1,                    // only show the top level(s) of nested modules
    MemoryUnit = MemoryUnit.KB,      // Bytes | KB | MB | GB
    ShowMacs = false,                // hide the Mult-Adds column
    ShowMemory = false,               // hide the Input/Forward/Params/Total size section
};

var summary = model.Summary(options, new long[] { 1, 1, 28, 28 });
```

## What you get back

`Summary()` returns a `ModelSummary`:

| Member | Meaning |
|---|---|
| `Layers` | `IReadOnlyList<LayerInfo>`, one entry per module that actually executed during the dry run (in execution order), plus the root model itself |
| `TotalParams` / `TrainableParams` / `NonTrainableParams` | Parameter counts across the whole model, computed from `requires_grad` |
| `ParamBytes` | Estimated native memory occupied by parameters |
| `InputBytes` / `ActivationBytes` | Estimated native memory for the dummy inputs and every intermediate activation produced |
| `EstimatedTotalBytes` | `ParamBytes + InputBytes + ActivationBytes` |
| `TotalMacs` | Sum of every layer's estimated multiply-accumulate operations |

Each `LayerInfo` carries `Name`, `LayerType`, `Depth`, `ExecutionOrder`, `InputShapes`, `OutputShape`, `TrainableParams`/`NonTrainableParams`, `ParamBytes`, `OutputBytes`, and `Macs`.

## Safety guarantees

- **No leaks.** Every tensor created during the dry run — the dummy inputs and every intermediate activation — is created inside a single `torch.NewDisposeScope()` and is deterministically disposed the moment the pass completes, success or failure.
- **No mutation.** Your model's parameters are never modified. Any forward hooks TorchSharpSummary registers to observe the pass are removed again before `Summary()` returns, and your model's original `training`/`eval` mode is restored afterward even if the pass throws.

## Coverage & limitations (v0.1)

- Per-layer shape capture works for modules with one, two, or three `Tensor` inputs and a single `Tensor` output — i.e. `Module<Tensor,Tensor>`, `Module<Tensor,Tensor,Tensor>`, and `Module<Tensor,Tensor,Tensor,Tensor>`. This covers essentially every built-in TorchSharp layer and typical branching modules. A module with a different forward signature (four-plus tensor inputs, non-tensor arguments, or a tuple return) won't get its own row, but its parameters are still counted correctly in the model-level totals.
- MAC/FLOP estimates are computed precisely for `Linear`, convolution (`Conv1d`/`Conv2d`/`Conv3d`, including grouped convolutions), and normalization layers (`BatchNorm*`, `InstanceNorm*`, `LayerNorm`, `GroupNorm` — counting their affine scale/shift step). Other layer types report `0` MACs (shown as `--`) — activations, pooling, dropout, and recurrent (RNN/LSTM/GRU) layers aren't yet modeled.
- The top-level model must expose a public `call(Tensor, ...)` method matching the number of input shapes passed in — true for any standard `torch.nn.Module<...>` subclass.
- A submodule invoked more than once in a single forward pass (weight sharing, or the same layer called in a loop) gets one row per call, each with its own children — never merged. Only the first occurrence carries that layer's parameter/memory counts (its weights are shared, not duplicated, so counting them again would inflate the totals); later occurrences are labeled `(recursive)`, matching `torchinfo`'s convention.

## Building from source

```bash
dotnet build
dotnet test
dotnet pack -c Release
```

## License

MIT

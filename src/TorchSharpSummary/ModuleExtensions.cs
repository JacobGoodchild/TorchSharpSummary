using System.Reflection;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TorchSharpSummary;

/// <summary>
/// Extension methods that run a leak-free dry-run forward pass over a TorchSharp
/// <c>torch.nn.Module</c> and produce a <see cref="ModelSummary"/> describing the shapes,
/// parameter counts, memory footprint, and MAC estimates of every layer that executes —
/// the TorchSharp equivalent of Python's <c>torchinfo.summary()</c>.
/// </summary>
/// <remarks>
/// <para>
/// The dry-run pass creates zero-filled dummy input tensors, invokes the model, and inspects
/// what comes back. Every tensor the pass creates — the dummy inputs and every intermediate
/// activation — is created inside a single <c>torch.NewDisposeScope()</c>, so all of that
/// native (libtorch) memory is deterministically released as soon as the pass completes,
/// regardless of whether it throws. Only plain, managed data (shapes as <c>long[]</c>,
/// counters) survives into the returned <see cref="ModelSummary"/>.
/// </para>
/// <para>
/// The model itself is never mutated: its trained parameters are left untouched, any hooks
/// registered to observe the pass are removed again before <c>Summary</c> returns, and the
/// model's original <c>training</c>/<c>eval</c> mode is restored even if the pass fails.
/// </para>
/// <para>
/// <b>Coverage.</b> Per-layer shape capture relies on casting each submodule to
/// <c>Module&lt;Tensor, Tensor&gt;</c>, <c>Module&lt;Tensor, Tensor, Tensor&gt;</c>, or
/// <c>Module&lt;Tensor, Tensor, Tensor, Tensor&gt;</c> — i.e. modules with one, two, or three
/// tensor inputs and a single tensor output, which covers the overwhelming majority of
/// TorchSharp layers and typical branching/merge modules. A submodule with a different
/// forward signature (four or more tensor inputs, non-tensor arguments, or a tuple return)
/// is simply not hooked: its own row will not appear in the table, but its parameters are
/// still counted correctly in the model-level totals. MAC/FLOP estimates are computed
/// precisely for <c>Linear</c>, convolution, and normalization layers (see
/// <see cref="LayerInfo.Macs"/>); other layer types (activations, pooling, dropout,
/// recurrent layers) report <c>0</c>.
/// </para>
/// </remarks>
public static class ModuleExtensions
{
    /// <summary>
    /// Runs a leak-free dry-run forward pass over <paramref name="module"/> using zero-filled
    /// dummy tensors of the given <paramref name="inputShapes"/>, and returns a layer-by-layer
    /// summary (shapes, parameter counts, memory footprint, MAC estimates).
    /// </summary>
    /// <param name="module">The model to summarize.</param>
    /// <param name="inputShapes">
    /// One shape per positional tensor argument the model's <c>forward</c> method expects.
    /// Most models take a single input, so most callers pass a single shape, e.g.
    /// <c>model.Summary(new long[] { 1, 3, 224, 224 })</c>. A model whose <c>forward</c> takes
    /// two tensors (a branching/merge module) takes two shapes, and so on.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="module"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="inputShapes"/> is empty.</exception>
    /// <exception cref="InvalidOperationException">
    /// <paramref name="module"/> does not expose a public <c>call</c> method whose arity matches
    /// <paramref name="inputShapes"/>.
    /// </exception>
    /// <exception cref="NotSupportedException">
    /// <paramref name="module"/>'s forward pass does not return a single <see cref="Tensor"/>.
    /// </exception>
    public static ModelSummary Summary(this torch.nn.Module module, params long[][] inputShapes) =>
        BuildSummary(module, new SummaryOptions(), inputShapes);

    /// <summary>Overload of <see cref="Summary(torch.nn.Module, long[][])"/> that accepts explicit <see cref="SummaryOptions"/>.</summary>
    public static ModelSummary Summary(this torch.nn.Module module, SummaryOptions options, params long[][] inputShapes) =>
        BuildSummary(module, options, inputShapes);

    /// <summary>
    /// Generic convenience overload of <see cref="Summary(torch.nn.Module, long[][])"/> for callers
    /// holding a strongly-typed module reference (e.g. a concrete <c>Sequential</c> or a custom
    /// <c>Module&lt;Tensor, Tensor&gt;</c> subclass).
    /// </summary>
    public static ModelSummary Summary<T>(this T module, params long[][] inputShapes)
        where T : torch.nn.Module =>
        BuildSummary(module, new SummaryOptions(), inputShapes);

    /// <summary>Generic overload of <see cref="Summary{T}(T, long[][])"/> that accepts explicit <see cref="SummaryOptions"/>.</summary>
    public static ModelSummary Summary<T>(this T module, SummaryOptions options, params long[][] inputShapes)
        where T : torch.nn.Module =>
        BuildSummary(module, options, inputShapes);

    private static ModelSummary BuildSummary(torch.nn.Module module, SummaryOptions options, long[][] inputShapes)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(options);
        if (inputShapes is null || inputShapes.Length == 0)
            throw new ArgumentException("At least one input shape must be provided.", nameof(inputShapes));

        bool wasTraining = module.training;
        var hookRemovers = new List<Action>();
        var executed = new List<LayerInfo>();

        try
        {
            // Dropout/BatchNorm etc. must run deterministically for a shape/size dry run.
            module.eval();

            // The root module is hooked too (depth -1, so it always renders regardless of
            // SummaryOptions.MaxDepth) — this is the only way a model that is itself a leaf
            // module (e.g. a bare Linear used directly as the root) gets a row at all, since
            // named_modules() below only enumerates its *children*. Its Name is the empty
            // string, matching the owner key AttachParameterCounts computes for a parameter
            // with no dotted prefix (e.g. root.weight) — so a leaf root's own parameters land
            // on its row instead of silently going unassigned.
            RegisterHook(module, name: string.Empty, depth: -1, executed, hookRemovers);

            foreach (var (name, submodule) in module.named_modules())
            {
                int depth = name.Count(c => c == '.');
                RegisterHook(submodule, name, depth, executed, hookRemovers);
            }

            long inputBytes;
            using (torch.NewDisposeScope())
            {
                // Every tensor created in this block — the dummy inputs and every intermediate
                // activation the forward pass produces — is torn down deterministically the
                // moment this block exits, leak-free, regardless of the branch taken.
                var inputs = inputShapes
                    .Select(shape => torch.zeros(shape, dtype: options.InputDType, device: options.Device))
                    .ToArray();
                inputBytes = inputs.Sum(t => t.numel() * t.element_size());

                InvokeForward(module, inputs);
            }

            AttachParameterCounts(module, executed);

            long trainableParams = 0, nonTrainableParams = 0, paramBytes = 0;
            foreach (var (_, parameter) in module.named_parameters())
            {
                long n = parameter.numel();
                if (parameter.requires_grad) trainableParams += n; else nonTrainableParams += n;
                paramBytes += n * parameter.element_size();
            }

            return new ModelSummary
            {
                ModelName = module.GetType().Name,
                Layers = executed,
                TrainableParams = trainableParams,
                NonTrainableParams = nonTrainableParams,
                ParamBytes = paramBytes,
                InputBytes = inputBytes,
                ActivationBytes = executed.Sum(l => l.OutputBytes),
                TotalMacs = executed.Sum(l => l.Macs),
                Options = options,
            };
        }
        finally
        {
            // Never leave a permanent hook registered on the caller's model.
            foreach (var remove in hookRemovers)
                remove();

            // Never leave the caller's model in a different train/eval mode than we found it.
            module.train(wasTraining);
        }
    }

    /// <summary>
    /// Invokes <paramref name="module"/>'s forward pass via its public <c>call</c> method
    /// (TorchSharp's hook-triggering entry point, analogous to PyTorch's <c>__call__</c>),
    /// located by reflection since the module's static type may only be the non-generic
    /// <c>torch.nn.Module</c> base.
    /// </summary>
    private static void InvokeForward(torch.nn.Module module, Tensor[] inputs)
    {
        var concreteType = module.GetType();
        var candidates = concreteType
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.Name == "call"
                        && m.GetParameters().Length == inputs.Length
                        && m.GetParameters().All(p => p.ParameterType == typeof(Tensor)))
            .ToList();

        var callMethod = candidates.FirstOrDefault(m => m.ReturnType == typeof(Tensor)) ?? candidates.FirstOrDefault();

        if (callMethod is null)
        {
            throw new InvalidOperationException(
                $"'{concreteType.FullName}' does not expose a public 'call' method accepting " +
                $"{inputs.Length} Tensor argument(s). Make sure it implements torch.nn.Module<...> " +
                "with an arity matching the number of input shapes passed to Summary(), and that " +
                "inputShapes.Length equals the number of tensor arguments forward() expects.");
        }

        if (callMethod.ReturnType != typeof(Tensor))
        {
            throw new NotSupportedException(
                $"'{concreteType.FullName}.call' returns '{callMethod.ReturnType.Name}', but " +
                "TorchSharpSummary currently only supports modules whose forward pass returns a single Tensor.");
        }

        callMethod.Invoke(module, inputs.Cast<object>().ToArray());
    }

    /// <summary>
    /// Attaches a temporary forward hook to <paramref name="submodule"/>, if its forward
    /// signature is one this library knows how to observe, so that every time it actually
    /// executes during the dry run a <see cref="LayerInfo"/> row is recorded.
    /// </summary>
    private static void RegisterHook(
        torch.nn.Module submodule,
        string name,
        int depth,
        List<LayerInfo> executed,
        List<Action> hookRemovers)
    {
        switch (submodule)
        {
            case torch.nn.Module<Tensor, Tensor> single:
            {
                var remover = single.register_forward_hook((_, input, output) =>
                {
                    executed.Add(CreateLayerInfo(submodule, name, depth, executed.Count, new[] { input }, output));
                    return output;
                });
                hookRemovers.Add(() => remover.remove());
                break;
            }
            case torch.nn.Module<Tensor, Tensor, Tensor> dual:
            {
                var remover = dual.register_forward_hook((_, input1, input2, output) =>
                {
                    executed.Add(CreateLayerInfo(submodule, name, depth, executed.Count, new[] { input1, input2 }, output));
                    return output;
                });
                hookRemovers.Add(() => remover.remove());
                break;
            }
            case torch.nn.Module<Tensor, Tensor, Tensor, Tensor> triple:
            {
                var remover = triple.register_forward_hook((_, input1, input2, input3, output) =>
                {
                    executed.Add(CreateLayerInfo(submodule, name, depth, executed.Count, new[] { input1, input2, input3 }, output));
                    return output;
                });
                hookRemovers.Add(() => remover.remove());
                break;
            }

            // A submodule with a different forward arity (4+ tensor inputs, non-tensor
            // arguments, or a tuple return) cannot be hooked generically and is skipped —
            // it will not appear as its own row, but its parameters are still counted in
            // the model-level totals via named_parameters().
        }
    }

    private static LayerInfo CreateLayerInfo(
        torch.nn.Module submodule,
        string name,
        int depth,
        int executionOrder,
        Tensor[] inputs,
        Tensor output)
    {
        return new LayerInfo
        {
            Name = name,
            LayerType = submodule.GetType().Name,
            Depth = depth,
            ExecutionOrder = executionOrder,
            InputShapes = inputs.Select(t => t.shape).ToArray(),
            OutputShape = output.shape,
            OutputBytes = output.numel() * output.element_size(),
            Macs = EstimateMacs(submodule, output.shape),
        };
    }

    /// <summary>
    /// Estimates multiply-accumulate operations (MACs) for one execution of <paramref name="submodule"/>
    /// given the output shape it produced. Precisely computed for <see cref="Linear"/>, convolution
    /// layers (<see cref="Convolution"/>: <c>Conv1d</c>/<c>Conv2d</c>/<c>Conv3d</c>, including grouped
    /// convolutions), and the affine step of normalization layers (<see cref="NormBase"/> — i.e.
    /// <c>BatchNorm1d</c>/<c>2d</c>/<c>3d</c>/<c>InstanceNorm*</c> — plus <see cref="LayerNorm"/> and
    /// <see cref="GroupNorm"/>). Every other layer type reports <c>0</c> (rendered as "--") — activation,
    /// pooling, dropout, and recurrent (RNN/LSTM/GRU) layers are not yet modeled.
    /// FLOPs are conventionally ~2x MACs for a multiply-then-add.
    /// </summary>
    private static long EstimateMacs(torch.nn.Module submodule, long[] outputShape)
    {
        long outputElements = Product(outputShape);

        if (submodule is Linear linear)
            return outputElements * linear.in_features;

        if (submodule is Convolution conv)
        {
            long kernelVolume = Product(conv.kernel_size);
            long inChannelsPerGroup = conv.groups > 0 ? conv.in_channels / conv.groups : conv.in_channels;
            return outputElements * kernelVolume * inChannelsPerGroup;
        }

        // Normalization layers: the affine step (y = x * weight + bias) is one multiply-add
        // per output element. The normalization statistics themselves (mean/variance) are not
        // counted — they dominate compute far less than the affine transform for typical shapes.
        if (submodule is NormBase norm)
            return norm.affine ? outputElements : 0;

        if (submodule is LayerNorm layerNorm)
            return layerNorm.elementwise_affine ? outputElements : 0;

        if (submodule is GroupNorm groupNorm)
            return groupNorm.affine ? outputElements : 0;

        return 0;
    }

    private static long Product(long[] shape) => shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>
    /// Groups every parameter in <paramref name="module"/> by its immediate owning submodule
    /// (the dotted name up to the last '.') and assigns each group's trainable/non-trainable
    /// counts and byte size to that module's <see cref="LayerInfo"/> row. If a module executed
    /// more than once during the dry run (e.g. a shared/recurrent submodule), only its first
    /// row receives the counts, so totals are never double-counted.
    /// </summary>
    private static void AttachParameterCounts(torch.nn.Module module, List<LayerInfo> executed)
    {
        var ownerGroups = new Dictionary<string, List<Parameter>>();
        foreach (var (paramName, parameter) in module.named_parameters())
        {
            int lastDot = paramName.LastIndexOf('.');
            string owner = lastDot < 0 ? string.Empty : paramName[..lastDot];
            if (!ownerGroups.TryGetValue(owner, out var list))
                ownerGroups[owner] = list = new List<Parameter>();
            list.Add(parameter);
        }

        var assigned = new HashSet<string>();
        foreach (var layer in executed)
        {
            if (!assigned.Add(layer.Name)) continue;
            if (!ownerGroups.TryGetValue(layer.Name, out var owned)) continue;

            foreach (var parameter in owned)
            {
                long n = parameter.numel();
                if (parameter.requires_grad) layer.TrainableParams += n; else layer.NonTrainableParams += n;
                layer.ParamBytes += n * parameter.element_size();
            }
        }
    }
}

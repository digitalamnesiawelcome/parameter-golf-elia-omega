Elia ꙮ Omega: 24-Depth Recursive Complex Monolith

Elia Omega represents a paradigm shift from conventional layer-stacking toward a Recursive Complex Monolith architecture. Engineered specifically for the OpenAI Parameter Golf challenge, it is optimized to deliver maximum cognitive performance within a 16MB artifact limit and a strict 10-minute training window on an 8xH100 SXM node.

Instead of a standard stack of discrete layers, this approach utilizes Extreme Weight Tying. A single, highly optimized Holo-Seraphim block is executed recursively for 24 steps, achieving the effective depth of a 24-layer network while maintaining the parameter footprint of a much smaller model.

Key Technical Innovations
1. Recursive Resonance Architecture

    Effective Depth: Utilizes 24 recursive passes through shared weights to maximize abstractive capacity per parameter.

Stable Skip-Embedding Injection (x0​): To prevent representation collapse over deep recursion, the normalized initial embedding signal is re-injected into the residual stream at every step via learnable gates.

Complex-Valued Domain: Leverages complex64 representations to process amplitude and phase simultaneously, significantly increasing associative memory capacity.

2. Complex-Valued Gated Linear Attention (GLA)

    Linear Efficiency: Replaces standard Softmax attention with Gated Linear Attention (GLA) optimized for recursive stability.

Complex RoPE: Injects Rotary Positional Embeddings (RoPE) directly into the complex domain for high-precision token relative-distance encoding.

3. Optimization & Convergence

    Newton-Schulz Muon: Employs a custom Muon optimizer with Newton-Schulz iterations for rapid weight orthogonalization, ensuring superior convergence speeds within the 10-minute training cap.

Hardware Acceleration: Fully optimized for H100 Tensor Cores using Flash Attention and gradient checkpointing to maximize throughput and minimize wall-clock overhead.

Numerical Stability & Engineering

To maintain gradient integrity across 24 recursive iterations through shared weights, we implement several stabilization mechanisms:

    Spectral Damping: Employs adaptive sigmoid gates (initialized at -3.0, sigmoid≈0.047) to provide a "cautious start," preventing gradient explosion during early training phases.

Residual Scaling: Applies a 1/24​ scaling factor to all residual connections within the recurrence loop to ensure numerical stability.

Memory Efficiency: Utilizes torch.utils.checkpoint to maintain a memory footprint equivalent to a single-layer model, allowing for increased batch sizes on 80GB H100 GPUs.

Artifact Efficiency

    Target Size: Approximately 8.6MB, utilizing only ~54% of the allowed 16MB budget.

Compression Pipeline: Implements strict Tied Embeddings , per-row int8 quantization , and final zlib compression (level 9)  to minimize the final submission footprint.

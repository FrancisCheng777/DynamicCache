# Source provenance

- Base repository: https://github.com/Robbyant/lingbot-va
- Base commit: `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- Base license: Apache-2.0, retained verbatim in `LICENSE.txt`.
- The original README is retained as `README_UPSTREAM.md`.
- Original copyright notices and the upstream Git history are retained.
- Method reference: [C³ache: Accelerating World Action Models with Cross Inference Chunk Cache](https://arxiv.org/abs/2606.08962).

DynamicCache independently implements the paper's whole-stack residual reuse rule
and adapts the execution boundaries to LingBot-VA. It is not the authors' official
C³ache code. No pretrained parameters, training data, private FastWAM code, or
third-party unpublished implementations are distributed here.

The upstream transformer and inference server carry modification notices. New
cache, evaluation, documentation, and test files in this fork are provided under
the repository's Apache-2.0 license. Upstream licenses inside
`wan_va/utils/Simple_Remote_Infer/` remain in place.

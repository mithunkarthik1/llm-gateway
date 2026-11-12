---
base_model: onnx-community/Qwen2.5-Coder-0.5B-Instruct
tags:
  - onnx
  - qwen2
  - text-generation
  - quantized
---

# Qwen2.5-Coder-0.5B-Instruct (tied-embedding q4, fully quantized)

This is a derivative of [onnx-community/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct), an ONNX export of [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).

Architecture, tokenizer, and generation config are unchanged from the base model — only the weight quantization differs.

## What's different

Qwen2.5-Coder-0.5B-Instruct uses tied input/output embeddings (`tie_word_embeddings: true`). The base repo's `q4` export quantizes the rest of the model but leaves `embed_tokens` unquantized at FP32 (~519MB), because the default quantization config excludes `Gather` ops from quantization.

This export re-quantizes with `op_types_to_quantize` including `"Gather"`, so `embed_tokens` is quantized along with everything else. Total on-disk size drops from ~862MB to ~390MB, with no observed quality regression on manual spot checks (fibonacci / debounce / palindrome completions).

## Files

- `onnx/model_q4.onnx` — graph file
- `onnx/model_q4.onnx_data` — external weight data (ONNX external-data format, referenced by `onnx/model_q4.onnx`)

## Usage

Same as the base model — e.g. with [🤗 Transformers.js](https://huggingface.co/docs/transformers.js):

```js
import { pipeline } from '@huggingface/transformers';

const generator = await pipeline('text-generation', 'Vinay121A/Qwen2.5-Coder-0.5B-Instruct-tied-embed-q4', {
  dtype: 'q4',
});
```

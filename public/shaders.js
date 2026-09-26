// shaders.js (T135): the WGSL of this project in one place. The GPU section of /benchmark/ (public/benchmark/gpu.js,
// T134) measures with some of them; the model's GPU worker (public/gpu.js) runs a prompt's tokens through the layers
// with the batched matrix and the steps of a layer below it. A plain ES module: both import it with the ?v= of their
// own URL (GitHub Pages keeps a file for ten minutes: all must come from the same deployment).
//
// The weights are this project's int8: values in groups of GROUP with one float32 scale each (llama2_numpy's layout),
// four values to a u32 as the shaders read them. Every matrix times vector: one workgroup of 64 per row, each thread a
// word (4 weights) at a time with the stride of the workgroup, so that neighbours read neighbouring words; the partial
// sums add up in the workgroup's memory. Rows past the most workgroups of a dimension go to its second one.

export const GROUP = 32;
// the tokens a workgroup of the batched matrix multiplies at once: each weight is read once for all of them
export const TILE = 8;

// the tokens of a request and the position of the first, written once per request: every step of a layer reads it
const STEP = /* wgsl */ `struct Step { tokens: u32, pos: u32, unused0: u32, unused1: u32 }`;

// a matrix times one vector, the weights widened to float32 on the way (the benchmark's)
export const WIDEN = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    let word = bitcast<i32>(w[row * shape.words + i]);
    let at = i * 4u;
    let dot = f32(extractBits(word, 0u, 8u)) * x[at] + f32(extractBits(word, 8u, 8u)) * x[at + 1u]
            + f32(extractBits(word, 16u, 8u)) * x[at + 2u] + f32(extractBits(word, 24u, 8u)) * x[at + 3u];
    sum += dot * scales[row * shape.perRow + i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;

// the same with the activations quantized to int8 as the CPU's matmul_q8 takes them (a float32 scale per group of
// 32), and WGSL's packed dot product: where the language feature is there (the benchmark's)
export const PACKED = /* wgsl */ `
requires packed_4x8_integer_dot_product;
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<storage, read> xs: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    sum += f32(dot4I8Packed(w[row * shape.words + i], xq[i])) * scales[row * shape.perRow + i / 8u] * xs[i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;

// A matrix times the tokens of a prompt (step.tokens of them), TILE at a time: one workgroup per row and tile of
// tokens (the third dimension of the dispatch). x holds the tokens xStride floats apart, y their outputs yStride
// apart, from row first on (a matrix cut into chunks of rows). add: the result is added to what y holds (the residual
// stream: x + W·v, as the CPU adds it after its matmul), else it replaces it.
export const BATCHED = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;
const TILE = ${TILE}u;
var<workgroup> partial: array<f32, ${TILE * 64}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  let first = id.z * TILE;
  var sums: array<f32, ${TILE}>;
  for (var i = t; i < shape.words; i += 64u) {
    let word = bitcast<i32>(w[row * shape.words + i]);
    let scale = scales[row * shape.perRow + i / 8u];
    let w0 = f32(extractBits(word, 0u, 8u)) * scale;
    let w1 = f32(extractBits(word, 8u, 8u)) * scale;
    let w2 = f32(extractBits(word, 16u, 8u)) * scale;
    let w3 = f32(extractBits(word, 24u, 8u)) * scale;
    for (var k = 0u; k < TILE; k++) {
      if (first + k < step.tokens) {
        let at = (first + k) * shape.xStride + i * 4u;
        sums[k] += w0 * x[at] + w1 * x[at + 1u] + w2 * x[at + 2u] + w3 * x[at + 3u];
      }
    }
  }
  for (var k = 0u; k < TILE; k++) { partial[k * 64u + t] = sums[k]; }
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) {
      for (var k = 0u; k < TILE; k++) { partial[k * 64u + t] += partial[k * 64u + t + half]; }
    }
    workgroupBarrier();
  }
  if (t < TILE && first + t < step.tokens) {
    let at = (first + t) * shape.yStride + shape.first + row;
    y[at] = select(0.0, y[at], shape.add != 0u) + partial[t * 64u];
  }
}`;

// ---- T146: a matrix times the tokens of a prompt by tiles (the benchmark measures them; the model's GPU worker is to
// take the fastest on each device, T147). BATCHED reads each weight once for TILE tokens but loads an activation for
// every multiply-add, and 64 threads add up every sum. These three take their form from public implementations
// instead (the owner, 2026-09-26: take the best public one rather than invent one):
//   regTile(half): llama.cpp's WebGPU register tiling (mul_mat_reg_tile.wgsl with mul_mat_decls.tmpl's Q8_0 and float
//     loaders). A workgroup owns TILE_M × WORKGROUP_SIZE_M rows by TILE_N × WORKGROUP_SIZE_N tokens; each step of
//     TILE_K = 32 (one group) widens the step's weights (× their scale) and copies the tokens' activations into the
//     workgroup's memory, then each thread multiplies its 4 rows by its 4 tokens with the sums in registers. half: the
//     workgroup's memory holds f16 as llama.cpp's does (shader-f16); the f32 form (no shader-f16) is this project's,
//     llama.cpp's register tiling is f16 only. The sums are f32 either way.
//   tfjsTile: TensorFlow.js's WebGPU makeMatMulPackedVec4Source (matmul_packed_webgpu.ts): the same classic tiles
//     (32 × 32, 8 × 8 threads, 4 × 4 a thread, 32 of the width a step), but the workgroup's memory is read as vec4 and
//     a thread's 4 tokens are one vec4 of sums (fma), where llama.cpp reads scalars (T146's review: a Mali GPU is
//     likely bound by the workgroup memory's reads, and on Apple llama.cpp's src0 rows 512 bytes apart share a bank).
//   dp4a(subgroups): ONNX Runtime Web's DP4A MatMulNBits (dp4a_matmul.wgsl.template, 8 bits, no zero points): a tile
//     of 64 tokens × 64 rows, 256 threads, 32 of the width a step as packed int8 in the workgroup's memory; each thread
//     one token × 16 rows, a group's int sum by dot4I8Packed times the two scales, as the CPU's matmul_q8 sums them.
//     The activations are quantized first (QUANTIZE). subgroups: the same with ORT's subgroupShuffle path where the
//     device's subgroups are 16 wide (Arm Valhall's), which reads the rows from the registers of the subgroup's lanes.
// Changed from the sources for this project's weights, and why: the int8 values and their float32 scales are two
// buffers (llama2_numpy's layout), not Q8_0's 34-byte blocks of f16 scale and 32 values, so the loaders read a group's
// scale from its own buffer; the weights are signed already (ORT's 8-bit ones are unsigned about 128); a group is 32
// for the activations too (ORT's scales_a are per 128); the outputs are written one float at a time, with each row and
// token checked and added to y where shape.add (a matrix cut in chunks of rows of any count, the residual stream),
// where ORT writes a vec4 and asks N % 16 == 0, llama.cpp a vec4 of rows and TensorFlow.js a vec4 of its columns; the workgroups are numbered as each
// source numbers them, over x and then y (a dispatch's dimension holds at most 65535; TensorFlow.js dispatches in two
// dimensions, here numbered as the others). The Shape, the Step and the
// bindings are BATCHED's (dp4a reads x as xq and adds the activations' scales, 6).

// llama.cpp's defaults (ggml-webgpu-shader-lib.hpp: WEBGPU_MUL_MAT_WG_SIZE_M/N 8, TILE_M/N 4, REG_TILE_K_QUANT 32)
// and a workgroup of 256 threads for a tile of 64 tokens (the benchmark measures both; the overrides are the pipeline's)
export const REG_TILES = [{ m: 8, n: 8 }, { m: 16, n: 16 }];
export const regTileShape = ({ m, n }) => ({ rows: 4 * m, tokens: 4 * n, threads: m * n });
// the workgroup's memory of a regTile: a step's weights and activations, 2 or 4 bytes each
export const regTileBytes = ({ m, n }, half) => 32 * 4 * (m + n) * (half ? 2 : 4);
export const DP4A_SHAPE = { rows: 64, tokens: 64, threads: 256 };

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl, mul_mat_decls.tmpl and
// quant_inner_loops.tmpl (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
export const regTile = (half) => /* wgsl */ `${half ? "enable f16;\n" : ""}
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
alias shmem_t = ${half ? "f16" : "f32"};
@group(0) @binding(0) var<storage, read> w: array<u32>;             // M rows, K columns: 4 int8 to a u32
@group(0) @binding(1) var<storage, read> scales: array<f32>;        // a scale a row and group of 32
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;       // N tokens, K columns (xStride floats apart)
@group(0) @binding(3) var<storage, read_write> y: array<f32>;       // N tokens, M rows (yStride floats apart)
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;

override WORKGROUP_SIZE_M: u32 = 8u;
override WORKGROUP_SIZE_N: u32 = 8u;
const TILE_M = 4u;
const TILE_N = 4u;
const TILE_K = 32u;
const BLOCK_SIZE = 32u;
const BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const BYTES_PER_THREAD = 16u;  // NQ(16) weights use 16 bytes of q
const BYTES_PER_INNER_LOOP = 4u;
override TOTAL_WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
override TILE_SRC0_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC1_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_N * TILE_N;
override TILE_SHMEM: u32 = TILE_SRC0_SHMEM + TILE_SRC1_SHMEM;
var<workgroup> shmem: array<shmem_t, TILE_SHMEM>;

fn get_byte_i32(value: u32, index: u32) -> i32 {
  return bitcast<i32>(((value >> (index * 8u)) & 0xFFu) << 24u) >> 24u;
}
// Q8_0's loader: NQ weights a thread, widened with their scale (here a float32 of its own buffer, the product rounded
// once into the memory's type)
fn init_shmem_src0(thread_id: u32, offset_m: u32, k_outer: u32) {
  for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
    let block_idx = i / BLOCK_SIZE;
    let block_offset = (i % BLOCK_SIZE) / NQ;
    let shmem_idx = block_idx * BLOCK_SIZE + block_offset * BYTES_PER_THREAD;
    let tile_m = block_idx / BLOCKS_K;
    let global_m = offset_m + tile_m;
    let block_k = block_idx % BLOCKS_K;
    let global_block_k = k_outer / BLOCK_SIZE + block_k;
    if (global_m < shape.rows && global_block_k < shape.perRow) {
      let d = scales[global_m * shape.perRow + global_block_k];
      for (var j = 0u; j < BYTES_PER_THREAD / BYTES_PER_INNER_LOOP; j += 1u) {
        let q_packed = w[global_m * shape.words + global_block_k * (BLOCK_SIZE / 4u) + (block_offset * BYTES_PER_THREAD) / 4u + j];
        for (var k = 0u; k < 4u; k++) {
          shmem[shmem_idx + j * BYTES_PER_INNER_LOOP + k] = shmem_t(f32(get_byte_i32(q_packed, k)) * d);
        }
      }
    }
  }
}
// the activations' loader, four at a time: llama.cpp's VEC loader, which llama.cpp itself takes only for F32 and F16
// weights (Q8_0 takes its SCALAR one; x here is float32 and four aligned). A token past the request or a column past
// the width reads 0
fn init_shmem_src1(thread_id: u32, offset_n: u32, k_outer: u32) {
  let k = shape.words * 4u;
  for (var elem_idx = thread_id * 4u; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * 4u) {
    let tile_n = elem_idx / TILE_K;
    let tile_k = elem_idx % TILE_K;
    let global_n = offset_n + tile_n;
    let global_k = k_outer + tile_k;
    let src1_idx = global_n * shape.xStride + global_k;
    let src1_val = select(vec4<f32>(0.0), x[src1_idx / 4u], global_n < step.tokens && global_k < k);
    let at = TILE_SRC0_SHMEM + elem_idx;
    shmem[at] = shmem_t(src1_val.x);
    shmem[at + 1u] = shmem_t(src1_val.y);
    shmem[at + 2u] = shmem_t(src1_val.z);
    shmem[at + 3u] = shmem_t(src1_val.w);
  }
}

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_id = local_id.x;
  let local_m = thread_id % WORKGROUP_SIZE_M;
  let local_n = thread_id / WORKGROUP_SIZE_M;

  let wg_n_count = (step.tokens + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
  let wg_m_count = (shape.rows + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
  let wg_linear = wg_id.y * num_wg.x + wg_id.x;
  if (wg_linear >= wg_m_count * wg_n_count) {
    return;
  }
  let wg_m = wg_linear % wg_m_count;
  let wg_n = wg_linear / wg_m_count;

  let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
  let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;
  let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
  let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

  var acc: array<array<f32, TILE_N>, TILE_M>;
  let k = shape.words * 4u;
  for (var k_outer = 0u; k_outer < k; k_outer += TILE_K) {
    init_shmem_src0(thread_id, offset_m, k_outer);
    init_shmem_src1(thread_id, offset_n, k_outer);
    workgroupBarrier();
    let k_end = min(TILE_K, k - k_outer);
    for (var k_inner = 0u; k_inner < k_end; k_inner++) {
      var src0_tile: array<shmem_t, TILE_M>;
      for (var tm = 0u; tm < TILE_M; tm++) {
        let src0_m = local_m * TILE_M + tm;
        let src0_idx = k_inner + src0_m * TILE_K;
        src0_tile[tm] = shmem[src0_idx];
      }
      for (var tn = 0u; tn < TILE_N; tn++) {
        let src1_n = local_n * TILE_N + tn;
        let src1_idx = src1_n * TILE_K + k_inner;
        let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
        for (var tm = 0u; tm < TILE_M; tm++) {
          acc[tm][tn] += f32(src0_tile[tm]) * f32(src1_val);
        }
      }
    }
    workgroupBarrier();
  }

  for (var tn = 0u; tn < TILE_N; tn++) {
    let global_col = output_col_base + tn;
    if (global_col < step.tokens) {
      for (var tm = 0u; tm < TILE_M; tm++) {
        let global_row = output_row_base + tm;
        if (global_row < shape.rows) {
          let at = global_col * shape.yStride + shape.first + global_row;
          y[at] = select(0.0, y[at], shape.add != 0u) + acc[tm][tn];
        }
      }
    }
  }
}`;

// TensorFlow.js's tile: 32 rows × 32 tokens, 8 × 8 threads, 4 rows × 4 tokens a thread
export const TFJS_SHAPE = { rows: 32, tokens: 32, threads: 64 };

// Adapted from TensorFlow.js, tfjs-backend-webgpu/src/matmul_packed_webgpu.ts (makeMatMulPackedVec4Source,
// matMulReadFnSource and matMulReadWriteFnSource; https://github.com/tensorflow/tfjs, 2026-09-26).
// Copyright 2019 Google LLC. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
// Changed: A is the weights (M = the rows), read as 4 int8 of a u32 widened with their group's scale; B is the
// activations transposed (N = the tokens, a vec4 of 4 tokens at one column of the width), so that a thread's vec4 of
// sums is 4 tokens of a row; the tile's number is linear over x and then y as the other tiled shaders'; mm_write
// writes the 4 tokens one float at a time with shape.add; workPerThread [4, 4], workgroupSize [8, 8, 1] and
// tileInner 32 are TensorFlow.js's for large products (computeWorkgroupInfoForMatMul), and not transposed.
export const tfjsTile = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;

const rowPerThread = 4;
const colPerThread = 4;
const tileInner = 32;
const innerElementSize = 4;
const rowPerThreadB = 4;  // tileInner / workgroupSize[1]
const tileAOuter = 32;
const tileBOuter = 32;

var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;

// four weights of a row (columns col to col + 3) times their group's scale
fn mm_readA(row: i32, col: i32) -> vec4<f32> {
  var value = vec4<f32>(0.0);
  if (row < i32(shape.rows) && col < i32(shape.words * 4u)) {
    let word = w[u32(row) * shape.words + u32(col) / 4u];
    let q = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
    value = vec4<f32>(q) * scales[u32(row) * shape.perRow + u32(col) / ${GROUP}u];
  }
  return value;
}
// the activations of four tokens (col to col + 3) at one column of the width (row)
fn mm_readB(row: i32, col: i32) -> vec4<f32> {
  var value = vec4<f32>(0.0);
  if (row < i32(shape.words * 4u)) {
    for (var i = 0; i < 4; i++) {
      if (col + i < i32(step.tokens)) {
        value[i] = x[u32(col + i) * shape.xStride + u32(row)];
      }
    }
  }
  return value;
}
fn mm_write(row: i32, col: i32, valueIn: vec4<f32>) {
  if (row < i32(shape.rows)) {
    for (var i = 0; i < 4; i++) {
      if (col + i < i32(step.tokens)) {
        let at = u32(col + i) * shape.yStride + shape.first + u32(row);
        y[at] = select(0.0, y[at], shape.add != 0u) + valueIn[i];
      }
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) numWorkgroups: vec3<u32>) {
  let tilesA = (shape.rows + 31u) / 32u;
  let tilesB = (step.tokens + 31u) / 32u;
  let linear = workgroupId.y * numWorkgroups.x + workgroupId.x;
  if (linear >= tilesA * tilesB) {
    return;
  }
  let localRow = i32(localId.y);
  let tileRow = localRow * rowPerThread;
  let tileCol = i32(localId.x);

  let globalRow = i32(linear % tilesA) * tileAOuter + tileRow;
  let globalCol = i32(linear / tilesA) * tileBOuter + tileCol * colPerThread;

  let numTiles = (i32(shape.words * 4u) - 1) / tileInner + 1;
  var kStart = 0;

  var acc: array<vec4<f32>, rowPerThread>;

  // Loop over shared dimension.
  let tileRowB = localRow * rowPerThreadB;
  for (var t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (var innerRow = 0; innerRow < rowPerThread; innerRow++) {
          let inputRow = tileRow + innerRow;
          let inputCol = tileCol;
          mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, kStart + inputCol * innerElementSize);
      }

      // Load one tile of B into local memory.
      for (var innerRow = 0; innerRow < rowPerThreadB; innerRow++) {
          let inputRow = tileRowB + innerRow;
          let inputCol = tileCol;
          mm_Bsub[inputRow][inputCol] = mm_readB(kStart + inputRow, globalCol);
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0; k < tileInner / innerElementSize; k++) {
        let BCached0 = mm_Bsub[k * innerElementSize + 0][tileCol];
        let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
        let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
        let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];
        for (var i = 0; i < rowPerThread; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
          acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
          acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
          acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
      workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow++) {
      mm_write(globalRow + innerRow, globalCol, acc[innerRow]);
  }
}`;

// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul.wgsl.template and
// dp4a_matmul_common.wgsl.template (https://github.com/microsoft/onnxruntime, commit 3756d4dc, 2026-09-26), under the
// MIT License:
//
// Copyright (c) Microsoft Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// A is the activations (M = the tokens), B the weights (N = the rows): "a_global" is a token and "b_global" a row.
const sdp8ai = /* wgsl */ `
// Scaled dot product of 8 packed integers.
fn SDP8AI(a1: vec4<u32>, b1: vec4<u32>, a2: vec4<u32>, b2: vec4<u32>, scale: f32) -> f32 {
  var local_sum = dot4I8Packed(a1[0], b1[0]);
  local_sum += dot4I8Packed(a1[1], b1[1]);
  local_sum += dot4I8Packed(a1[2], b1[2]);
  local_sum += dot4I8Packed(a1[3], b1[3]);
  local_sum += dot4I8Packed(a2[0], b2[0]);
  local_sum += dot4I8Packed(a2[1], b2[1]);
  local_sum += dot4I8Packed(a2[2], b2[2]);
  local_sum += dot4I8Packed(a2[3], b2[3]);
  return f32(local_sum) * scale;
}`;
// ORT's step 2, one line a row of the subtile: from the workgroup's memory, or from the lanes of a subgroup of 16
const dp4aLines = (subgroup) => Array.from({ length: 16 }, (_, i) => `    lane_output${(i >> 2) + 1}[${i & 3}] += ` + (subgroup
  ? `SDP8AI(own_a0, subgroupShuffle(own_b0, ${i}u), own_a1, subgroupShuffle(own_b1, ${i}u), subgroupShuffle(own_scale_b, ${i}u) * own_scale_a);`
  : `SDP8AI(own_a0, tile_B[0][base_B + ${i}u], own_a1, tile_B[1][base_B + ${i}u], own_scale_a * scale_B[base_B + ${i}u]);`)).join("\n");
export const dp4a = (subgroups) => /* wgsl */ `requires packed_4x8_integer_dot_product;
${subgroups ? "enable subgroups;\n" : ""}
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized activations (QUANTIZE's xq)
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;
@group(0) @binding(6) var<storage, read> scales_a: array<f32>;       // QUANTIZE's xs: a scale a token and group of 32
${sdp8ai}

const tile_size = 64u;
const subtile_size = 16u;
const tile_size_k_vec = 2u;

// Shared memory
var<workgroup> tile_A: array<array<vec4<u32>, tile_size>, tile_size_k_vec>;  // 64 x 32
var<workgroup> scale_A: array<f32, tile_size>;                                // 64 x 1
var<workgroup> tile_B: array<array<vec4<u32>, tile_size>, tile_size_k_vec>;  // 64 x 32
var<workgroup> scale_B: array<f32, tile_size>;                                // 64 x 1

fn loadSHMA(a_global_base: u32, kidx_v: u32, row: u32, col: u32) {
  let a_global = a_global_base + row;
  if (a_global >= step.tokens) {
    return;
  }
  tile_A[col][row] = a[a_global * (shape.xStride / 16u) + kidx_v + col];
  if (col == 0u) {
    // kidx_v covers 16 values of k: a group of 32 is two
    scale_A[row] = scales_a[a_global * (shape.xStride / 32u) + kidx_v / 2u];
  }
}
fn loadSHMB(b_global_base: u32, kidx_v: u32, row: u32, col: u32) {
  let b_global = b_global_base + row;
  if (b_global >= shape.rows) {
    return;
  }
  tile_B[col][row] = b[b_global * (shape.words / 4u) + kidx_v + col];
  if (col == 0u) {
    scale_B[row] = scales_b[b_global * shape.perRow + kidx_v / 2u];
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32${subgroups ? `,
        @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_id: u32` : ""}) {
  let num_M_tile = (step.tokens + tile_size - 1u) / tile_size;
  let num_N_tile = (shape.rows + tile_size - 1u) / tile_size;
  let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
  if (workgroup_idx >= num_M_tile * num_N_tile) {
    return;
  }
  // During the load phase we use all 256 threads to load 64 rows of A/B.
  // For each row we load tile_size_k_vec (2) vectorized elements, which are 32 elements of K.
  let a_global_base = (workgroup_idx / num_N_tile) * tile_size;
  let b_global_base = (workgroup_idx % num_N_tile) * tile_size;
  let load_AorB = local_idx / 128u;
  let load_row = (local_idx % 128u) / 2u;
  let load_col = local_idx % 2u;

  // During the compute phase, we have the 64x64 tile split into subtiles of 16x16. We have a grid of 4x4 subtiles.
  let subtile_id = local_idx / subtile_size;
  let subtile_idx = subtile_id / 4u;
  let subtile_idy = subtile_id % 4u;
  let base_A = subtile_idx * 16u;
  let base_B = subtile_idy * 16u;
  // For each subtile we have 16 threads assigned.
  let a_idx = local_idx % subtile_size;

  var lane_output1: vec4<f32>;
  var lane_output2: vec4<f32>;
  var lane_output3: vec4<f32>;
  var lane_output4: vec4<f32>;
  // K's vectorization is 16 items per index; tile_size_k_vec (2) is the k tile of 32 in it.
  let K16 = shape.words / 4u;
  for (var kidx_v = 0u; kidx_v < K16; kidx_v += tile_size_k_vec) {
    // Load Phase: Populate shared memory for the workgroup.
    if (load_AorB == 0u) {
      loadSHMA(a_global_base, kidx_v, load_row, load_col);
    } else {
      loadSHMB(b_global_base, kidx_v, load_row, load_col);
    }
    workgroupBarrier();

    // Compute phase: Perform matmul for this subtile 16 x 32 x 16.
    // Step 1: Load from shared memory into registers across entire subgroup.
    let own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
    let own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
    let own_scale_a: f32 = scale_A[base_A + a_idx];
${subgroups ? `    if (sg_size == 16u) {
      let own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
      let own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
      let own_scale_b: f32 = scale_B[base_B + sg_id];
      // Step 2: Access registers across the subgroup using subgroupShuffle and perform the matmul.
${dp4aLines(true).replace(/^/gm, "  ")}
    } else {
      // Code for other subgroup sizes, simply doesn't use subgroups at all.
${dp4aLines(false).replace(/^/gm, "  ")}
    }` : `    // Relies on reads from single location tile_B[][base_B + col] by all being optimized by the hardware.
${dp4aLines(false)}`}
    workgroupBarrier();
  }
  let a_global = a_global_base + base_A + a_idx;
  let b_global = b_global_base + base_B;
  if (a_global < step.tokens) {
    let outputs = array<vec4<f32>, 4>(lane_output1, lane_output2, lane_output3, lane_output4);
    for (var i = 0u; i < 16u; i++) {
      if (b_global + i < shape.rows) {
        let at = a_global * shape.yStride + shape.first + b_global + i;
        y[at] = select(0.0, y[at], shape.add != 0u) + outputs[i / 4u][i % 4u];
      }
    }
  }
}`;

// The activations of the packed shaders, as the CPU's quantize_x makes them (kernels/kernel.ts): per token and group
// of 32, the scale is the largest |value| / 127 and a value round(value × (1 / scale)) (half to even), clamped to
// ±127, four to a u32 with the first in the lowest byte. ORT's dp4a_quantize is not taken: its pack4x8snorm rounds
// as ⌊0.5 + 127 × value⌋ (half up, not the CPU's half to even), and its groups are 128. One thread a group; the tokens are the dispatch's y. x holds
// the tokens xStride floats apart, xq the same bytes apart and xs the scales xStride / 32 floats apart.
export const QUANTIZE = /* wgsl */ `
struct Quantize { n: u32, xStride: u32, unused0: u32, unused1: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xs: array<f32>;
@group(0) @binding(3) var<uniform> quantize: Quantize;
@group(0) @binding(4) var<uniform> step: Step;
fn packed(v: vec4<i32>) -> u32 {
  let b = bitcast<vec4<u32>>(v) & vec4<u32>(0xffu);
  return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let token = id.y;
  if (g >= quantize.n / ${GROUP}u || token >= step.tokens) { return; }
  let at = (token * quantize.xStride) / 4u + g * 8u;
  var largest = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let v = abs(x[at + k]);
    largest = max(largest, max(max(v.x, v.y), max(v.z, v.w)));
  }
  let scale = largest / 127.0;
  let inverse = select(0.0, 1.0 / scale, scale > 0.0);
  for (var k = 0u; k < 8u; k++) {
    xq[at + k] = packed(clamp(vec4<i32>(round(x[at + k] * inverse)), vec4<i32>(-127), vec4<i32>(127)));
  }
  xs[token * (quantize.xStride / ${GROUP}u) + g] = scale;
}`;

// the most likely token: the first index of the largest logit, in one workgroup, so that only 4 bytes come back
export const ARGMAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(2) var<uniform> count: vec4u;
var<workgroup> best: array<f32, 256>;
var<workgroup> index: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  var value = -3.4e38;
  var at = 0u;
  for (var i = t; i < count.x; i += 256u) {
    if (logits[i] > value) { value = logits[i]; at = i; }
  }
  best[t] = value;
  index[t] = at;
  workgroupBarrier();
  for (var half = 128u; half > 0u; half >>= 1u) {
    if (t < half && (best[t + half] > best[t] || (best[t + half] == best[t] && index[t + half] < index[t]))) {
      best[t] = best[t + half];
      index[t] = index[t + half];
    }
    workgroupBarrier();
  }
  if (t == 0u) { chosen[0] = index[0]; }
}`;

// a dispatch that does nothing: what a dispatch costs by itself (the benchmark's)
export const EMPTY = /* wgsl */ `@compute @workgroup_size(1) fn main() {}`;

// the benchmark's stand-in for the small steps of a layer (norms, RoPE, a short attention, SwiGLU, the residual adds):
// what they cost is mostly that they are dispatches of their own, so one that adds a vector stands for each
export const SMALL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&x);
  for (var i = id.x; i < n; i += 64u) { y[i] = y[i] + x[i]; }
}`;

// ---- the steps of a layer besides its matrices, for the tokens of a prompt (the model's GPU worker). Each does for
// every token what the CPU's kernel of the same name (kernels/kernel.ts) does for one, in float32.

// RMSNorm: one workgroup per token. out = weight * (x / sqrt(mean(x²) + eps)); weight: this layer's, from float at
export const RMSNORM = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> norm: Norm;
@group(0) @binding(4) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let token = id.x;
  if (token >= step.tokens) { return; }
  let row = token * norm.size;
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) { squares += x[row + i] * x[row + i]; }
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var i = t; i < norm.size; i += 64u) { out[row + i] = weight[norm.at + i] * (s * x[row + i]); }
}`;

// RoPE on q and k, and the keys and values of every token into this layer's cache at its position (step.pos + the
// token): one workgroup per token. Pairs of neighbours turn (llama2.c's order), the first turned of every head (all
// of it but for GPT-NeoX); angles holds, per token, the cos of its headSize / 2 angles and then their sin.
export const ROPE = /* wgsl */ `
struct Rope { heads: u32, kvHeads: u32, headSize: u32, turned: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> values: array<f32>;
@group(0) @binding(5) var<storage, read> angles: array<f32>;
@group(0) @binding(6) var<uniform> rope: Rope;
@group(0) @binding(7) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let token = id.x;
  if (token >= step.tokens) { return; }
  let size = rope.headSize;
  let half = size / 2u;
  let pairs = rope.turned / 2u;
  let angle = token * size;
  // q in place, a pair per thread
  for (var p = t; p < rope.heads * pairs; p += 64u) {
    let i = p % pairs;
    let at = token * rope.heads * size + (p / pairs) * size + 2u * i;
    let c = angles[angle + i];
    let s = angles[angle + half + i];
    let v0 = q[at];
    let v1 = q[at + 1u];
    q[at] = v0 * c - v1 * s;
    q[at + 1u] = v0 * s + v1 * c;
  }
  // k turned on its way into the cache (the first of a pair from both, the second from both), v as it is
  let kvDim = rope.kvHeads * size;
  let row = (step.pos + token) * kvDim;
  for (var j = t; j < kvDim; j += 64u) {
    let at = token * kvDim + j;
    var key = k[at];
    let inHead = j % size;
    if (inHead < rope.turned) {
      let c = angles[angle + inHead / 2u];
      let s = angles[angle + half + inHead / 2u];
      if (inHead % 2u == 0u) { key = k[at] * c - k[at + 1u] * s; } else { key = k[at - 1u] * s + k[at] * c; }
    }
    keys[row + j] = key;
    values[row + j] = v[at];
  }
}`;

// Attention: one workgroup per head and token. The token at position step.pos + token sees the positions up to its own
// (causal: none after it, also where the cache holds the later tokens of the same request). The scores go to scores
// ([tokens][heads][positions]), then their softmax, then the weighted sum of the values into out. Grouped-query
// attention: heads / kvHeads query heads share a head of keys and values.
export const ATTENTION = /* wgsl */ `
struct Attention { heads: u32, kvHeads: u32, headSize: u32, positions: u32, scale: f32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> attention: Attention;
@group(0) @binding(6) var<uniform> step: Step;
var<workgroup> head: array<f32, 256>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let h = id.x;
  let token = id.y;
  if (token >= step.tokens) { return; }
  let size = attention.headSize;
  let kvDim = attention.kvHeads * size;
  let kv = (h / (attention.heads / attention.kvHeads)) * size;
  let at = token * attention.heads * size + h * size;
  let seen = step.pos + token + 1u;
  let first = (token * attention.heads + h) * attention.positions;
  for (var j = t; j < size; j += 64u) { head[j] = q[at + j]; }
  workgroupBarrier();
  // 1. the scores, and the largest
  var largest = -3.4e38;
  for (var p = t; p < seen; p += 64u) {
    var dot = 0.0;
    for (var j = 0u; j < size; j++) { dot += head[j] * keys[p * kvDim + kv + j]; }
    let score = dot * attention.scale;
    scores[first + p] = score;
    largest = max(largest, score);
  }
  partial[t] = largest;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] = max(partial[t], partial[t + half]); }
    workgroupBarrier();
  }
  let most = partial[0];
  workgroupBarrier();
  // 2. the exponentials and their sum
  var sum = 0.0;
  for (var p = t; p < seen; p += 64u) {
    let e = exp(scores[first + p] - most);
    scores[first + p] = e;
    sum += e;
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let inverse = 1.0 / partial[0];
  // every thread reads every position's weight now, written by the others
  storageBarrier();
  // 3. the weighted sum of the values, a value of the head per thread
  for (var d = t; d < size; d += 64u) {
    var total = 0.0;
    for (var p = 0u; p < seen; p++) { total += scores[first + p] * values[p * kvDim + kv + d]; }
    out[at + d] = total * inverse;
  }
}`;

// SwiGLU: gate = silu(gate) * up, for every token (the second dimension of the dispatch)
export const SWIGLU = /* wgsl */ `
struct Size { n: u32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> size: Size;
@group(0) @binding(3) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let i = id.x * 64u + t;
  if (id.y >= step.tokens || i >= size.n) { return; }
  let at = id.y * size.n + i;
  let v = gate[at];
  gate[at] = v / (1.0 + exp(-v)) * up[at];
}`;

// ---- T168: the device's ceilings, for the GPU section of /benchmark/ to say what share of them the prompt's shaders
// reach: the multiply-adds of f32 and f16 (GFLOPS), WGSL's dot4I8Packed (GOPS: a device that emulates it shows it
// here), and reading the workgroup's memory and a storage buffer (GB/s). The shapes are clpeak's
// (https://github.com/krrishnarraj/clpeak, src/opencl/kernels/: compute_sp, compute_int8_dp, global_bandwidth), taken
// as a method only and written anew here: clpeak is under the GPL-3.0, and no line of it is copied. What the method
// keeps a compiler from making the loops cheaper than they look:
//   - a multiply-add chain is the squaring recurrence x = x·x + c (c differs by lane), which no algebra folds; two
//     chains of vec4 give each thread 8 independent ones (clpeak: "N is 1 from width 4 up"). c is in [-1.55, -1], where
//     the recurrence stays in [-2, 2] (the real axis of the Mandelbrot set): no infinities and no subnormals to time;
//   - a dot4I8Packed chain is two accumulators feeding each other, a = dot(x, b) + a, b = dot(x, a) + b (clpeak's
//     compute_int8_dp: with both operands loop-invariant a compiler may turn the chain into one multiply); four pairs;
//     WGSL's dot has no accumulating form, so the add is part of each dot here as in the DP4A shader of the prompt;
//   - every thread writes what its chains or reads came to, and the loop count comes from a uniform;
//   - the workgroup's memory is read at addresses that move with the loop (nothing to hoist out of it), 16 bytes a
//     thread with neighbours on neighbouring vec4s; the storage buffer as clpeak's global_offset kernels read it, each
//     read a dispatch's threads apart, so that neighbours read neighbouring vec4s and the buffer once a dispatch.
// Every ceiling runs CEILING_WORKGROUP threads a workgroup and writes one u32 a thread to out; the uniform (plan)
// holds the loops and a seed. *_PER_LOOP: what one thread does in one pass of its loop, in FLOPs, ops or bytes
export const CEILING_WORKGROUP = 256;
export const FMA_PER_LOOP = 16 * 2 * 4 * 2;
export const DOT4_PER_LOOP = 8 * 4 * 2 * 8;
export const SHARED_PER_LOOP = 16 * 16;
export const GLOBAL_PER_THREAD = 16 * 16;
const CEILING_HEAD = /* wgsl */ `
struct Plan { loops: u32, seed: u32, unused0: u32, unused1: u32 }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> plan: Plan;`;
export const fmaCeiling = (half) => {
  const T = half ? "f16" : "f32";
  return /* wgsl */ `${half ? "enable f16;" : ""}
${CEILING_HEAD}
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  let c = vec4<${T}>(${T}(-1.0 - f32(lane) / 1024.0)) - vec4<${T}>(0.0, 0.1, 0.2, 0.3);
  var x = vec4<${T}>(${T}(f32(plan.seed & 255u) / 256.0));
  var y = x - vec4<${T}>(0.5);
  for (var i = 0u; i < plan.loops; i++) {
${"    x = fma(x, x, c);\n    y = fma(y, y, c);\n".repeat(16)}  }
  out[id.x] = bitcast<u32>(dot(vec4<f32>(x + y), vec4<f32>(1.0)));
}`;
};
export const DOT4_CEILING = /* wgsl */ `requires packed_4x8_integer_dot_product;
${CEILING_HEAD}
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = plan.seed ^ (id.x * 0x9e3779b9u);
  var a0 = i32(id.x); var a1 = a0 + 1; var a2 = a0 + 2; var a3 = a0 + 3;
  var b0 = i32(x); var b1 = b0 ^ 1; var b2 = b0 ^ 2; var b3 = b0 ^ 3;
  for (var i = 0u; i < plan.loops; i++) {
${[...Array(8)].map(() => [0, 1, 2, 3].map((k) =>
    `    a${k} = dot4I8Packed(x, bitcast<u32>(b${k})) + a${k};\n    b${k} = dot4I8Packed(x, bitcast<u32>(a${k})) + b${k};\n`).join("")).join("")}  }
  out[id.x] = bitcast<u32>(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3);
}`;
// 16 KiB of the workgroup's memory (every device has that much: WebGPU's default limit), filled once, then read
export const SHARED_CEILING = /* wgsl */ `${CEILING_HEAD}
var<workgroup> held: array<vec4<f32>, 1024>;
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  for (var j = 0u; j < 4u; j++) { held[lane + j * ${CEILING_WORKGROUP}u] = vec4<f32>(f32(lane + j), f32(plan.seed), 1.0, 2.0); }
  workgroupBarrier();
  var s0 = vec4<f32>(); var s1 = vec4<f32>(); var s2 = vec4<f32>(); var s3 = vec4<f32>();
  for (var i = 0u; i < plan.loops; i++) {
    let at = ((i * 64u) & 511u) + lane;
${[...Array(16)].map((_, k) => `    s${k % 4} += held[at + ${k * 16}u];\n`).join("")}  }
  out[id.x] = bitcast<u32>(dot(s0 + s1 + s2 + s3, vec4<f32>(1.0)));
}`;
// the buffer read once a dispatch: 16 vec4s a thread, each a dispatch's threads after the one before
export const GLOBAL_CEILING = /* wgsl */ `${CEILING_HEAD}
@group(0) @binding(2) var<storage, read> data: array<vec4<u32>>;
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let apart = groups.x * ${CEILING_WORKGROUP}u;
  var s0 = vec4<u32>(); var s1 = vec4<u32>(); var s2 = vec4<u32>(); var s3 = vec4<u32>();
${[...Array(16)].map((_, k) => `  s${k % 4} += data[id.x + ${k}u * apart];\n`).join("")}  let s = s0 + s1 + s2 + s3;
  out[id.x] = s.x ^ s.y ^ s.z ^ s.w ^ plan.seed;
}`;

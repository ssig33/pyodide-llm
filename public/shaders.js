// shaders.js (T135): the WGSL of this project in one place. The GPU section of /benchmark/ (public/benchmark/gpu.js,
// T134) measures with some of them; the model's GPU worker (public/gpu.js) runs a prompt's tokens through the layers
// with the tiled matrices of T146 (the one each device runs fastest, T147), the flash attention and the steps of a
// layer below them. A plain ES module: both import it with the ?v= of their
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

// ---- T147: the tiled shaders a device may run a prompt's matrices with (T146's), for the model's GPU worker, which
// checks each against JavaScript on a small matrix (tiledOff) and times the right ones on the model's own weights, and
// takes the fastest: which is fastest differs from GPU to GPU (T146), and only the device can say. none: why a shape
// is not made here (the device's threads or workgroup memory), as the benchmark says it (public/benchmark/gpu.js).
export const promptForms = ({ half, subgroups, packed, memory, threads }) => {
  const past = ({ threads: wanted }, bytes) => (wanted > threads ? `${wanted} threads, the device ${threads}`
    : bytes > memory ? `${bytes} bytes of workgroup memory, the device ${memory}` : undefined);
  const forms = REG_TILES.map((tile) => {
    const shape = regTileShape(tile);
    return { name: `llama.cpp tiles ${shape.rows}×${shape.tokens}, ${half ? "f16" : "f32"}`, tile: shape, packed: false, half,
      code: regTile(half), constants: { WORKGROUP_SIZE_M: tile.m, WORKGROUP_SIZE_N: tile.n }, none: past(shape, regTileBytes(tile, half)) };
  });
  forms.push({ name: "TF.js tiles 32×32, vec4", tile: TFJS_SHAPE, packed: false, half: false, code: tfjsTile, none: past(TFJS_SHAPE, 2 * 32 * 32 * 4) });
  if (packed) {
    forms.push({ name: "ORT DP4A 64×64", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(false), none: past(DP4A_SHAPE, 4608) });
    if (subgroups) forms.push({ name: "ORT DP4A 64×64, subgroups", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(true), none: past(DP4A_SHAPE, 4608) });
  }
  return forms;
};

// x (groups of GROUP values) quantized as the CPU's quantize_x does it (and QUANTIZE): the largest |value| / 127,
// round half to even
export function quantizedLikeCpu(x) {
  const xq = new Int8Array(x.length), xs = new Float32Array(x.length / GROUP);
  for (let g = 0; g < xs.length; g++) {
    let largest = 0;
    for (let i = 0; i < GROUP; i++) largest = Math.max(largest, Math.abs(x[g * GROUP + i]));
    xs[g] = Math.fround(largest / 127);
    for (let i = 0; i < GROUP; i++) {
      const v = x[g * GROUP + i] / xs[g], r = Math.round(v);
      xq[g * GROUP + i] = Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 ? r - 1 : r;
    }
  }
  return { xq, xs };
}

// How far a tiled shader's products (got: y after the product twice, the second added: 2 × W·x) are from
// JavaScript's, the check of T146's review (public/benchmark/gpu.js's checkTiled): w, int8 [rows][n] with the float32
// scales s [rows][n / 32]; x, the tokens xStride apart; got, yStride apart. A packed form multiplies what the GPU
// quantized (xq, xs, xStride apart), held to JavaScript's quantize_x first: a scale may differ in its last bits (WGSL's
// division is not rounded exactly) and a value then by 1, a wrong index by far more. f32 forms: no more than 1e-4 of
// the row's and token's sum of |products| (a float32 sum in another order may differ by 544 × 2^-24 = 3.2e-5 of it at
// most; a wrong index, scale or group by about 1 / sqrt(544) = 4e-2). f16 forms: WGSL leaves the direction of the
// rounding to f16 to the device, so each weight × scale and activation is within 1 ulp (2^-10, or 2^-24 where it is
// subnormal): no more than |products| × (2^-9 + 2^-20 + (n + 1) × 2^-24) + 2^-24 × Σ(|weight| + |activation|).
// Returns { worst, wrong }: the worst difference over the sum of |products|, and why it is wrong, or null.
export function tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half }) {
  const perRow = n / GROUP;
  let worst = 0, over = false, far = false, apart = 0, values = 0;
  for (let t = 0; t < tokens; t++) {
    const at = t * xStride, groups = t * (xStride / GROUP);
    const mine = xq ? quantizedLikeCpu(x.subarray(at, at + n)) : null;
    if (mine) {
      for (let g = 0; g < perRow; g++) far ||= Math.abs(xs[groups + g] - mine.xs[g]) > 1e-6 * mine.xs[g];
      for (let i = 0; i < n; i++) {
        far ||= Math.abs(xq[at + i] - mine.xq[i]) > 1;
        apart += xq[at + i] !== mine.xq[i];
      }
      values += n;
    }
    for (let r = 0; r < rows; r++) {
      let want = 0, size = 0, small = 0;
      for (let g = 0; g < perRow; g++) {
        const scale = s[r * perRow + g] * (mine ? xs[groups + g] : 1);
        for (let i = g * GROUP; i < (g + 1) * GROUP; i++) {
          const weight = Math.fround(w[r * n + i] * s[r * perRow + g]), value = x[at + i];
          const product = mine ? w[r * n + i] * xq[at + i] * scale : half ? weight * value : w[r * n + i] * value * s[r * perRow + g];
          want += product;
          size += Math.abs(product);
          small += Math.abs(weight) + Math.abs(value);
        }
      }
      const off = Math.abs(got[t * yStride + r] / 2 - want);
      worst = Math.max(worst, off / size);
      over ||= half ? off > size * (2 ** -9 + 2 ** -20 + (n + 1) * 2 ** -24) + small * 2 ** -24 : off >= 1e-4 * size;
    }
  }
  const wrong = far ? "the quantized activations are far from quantize_x's" : apart > 0.01 * values
    ? `${apart} of ${values} quantized activations are not quantize_x's` : over ? `products ${worst.toExponential(2)} from JavaScript's` : null;
  return { worst, wrong };
}

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
// of it but for GPT-NeoX); angles holds, per token, the cos of its headSize / 2 angles and then their sin. The cache
// holds float16 (T147: as the CPU's cache does, T110, and as llama.cpp's flash attention reads its K and V), a pair of
// neighbours to a u32 (pack2x16float: no shader-f16 needed), the pair RoPE turns together.
export const ROPE = /* wgsl */ `
struct Rope { heads: u32, kvHeads: u32, headSize: u32, turned: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> values: array<u32>;
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
  // k turned on its way into the cache, v as it is: a pair of neighbours per thread
  let kvDim = rope.kvHeads * size;
  let row = (step.pos + token) * kvDim / 2u;
  for (var j = t; j < kvDim / 2u; j += 64u) {
    let at = token * kvDim + 2u * j;
    var key = vec2<f32>(k[at], k[at + 1u]);
    let inHead = (2u * j) % size;
    if (inHead < rope.turned) {
      let c = angles[angle + inHead / 2u];
      let s = angles[angle + half + inHead / 2u];
      key = vec2<f32>(key.x * c - key.y * s, key.x * s + key.y * c);
    }
    keys[row + j] = pack2x16float(key);
    values[row + j] = pack2x16float(vec2<f32>(v[at], v[at + 1u]));
  }
}`;

// T147: the attention of a prompt's tokens, llama.cpp's flash attention with tiles (flash_attn_tile.wgsl), where
// T135's (one workgroup a head and token) wrote every score to memory and read it three times, and read the same keys
// and values again for every head of q that shares them. A workgroup takes Q_TILE (4) tokens of one head: their q
// (scaled) in its memory, then KV_TILE positions of the head's keys at a time in its memory as f16 (or f32 where there
// is no shader-f16), one row of q a subgroup, a position a lane; the softmax online (the largest so far and the sum
// rescaled as a tile brings a larger one: subgroupMax and subgroupAdd), and the values of the tile the same way into
// each lane's vec4s of the output. Changed from the source, and why: the causal mask is the positions' order (a
// position past the row's own is not seen; llama.cpp adds a mask tensor of -inf), so the tiles stop at the last
// token's position; the keys and values are this project's cache (float16 pairs in u32, [positions][kvHeads ×
// headSize], a head's row from its offset), read four at a time as llama.cpp's vec4 loader (flash_attn_staging.tmpl);
// q and the output are float32 [tokens][heads × headSize]; no ALiBi, soft-cap or sinks. Where there are no subgroups
// (or no subgroup_id), LANES threads of the workgroup stand for a row's subgroup and add up in the workgroup's memory
// (this project's: llama.cpp's tile path needs subgroups). The shape is llama.cpp's choice
// (ggml-webgpu-shader-lib.hpp): Q_TILE 4, KV_TILE at most 64 and what the workgroup's memory holds, WG_SIZE the larger
// of 128 and 4 subgroups; MIN_SUBGROUP_SIZE sizes the registers.
export const FLASH_Q_TILE = 4;
export const flashShape = ({ headSize, half, subgroups, memory, threads, subgroupMin = 4, subgroupMax = 128 }) => {
  const bytes = half ? 2 : 4;
  const wgSize = subgroups ? Math.min(threads, Math.max(128, FLASH_Q_TILE * subgroupMax)) : 128;
  // llama.cpp's ggml_webgpu_flash_attn_wg_mem_bytes, for this shader's arrays: q, then per position its keys or values
  // and a weight of each row
  const base = FLASH_Q_TILE * headSize * 4 + (subgroups ? 0 : wgSize * 4), perPosition = headSize * bytes + FLASH_Q_TILE * bytes;
  const kvTile = Math.min(64, Math.floor((memory - base) / perPosition));
  return { headSize, half, subgroups, wgSize, kvTile, minSubgroup: subgroups ? subgroupMin : wgSize / FLASH_Q_TILE,
    none: subgroups && wgSize < FLASH_Q_TILE * subgroupMax ? `subgroups of ${subgroupMax} are more than ${threads} threads / 4`
      : kvTile < 1 ? `a head of ${headSize} is more than the workgroup's memory (${memory} bytes)` : undefined };
};

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl, flash_attn_decls.tmpl and
// flash_attn_staging.tmpl (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
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
export const flashTile = ({ headSize, half, subgroups, wgSize, kvTile, minSubgroup }) => /* wgsl */ `${half ? "enable f16;\n" : ""}${subgroups ? "enable subgroups;\n" : ""}
struct Params { heads: u32, kvHeads: u32, scale: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> Q: array<f32>;             // [tokens][heads × HEAD_DIM]
@group(0) @binding(1) var<storage, read> K: array<vec2<u32>>;       // [positions][kvHeads × HEAD_DIM]: 4 f16 a vec2<u32>
@group(0) @binding(2) var<storage, read> V: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;  // [tokens][heads × HEAD_DIM]
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> step: Step;

alias shmem_t = ${half ? "f16" : "f32"};
const HEAD_DIM_QK: u32 = ${headSize}u;
const HEAD_DIM_V: u32 = ${headSize}u;
const Q_TILE: u32 = ${FLASH_Q_TILE}u;
const KV_TILE: u32 = ${kvTile}u;
const WG_SIZE: u32 = ${wgSize}u;
const MIN_SUBGROUP_SIZE: u32 = ${minSubgroup}u;
// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
const V_CHUNKS: u32 = HEAD_DIM_V / 4u;
const SCORE_REGS_PER_LANE: u32 = (KV_TILE + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;
const OUT_REGS_PER_LANE: u32 = (V_CHUNKS + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;

const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
var<workgroup> kv_shmem: array<shmem_t, kv_shmem_size>;
var<workgroup> q_shmem: array<f32, Q_TILE * HEAD_DIM_QK>;
var<workgroup> p_shmem: array<shmem_t, Q_TILE * KV_TILE>;

fn halves(pair: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(pair.x), unpack2x16float(pair.y));
}
fn load_k_tile_block(local_x: u32, kv_count: u32, kv_tile: u32, k_head_offset: u32) {
    let stride_k1 = params.kvHeads * HEAD_DIM_QK;
    for (var vec_idx_local = local_x; vec_idx_local < kv_count * Q_CHUNKS; vec_idx_local += WG_SIZE) {
        let kv_local = vec_idx_local / Q_CHUNKS;
        let chunk = vec_idx_local % Q_CHUNKS;
        let global_k_row = kv_tile + kv_local;
        let k_vec_index = (k_head_offset + global_k_row * stride_k1 + chunk * 4u) >> 2u;
        let k4 = halves(K[k_vec_index]);
        let kv_off = kv_local * HEAD_DIM_QK + chunk * 4u;
        kv_shmem[kv_off + 0u] = shmem_t(k4.x);
        kv_shmem[kv_off + 1u] = shmem_t(k4.y);
        kv_shmem[kv_off + 2u] = shmem_t(k4.z);
        kv_shmem[kv_off + 3u] = shmem_t(k4.w);
    }
}
fn load_v_tile_block(local_x: u32, kv_count: u32, kv_tile: u32, v_head_offset: u32) {
    let stride_v1 = params.kvHeads * HEAD_DIM_V;
    for (var vec_idx_local = local_x; vec_idx_local < kv_count * V_CHUNKS; vec_idx_local += WG_SIZE) {
        let kv_local = vec_idx_local / V_CHUNKS;
        let chunk = vec_idx_local % V_CHUNKS;
        let global_v_row = kv_tile + kv_local;
        let v_vec_index = (v_head_offset + global_v_row * stride_v1 + chunk * 4u) >> 2u;
        let v4 = halves(V[v_vec_index]);
        let kv_off = kv_local * HEAD_DIM_V + chunk * 4u;
        kv_shmem[kv_off + 0u] = shmem_t(v4.x);
        kv_shmem[kv_off + 1u] = shmem_t(v4.y);
        kv_shmem[kv_off + 2u] = shmem_t(v4.z);
        kv_shmem[kv_off + 3u] = shmem_t(v4.w);
    }
}
${subgroups ? `fn row_max(value: f32) -> f32 { return subgroupMax(value); }
fn row_sum(value: f32) -> f32 { return subgroupAdd(value); }` : `// a row's LANES threads stand for its subgroup: their largest and their sum through the workgroup's memory
const LANES: u32 = WG_SIZE / Q_TILE;
var<workgroup> lanes: array<f32, WG_SIZE>;
var<private> lane_x: u32;
fn row_max(value: f32) -> f32 {
  lanes[lane_x] = value;
  workgroupBarrier();
  for (var half = LANES / 2u; half > 0u; half >>= 1u) {
    if (lane_x % LANES < half) { lanes[lane_x] = max(lanes[lane_x], lanes[lane_x + half]); }
    workgroupBarrier();
  }
  let result = lanes[lane_x - lane_x % LANES];
  workgroupBarrier();
  return result;
}
fn row_sum(value: f32) -> f32 {
  lanes[lane_x] = value;
  workgroupBarrier();
  for (var half = LANES / 2u; half > 0u; half >>= 1u) {
    if (lane_x % LANES < half) { lanes[lane_x] += lanes[lane_x + half]; }
    workgroupBarrier();
  }
  let result = lanes[lane_x - lane_x % LANES];
  workgroupBarrier();
  return result;
}`}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>${subgroups ? `,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    if (subgroup_size == 0u || num_subgroups < Q_TILE) {
        return;
    }` : `) {
    lane_x = local_id.x;
    let subgroup_id = local_id.x / LANES;
    let subgroup_size = LANES;
    let sg_inv_id = local_id.x % LANES;`}

    let wg_per_head = (step.tokens + Q_TILE - 1u) / Q_TILE;
    let head_idx = wg_id.x / wg_per_head;
    let k_head_idx = head_idx / (params.heads / params.kvHeads);
    let k_head_offset = k_head_idx * HEAD_DIM_QK;
    let v_head_offset = k_head_idx * HEAD_DIM_V;
    let stride_q1 = params.heads * HEAD_DIM_QK;

    let wg_in_head = wg_id.x % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;
    let global_q_row = q_row_start + subgroup_id;
    let row_active = subgroup_id < Q_TILE && global_q_row < step.tokens;
    // causal: the tile's positions up to its last token's, each row's up to its own
    let seq_len_kv = step.pos + min(q_row_start + Q_TILE, step.tokens);
    let row_position = step.pos + global_q_row;

    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_tile_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_tile_row;
        let global_q_row_offset = head_q_row * stride_q1 + head_idx * HEAD_DIM_QK;
        q_shmem[elem_idx] = select(
            0.0,
            Q[global_q_row_offset + q_col] * params.scale,
            head_q_row < step.tokens);
    }

    workgroupBarrier();

    var row_max_now = FLOAT_MIN;
    var exp_sum = 0.0;
    var out_regs: array<vec4<f32>, OUT_REGS_PER_LANE>;
    for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
        out_regs[reg_idx] = vec4<f32>(0.0);
    }

    let q_base = subgroup_id * HEAD_DIM_QK;
    let subgroup_p_offset = subgroup_id * KV_TILE;

    for (var kv_tile = 0u; kv_tile < seq_len_kv; kv_tile += KV_TILE) {
        let kv_count = min(KV_TILE, seq_len_kv - kv_tile);
        let score_slots = min(SCORE_REGS_PER_LANE, (kv_count + subgroup_size - 1u) / subgroup_size);
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        var local_scores: array<f32, SCORE_REGS_PER_LANE>;
        for (var slot = 0u; slot < SCORE_REGS_PER_LANE; slot += 1u) {
            local_scores[slot] = FLOAT_MIN;
        }

        load_k_tile_block(local_id.x, kv_count, kv_tile, k_head_offset);

        workgroupBarrier();

        var local_max = FLOAT_MIN;
        if (row_active) {
            for (var slot = 0u; slot < score_slots; slot += 1u) {
                let kv_local = sg_inv_id + slot * subgroup_size;
                if (kv_local >= kv_count) {
                    continue;
                }

                let global_k_row = kv_tile + kv_local;
                var dot_val = 0.0;
                for (var chunk = 0u; chunk < Q_CHUNKS; chunk += 1u) {
                    let q_off = q_base + chunk * 4u;
                    let qv = vec4<f32>(
                        q_shmem[q_off + 0u],
                        q_shmem[q_off + 1u],
                        q_shmem[q_off + 2u],
                        q_shmem[q_off + 3u]);
                    let kv_off = kv_local * HEAD_DIM_QK + chunk * 4u;
                    let kv = vec4<shmem_t>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    dot_val += dot(qv, vec4<f32>(kv));
                }
                // the causal mask: no position after the row's own
                if (global_k_row > row_position) {
                    dot_val = FLOAT_MIN;
                }
                local_scores[slot] = dot_val;
                local_max = max(local_max, dot_val);
            }
        }

        let tile_max = row_max(local_max);
        let new_max = max(row_max_now, tile_max);
        let cur_exp = exp(row_max_now - new_max);
        exp_sum *= cur_exp;
        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
            out_regs[reg_idx] *= cur_exp;
        }

        var local_sum = 0.0;
        for (var slot = 0u; slot < score_slots; slot += 1u) {
            let kv_local = sg_inv_id + slot * subgroup_size;
            if (row_active && kv_local < kv_count) {
                let p = exp(local_scores[slot] - new_max);
                p_shmem[subgroup_p_offset + kv_local] = shmem_t(p);
                local_sum += p;
            }
        }

        workgroupBarrier();

        load_v_tile_block(local_id.x, kv_count, kv_tile, v_head_offset);

        workgroupBarrier();

        let tile_sum = row_sum(local_sum);
        exp_sum += tile_sum;
        row_max_now = new_max;

        if (row_active) {
            for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
                let chunk = sg_inv_id + reg_idx * subgroup_size;
                if (chunk >= V_CHUNKS) {
                    continue;
                }

                var acc = out_regs[reg_idx];
                for (var kv_local = 0u; kv_local < kv_count; kv_local += 1u) {
                    let p = f32(p_shmem[subgroup_p_offset + kv_local]);
                    let kv_off = kv_local * HEAD_DIM_V + chunk * 4u;
                    let v4 = vec4<shmem_t>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    acc += p * vec4<f32>(v4);
                }
                out_regs[reg_idx] = acc;
            }
        }

        workgroupBarrier();
    }

    if (row_active) {
        let inv_exp_sum = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
        let row_base = global_q_row * stride_q1 + head_idx * HEAD_DIM_V;
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
            let chunk = sg_inv_id + reg_idx * subgroup_size;
            if (chunk >= V_CHUNKS) {
                continue;
            }
            let dst_vec_index = (row_base + chunk * 4u) >> 2u;
            dst[dst_vec_index] = out_regs[reg_idx] * inv_exp_sum;
        }
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
//   - the multiply-adds in two shapes, as clpeak races two (mad_chain.cl) and keeps the faster, since no one shape is
//     the fastest on every device: "square", the recurrence x = x·x + c (c differs by lane), which no algebra folds,
//     two chains of vec4 (8 independent multiply-adds a thread); and "affine", x = x·a + b with a and b the same for
//     every lane (from the uniform), one chain of vec4 (clpeak: "N is 1 from width 4 up": the vector is the
//     parallelism), whose three operands are distinct registers (clpeak: Intel's GPUs halve a multiply-add that reads
//     one register twice, as x·x does). Floating point does not reassociate, so the affine chain is not folded either.
//     c is in [-1.55, -1], where the square stays in [-2, 2] (the real axis of the Mandelbrot set), and a = 0.999,
//     b = 0.001 draw the affine one to 1: no infinities and no subnormals to time. Both do FMA_PER_LOOP a loop;
//   - a dot4I8Packed chain is two accumulators feeding each other, a = dot(x, b) + a, b = dot(x, a) + b (clpeak's
//     compute_int8_dp: with both operands loop-invariant a compiler may turn the chain into one multiply); four pairs;
//     WGSL's dot has no accumulating form, so the add is part of each dot here as in the DP4A shader of the prompt;
//   - every thread writes what its chains or reads came to, and the loop count comes from a uniform;
//   - the workgroup's memory is read at addresses that move with the loop (nothing to hoist out of it), 16 bytes a
//     thread with neighbours on neighbouring vec4s; the storage buffer as clpeak's global_offset kernels read it, each
//     read a dispatch's threads apart, so that neighbours read neighbouring vec4s and the buffer once a dispatch.
// Every ceiling runs CEILING_WORKGROUP threads a workgroup and writes one u32 a thread to out; the uniform (plan)
// holds the loops, a seed and the affine chain's a and b. *_PER_LOOP: what one thread does in one pass of its loop, in FLOPs, ops or bytes
export const CEILING_WORKGROUP = 256;
export const FMA_PER_LOOP = 32 * 4 * 2;
export const DOT4_PER_LOOP = 8 * 4 * 2 * 8;
export const SHARED_PER_LOOP = 16 * 16;
export const GLOBAL_PER_THREAD = 16 * 16;
const CEILING_HEAD = /* wgsl */ `
struct Plan { loops: u32, seed: u32, a: f32, b: f32 }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> plan: Plan;`;
export const FMA_SHAPES = ["square", "affine"];
export const fmaCeiling = (half, shape) => {
  const T = half ? "f16" : "f32";
  const body = shape === "square"
    ? `  let c = vec4<${T}>(${T}(-1.0 - f32(lane) / 1024.0)) - vec4<${T}>(0.0, 0.1, 0.2, 0.3);
  var x = vec4<${T}>(${T}(f32(plan.seed & 255u) / 256.0));
  var y = x - vec4<${T}>(0.5);
  for (var i = 0u; i < plan.loops; i++) {
${"    x = fma(x, x, c);\n    y = fma(y, y, c);\n".repeat(16)}  }
  out[id.x] = bitcast<u32>(dot(vec4<f32>(x + y), vec4<f32>(1.0)));`
    : `  let a = vec4<${T}>(${T}(plan.a));
  let b = vec4<${T}>(${T}(plan.b));
  var x = vec4<${T}>(${T}(f32(lane) / 256.0)) + vec4<${T}>(0.0, 0.1, 0.2, 0.3);
  for (var i = 0u; i < plan.loops; i++) {
${"    x = fma(x, a, b);\n".repeat(32)}  }
  out[id.x] = bitcast<u32>(dot(vec4<f32>(x), vec4<f32>(1.0)));`;
  return /* wgsl */ `${half ? "enable f16;" : ""}
${CEILING_HEAD}
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
${body}
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
// the buffer read once a dispatch (plan.loops unused: its size is the work): 16 vec4s a thread, each a dispatch's threads after the one before
export const GLOBAL_CEILING = /* wgsl */ `${CEILING_HEAD}
@group(0) @binding(2) var<storage, read> data: array<vec4<u32>>;
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let apart = groups.x * ${CEILING_WORKGROUP}u;
  var s0 = vec4<u32>(); var s1 = vec4<u32>(); var s2 = vec4<u32>(); var s3 = vec4<u32>();
${[...Array(16)].map((_, k) => `  s${k % 4} += data[id.x + ${k}u * apart];\n`).join("")}  let s = s0 + s1 + s2 + s3;
  out[id.x] = s.x ^ s.y ^ s.z ^ s.w ^ plan.seed;
}`;

// ---- T149: a matrix times one vector (a generated token's), in the forms of public implementations, for the GPU
// section of /benchmark/ to measure beside WIDEN and PACKED (the model's GPU worker is to take the fastest on each
// device, T152). WIDEN and PACKED give a row a workgroup of 64 threads that read one u32 each at a time; on the owner's
// Android they read Llama 3.2 1B's classifier at 27 to 36 GB/s but its w1 (16.8 MB) at 3.3 to 15.7 (T134). These read
// more at once and give a workgroup several rows:
//   mulMatVec({ packed, subgroups }): llama.cpp's WebGPU mul_mat_vec (mul_mat_vec.wgsl with mul_mat_vec_acc.tmpl's
//     MUL_ACC_Q8_0, or with mul_mat_vec_q_acc.tmpl's MMVQ path for Q8_0 where packed): 256 threads for OUTPUTS_PER_WG
//     rows, four threads to a group of 32 (8 weights, two u32, a thread), the vector's 8 values held in registers for
//     every row of the workgroup; the sums added up in the workgroup's memory, or with subgroupAdd where subgroups.
//     llama.cpp takes MMVQ (the int8 dot) only on AMD, Intel and NVIDIA; the benchmark measures both everywhere.
//   ortMatVec: ONNX Runtime's MatMulNBits for 8 bits (matmul_nbits.wgsl.template, MatMulNBitsProgram: the form it takes
//     for a token where DP4A is not taken): 128 threads for tile_size 8 rows, 32 of them along the width, each reading
//     a vec4<u32> (16 weights) at a time; the vector's 512 values of a step in the workgroup's memory.
//   ortDp4aMatVec: ONNX Runtime's DP4A MatMulNBits for small M (dp4a_matmul_small_m.wgsl.template, which ORT takes for a
//     token on devices with subgroups other than Apple's when the output is f32): 128 threads for 4 rows, 32 along the
//     width, each a group of 32 as two vec4<u32> by dot4I8Packed against the quantized vector in the workgroup's memory.
// Changed from the sources for this project's weights, and why: the int8 values and their float32 scales are two
// buffers (llama2_numpy's layout), so a group's scale is read from its own buffer (llama.cpp's Q8_0 blocks carry an f16
// scale in 34 bytes; ORT's scales are their own buffer already); the weights are signed (ORT's 8 bits are unsigned
// about 128: no zero point is taken off, and the widened form reads the bytes signed with extractBits rather than
// unpack4xU8, which also needs no language feature); the vector's quantized scales are one a group of 32 (ORT's are
// one a 128, llama.cpp's q8_1 blocks carry theirs); one vector (llama.cpp's NUM_COLS 1, ORT's M 1, no batches, no bias,
// no weight index: those loops and offsets are left out); the output goes to y from row first on (a matrix cut in
// chunks of rows). The bindings and the Shape are WIDEN's and PACKED's (x, or xq with its scales xs at 5), so that
// the benchmark binds them all alike.
export const MUL_MAT_VEC_ROWS = 4;  // llama.cpp's WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG
export const ORT_MATVEC_ROWS = 8;  // ORT's tile_size for MatMulNBitsProgram
export const ORT_DP4A_MATVEC_ROWS = 4;  // ORT's tile_size_n for DP4AMatMulNBitsSmallMProgram
const MATVEC_SHAPE = /* wgsl */ `struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }`;

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl, mul_mat_vec_acc.tmpl (MUL_ACC_Q8_0),
// mul_mat_vec_q_acc.tmpl (MMVQ, LEGACY_QUANTS, MUL_ACC_Q8_0) and common_decls.tmpl (get_byte_i32), with the defaults
// of ggml-webgpu-shader-lib.hpp (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT
// License:
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
// The preprocessor's defines are the template's parameters here (a function-scope array cannot take an override's
// size): WG_SIZE 256, OUTPUTS_PER_WG 4, NUM_COLS 1 (its loop and the barrier after each column left out).
const mulMatVecAcc = (packed) => (packed ? /* wgsl */ `
fn accumulate_vec_q_dot(thread_id: u32, row_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.perRow;

    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let inner_id = thread_id % THREADS_PER_BLOCK;
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                // repack_a: the block's two words of this thread; get_dm: the block's scale, from its own buffer
                let block_word_base = output_row * params.words + block * (BLOCK_SIZE / 4u);
                let a_repacked = vec2<u32>(src0[block_word_base + inner_id * 2u], src0[block_word_base + inner_id * 2u + 1u]);
                let da = scales[output_row * params.perRow + block];
                // repack_b_qs and repack_b_dm: the quantized vector's two words and its group's scale
                let b_repacked = vec2<u32>(src1[block * (BLOCK_SIZE / 4u) + inner_id * 2u], src1[block * (BLOCK_SIZE / 4u) + inner_id * 2u + 1u]);
                let b_ds = src1_scales[block];

                let row_sum = dot4I8Packed(a_repacked[0], b_repacked[0]) + dot4I8Packed(a_repacked[1], b_repacked[1]);

                acc[row] += f32(row_sum) * (da * b_ds);
            }
        }
    }

    return acc;
}` : /* wgsl */ `
fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

fn accumulate_vec_dot(thread_id: u32, row_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.perRow;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = src1[x_base + i];
        }
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                // the block's scale from its own buffer, its values the row's words from block * 8 on
                let d = scales[output_row * params.perRow + block];
                let block_word_base = output_row * params.words + block * (BLOCK_SIZE / 4u);
                var q_packed: array<u32, ELEMS_PER_THREAD / 4u>;
                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    q_packed[packed_idx] = src0[block_word_base + thread_within_block * 2u + packed_idx];
                }
                var row_sum = 0.0;
                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                        let q_val = f32(get_byte_i32(q_packed[packed_idx], byte_idx)) * d;
                        row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                    }
                }
                acc[row] += row_sum;
            }
        }
    }

    return acc;
}`);
export const mulMatVec = ({ packed, subgroups }) => /* wgsl */ `${subgroups ? "enable subgroups;\nrequires subgroup_id;\n" : ""}${packed ? "requires packed_4x8_integer_dot_product;\n" : ""}
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<${packed ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: Shape;
${packed ? "@group(0) @binding(5) var<storage, read> src1_scales: array<f32>;" : ""}

const WG_SIZE = 256u;
const OUTPUTS_PER_WG = ${MUL_MAT_VEC_ROWS}u;
const BLOCK_SIZE = 32u;
const THREADS_PER_BLOCK = 4u;
const ELEMS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;
${mulMatVecAcc(packed)}

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>${subgroups ? `,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_size) subgroup_size: u32` : ""}
) {
    let thread_id = local_id.x;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.rows + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    if (wg_linear >= output_groups) {
        return;
    }

    let row_base = wg_linear * OUTPUTS_PER_WG;
    let dst_idx_base = params.first + row_base;

    let acc = ${packed ? "accumulate_vec_q_dot" : "accumulate_vec_dot"}(thread_id, row_base);
${subgroups ? `
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        let subgroup_total = subgroupAdd(acc[row]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(row, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var row = subgroup_id; (row < OUTPUTS_PER_WG) && (row_base + row < params.rows); row += num_subgroups) {
        var row_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            row_acc += partial_sums[partial_index(row, k)];
        }
        let row_total = subgroupAdd(row_acc);
        if (subgroup_invocation_id == 0) {
            dst[dst_idx_base + row] = row_total;
        }
    }` : `
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        partial_sums[partial_index(row, thread_id)] = acc[row];
    }

    workgroupBarrier();

    var stride = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                partial_sums[partial_index(row, thread_id)] += partial_sums[partial_index(row, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < OUTPUTS_PER_WG) {
        let output_row = row_base + thread_id;
        if (output_row < params.rows) {
            dst[dst_idx_base + thread_id] = partial_sums[partial_index(thread_id, 0)];
        }
    }`}
}`;

// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.wgsl.template (n_bits 8,
// component_a 4, component_b 4, no zero points) and dp4a_matmul_small_m.wgsl.template (n_bits 8) with the parameters
// matmul_nbits.cc and dp4a_matmul_nbits.cc give them (https://github.com/microsoft/onnxruntime, commit 3756d4dc,
// 2026-09-26), under the MIT License:
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
// A is the vector (M = 1, a_global 0), B the weights (N = the rows); workgroup_idx is linear over x and then y.
export const ortMatVec = /* wgsl */ `
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;       // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Shape;

const workgroup_size_x = 128u;
const tile_size = ${ORT_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;
const component_a = 4u;
const component_b = 4u;
const elements_in_value_b = component_b * (32u / 8u);
const tile_size_k = tile_size_k_vec * elements_in_value_b;
const a_length_per_tile = tile_size_k / component_a;
const block_size = 32u;

// four signed int8 of a u32 as floats (ORT's unpack4xU8 less the zero point of its unsigned 8 bits)
fn unpacked(v: u32) -> vec4<f32> {
  let w = bitcast<i32>(v);
  return vec4<f32>(vec4<i32>(extractBits(w, 0u, 8u), extractBits(w, 8u, 8u), extractBits(w, 16u, 8u), extractBits(w, 24u, 8u)));
}

// Shared memory
var<workgroup> tile_A : array<vec4<f32>, a_length_per_tile>;
var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size>;

fn loadSHMA(kidx: u32, col: u32)
{
    let k_offset = kidx / component_a + col;
    if (k_offset < uniforms.words) {
        tile_A[col] = a[k_offset];
    } else {
        tile_A[col] = vec4<f32>(0);
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
  let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
  let num_N_tile = (uniforms.rows + tile_size - 1u) / tile_size;
  if (workgroup_idx >= num_N_tile) {
    return;
  }
  let K = uniforms.words * 4u;
  let K_of_b = K / elements_in_value_b;
  let b_global_base = workgroup_idx * tile_size;

  let idx = local_idx % tile_size_k_vec;
  let idy = local_idx / tile_size_k_vec;

  for (var kidx = 0u; kidx < K; kidx += tile_size_k)
  {
    for (var id = local_idx; id < a_length_per_tile; id += workgroup_size_x)
    {
      loadSHMA(kidx, id);
    }
    workgroupBarrier();

    for (var local_row_offset = 0u; local_row_offset < tile_size; local_row_offset += sub_tile_count)
    {
      var b_global = b_global_base + local_row_offset + idy;
      var k_offset = kidx / elements_in_value_b + idx;
      if (b_global < uniforms.rows && k_offset < K_of_b)
      {
        let block_idx = (kidx + idx * elements_in_value_b) / block_size;
        let scale_b = scales_b[b_global * uniforms.perRow + block_idx];
        var b_value = b[b_global * K_of_b + k_offset];

        var sum = f32(0);
        var a_offset = idx * (4u / component_a) * component_b;
        for (var i = 0u; i < component_b; i++) {
            let b_value_unpacked = unpacked(b_value[i]) * scale_b;
            sum += dot(tile_A[a_offset], b_value_unpacked);
            a_offset += 1;
        }

        inter_results[local_row_offset + idy][idx] += sum;
      }
    }
    workgroupBarrier();
  }

  if (local_idx < tile_size) {
    var output_value = f32(0);
    for (var b = 0u; b < tile_size_k_vec; b++) {
      output_value += inter_results[local_idx][b];
    }
    let b_global =  b_global_base + local_idx;
    if (b_global < uniforms.rows) {
      output[uniforms.first + b_global] = output_value;
    }
  }
}`;

// ORT's scale_A holds a scale for each 128 of the vector (8 a step): here one for each group of 32 (32 a step), loaded
// with a bound of its own (the tile's bound is in 16s)
export const ortDp4aMatVec = /* wgsl */ `requires packed_4x8_integer_dot_product;
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized vector (xq)
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Shape;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;       // its scales (xs), one a group of 32
${sdp8ai}

const workgroup_size_x = 128u;
const tile_size = ${ORT_DP4A_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;

const double_tile_size_k_vec = 2 * tile_size_k_vec;

var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size>;
var<workgroup> tile_A : array<vec4<u32>, double_tile_size_k_vec>;
const scale_a_size_in_tile_a = double_tile_size_k_vec / 2;
var<workgroup> scale_A : array<f32, scale_a_size_in_tile_a>;

fn loadSHMA(kidx_v: u32, col: u32)
{
    let K16 = uniforms.words / 4u;
    let k_offset = kidx_v + col;
    if (k_offset >= K16) {
    return;
    }

    tile_A[col] = a[k_offset];
    if (col < scale_a_size_in_tile_a && kidx_v / 2u + col < uniforms.perRow)
    {
    // kidx_v - covers 16 values of k in input_a
    scale_A[col] = scales_a[kidx_v / 2u + col];
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
    let num_N_tile = (uniforms.rows + tile_size - 1u) / tile_size;
    if (workgroup_idx >= num_N_tile) {
        return;
    }
    let K32 = uniforms.perRow;
    let b_global_base = workgroup_idx * tile_size;
    // Handle each workgroup threads as a block of [sub_tile_count][tile_size_k_vec]
    let local_col = local_idx % tile_size_k_vec;
    let local_row = local_idx / tile_size_k_vec;

    for (var kidx_v:u32 = 0; kidx_v < K32; kidx_v += tile_size_k_vec)
    {
        // Load Phase: Populate shared memory for the workgroup.
        if (local_idx < double_tile_size_k_vec)
        {
        loadSHMA(kidx_v * 2, local_idx);
        }
        workgroupBarrier();
        var own_a: vec4<u32> = tile_A[local_col * 2];
        var own_a1: vec4<u32> = tile_A[local_col * 2 + 1];
        var own_scale_a = scale_A[local_col];
        let k_offset = kidx_v + local_col;
        // k_offset - covers 32 values of k in input_b
        let block_idx = k_offset;
        // calculate intermediate results into inter_results.
        for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_count) {
            let b_global = b_global_base + row_offset + local_row;
            if (b_global < uniforms.rows && k_offset < K32)
            {
                let b_offset = b_global * K32 + k_offset;
                let own_scale_b = scales_b[b_global * uniforms.perRow + block_idx];
                let own_b = b[b_offset * 2];
                let own_b1 = b[b_offset * 2 + 1];
                inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);
            }
        }
        workgroupBarrier();
    }

    if (local_idx < tile_size) {
      // Do reduce sum to get final output.
      var output_value = f32(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      let b_global =  b_global_base + local_idx;
      if (b_global < uniforms.rows) {
        output[uniforms.first + b_global] = output_value;
      }
    }
}`;

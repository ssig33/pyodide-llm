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

// ---- T155: int6 weights (T98) on the GPU. The model's GPU worker widens every int6 matrix once, as it puts it on the
// GPU, into the int8 the tiled shaders read (and keeps no int6 there): an int6 value is an int8 with its two low bits
// zero and its scale is a quarter (llama2_numpy.pack6), so the widened values and the checkpoint's own scales are an
// int8 matrix, the same products to the bit, and every tiled shader, check and timing stays as it is.
//
// WIDEN_SIX: the WGSL of that widening, one dispatch a piece of a matrix (gpu.js's upload). No public implementation
// has this packing (llama.cpp's Q6_K packs otherwise), so it is written apart (T155: Fable high). What gpu.js gives it:
//   @group(0) @binding(0) var<storage, read> packed: array<u32>;         the groups as llama2_numpy.pack6 writes them,
//                                                                        24 bytes (6 words) a group, one after another,
//                                                                        from word 0 (the piece alone, uploaded apart)
//   @group(0) @binding(1) var<storage, read_write> values: array<u32>;   the same groups widened, 32 int8 (8 words) a
//                                                                        group, value j of a group at its byte j (four
//                                                                        to a word, the lowest byte first), from word 0
//   @group(0) @binding(2) var<uniform> widen: vec4<u32>;                 x: the groups; y, z, w: 0
// dispatched as @workgroup_size(WIDEN_SIX_WORKGROUP) with sixDispatch(groups, the device's
// maxComputeWorkgroupsPerDimension): a thread a group, group number (workgroup_id.y × num_workgroups.x +
// workgroup_id.x) × WIDEN_SIX_WORKGROUP + local_invocation_index, those past widen.x doing nothing. gpu.js checks it
// against sixValues() below as it starts (every group of random bytes: any 24 bytes are a group), to the bit.
//
// The form (T155, Fable): four values a word, as the bytes lie. A group's six words are its 24 bytes: low[0..15] in
// words 0..3 (byte j of the group at byte j % 4 of word j / 4) and top[0..7] in words 4 and 5. An int6 value's int8 is
// its six bits shifted up twice, so it is a byte with nothing to sign-extend: bits 2-5 from a nibble of low[j] (value
// j its low nibble, value j + 16 its high one) and bits 6-7 from a pair of bits of top[j % 8] (values k, k + 8,
// k + 16, k + 24 at bits 0, 2, 4, 6). Because value j sits in the same byte of its word as low[j] and top[j % 8] do
// (word i of the output takes low from word i and top from word 4 + i % 2), each output word is two masked shifts
// of two input words: the nibbles of four values to bits 2-5 at once (0x0f0f0f0f, or the high nibbles as word >> 4),
// and their pairs to bits 6-7 at once (top << (6 - pair's bit), masked to 0xc0c0c0c0: what a shift carries across a
// byte lands under the mask). Masks and shifts by constants only, as kernels/six.ts. The packing is this project's,
// but the form is llama.cpp's for Q6_K, whose six bits lie as nibbles and pairs of bits too: four values a 32-bit
// word, masks by the byte (0x0F0F0F0F, 0x30303030, 0xC0C0C0C0) and shifts (https://github.com/ggml-org/llama.cpp,
// commit 95887577, MIT: ggml-cuda/vecdotq.cuh's vec_dot_q6_K_q8_1_impl_mmvq and ggml-metal/kernels/dequantize.h's
// dequantize_q6_K; no line copied: Q6_K's values are unsigned with 32 taken off, where pack6's are their int8's). A
// thread a group: it reads 6 words and writes 8, the 64 threads of a workgroup reading 1536 bytes in a row and writing
// 2048, every line of the cache used whole; the traffic is the 56 bytes a group either way, and the widening runs once
// as a model is put on the GPU (Llama 3.2 1B's layers: 0.73 GB read and 0.97 GB written by it; their scales, 0.12 GB,
// go up as an int8 model's do). Splitting a group over two threads would read the top words twice for 4 fewer stores
// a thread: not taken.
export const WIDEN_SIX_WORKGROUP = 64;
export const WIDEN_SIX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<uniform> widen: vec4<u32>;
// four int8 from the low nibbles of low's bytes and the pair of bits at bit (bits 0, 2, 4 or 6) of top's bytes
fn widened(low: u32, top: u32, bit: u32) -> u32 {
  return ((low & 0x0f0f0f0fu) << 2u) | ((top << (6u - bit)) & 0xc0c0c0c0u);
}
@compute @workgroup_size(${WIDEN_SIX_WORKGROUP})
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let g = (id.y * count.x + id.x) * ${WIDEN_SIX_WORKGROUP}u + t;
  if (g >= widen.x) { return; }
  let at = g * 6u;
  let low = vec4<u32>(packed[at], packed[at + 1u], packed[at + 2u], packed[at + 3u]);
  let top = vec2<u32>(packed[at + 4u], packed[at + 5u]);
  let out = g * 8u;
  values[out] = widened(low.x, top.x, 0u);            // values 0..3: low nibbles, top bits 0-1 of top[0..3]
  values[out + 1u] = widened(low.y, top.y, 0u);       // 4..7: top[4..7]
  values[out + 2u] = widened(low.z, top.x, 2u);       // 8..11: bits 2-3 of top[0..3]
  values[out + 3u] = widened(low.w, top.y, 2u);       // 12..15
  values[out + 4u] = widened(low.x >> 4u, top.x, 4u); // 16..19: high nibbles, bits 4-5
  values[out + 5u] = widened(low.y >> 4u, top.y, 4u); // 20..23
  values[out + 6u] = widened(low.z >> 4u, top.x, 6u); // 24..27: bits 6-7
  values[out + 7u] = widened(low.w >> 4u, top.y, 6u); // 28..31
}`;
/** the workgroups [x, y] of WIDEN_SIX for groups: a second dimension past the device's most of one */
export const sixDispatch = (groups, most) => {
  const workgroups = Math.ceil(groups / WIDEN_SIX_WORKGROUP), x = Math.min(workgroups, most);
  return [Math.max(1, x), Math.ceil(workgroups / Math.max(1, x))];
};
/** JavaScript's widening (the check's answer): the int8 values of groups of 24 packed bytes (a Uint8Array), as
 * llama2_numpy.unpack6 and forward.js's weightAt read them */
export function sixValues(packed) {
  const groups = packed.length / 24, out = new Int8Array(groups * 32);
  for (let g = 0; g < groups; g++) {
    for (let j = 0; j < 32; j++) {
      const low = j < 16 ? packed[g * 24 + j] & 15 : packed[g * 24 + j - 16] >> 4;
      const top = (packed[g * 24 + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      out[g * 32 + j] = ((low | (top << 4)) << 2) << 24 >> 24;
    }
  }
  return out;
}

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
  const dp4aNone = packed ? past(DP4A_SHAPE, 4608) : "no packed int8 dot here";
  forms.push({ name: "ORT DP4A 64×64", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(false), none: dp4aNone });
  if (subgroups) forms.push({ name: "ORT DP4A 64×64, subgroups", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(true), none: dp4aNone });
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
// Returns { worst, wrong, far, apart, values }: the worst difference over the sum of |products|, why it is wrong or
// null, and of a packed form whether a quantized value is far from quantize_x's and how many of values differ by 1.
// the f32 forms' line of tiledOff, a share of the sum of |products| (the matrix × vector checks of the benchmark hold theirs to it too)
export const TILED_LINE = 1e-4;
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
      over ||= half ? off > size * (2 ** -9 + 2 ** -20 + (n + 1) * 2 ** -24) + small * 2 ** -24 : off >= TILED_LINE * size;
    }
  }
  const wrong = far ? "the quantized activations are far from quantize_x's" : apart > 0.01 * values
    ? `${apart} of ${values} quantized activations are not quantize_x's` : over ? `products ${worst.toExponential(2)} from JavaScript's` : null;
  return { worst, wrong, far, apart, values };
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

// RMSNorm: one workgroup per row, out = weight * (x / sqrt(mean(x²) + eps)), weight this layer's from float at. The
// rows of a token are the dispatch's x (one for the layer's norms, a row of dim; T153: Qwen3's norms of q and k, a row
// a head of headSize, heads of them), the tokens its y. inPlace (T153): x is written over (a head's q or k), as
// llama.cpp's rms_norm_mul.wgsl has it (INPLACE: the norm and the weight's product in one dispatch, the weight's row
// the same for every row, mul_src_ne1 1); else out is another buffer (xb, the layer's norms).
//
// The row of a norm, its weight broadcast over the rows and the in-place form adapted from llama.cpp,
// ggml/src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl and binary.wgsl (OP_ADD, INPLACE; ADD below), and (T154)
// row_norm.wgsl (NORM: LAYER_NORM below) and unary.wgsl (GELU: GELU below), and the fusion of LAYER_NORM's last line
// from ggml/src/ggml-metal/kernels/norm.metal (kernel_norm_mul_add_f32, F == 3; ggml-metal-fusion.cpp's NORM_MUL_ADD)
// (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
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
const rmsNorm = (inPlace) => /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, ${inPlace ? "read_write" : "read"}> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${inPlace ? "" : "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"}@group(0) @binding(${inPlace ? 2 : 3}) var<uniform> norm: Norm;
@group(0) @binding(${inPlace ? 3 : 4}) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) rows: vec3u, @builtin(local_invocation_index) t: u32) {
  if (id.y >= step.tokens) { return; }
  let row = (id.y * rows.x + id.x) * norm.size;
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) { squares += x[row + i] * x[row + i]; }
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var i = t; i < norm.size; i += 64u) { ${inPlace ? "x" : "out"}[row + i] = weight[norm.at + i] * (s * x[row + i]); }
}`;
export const RMSNORM = rmsNorm(false);
export const HEAD_NORM = rmsNorm(true);

// T154: LayerNorm (GPT-2's and GPT-NeoX's), out = weight * ((x - mean) / sqrt(variance + eps)) + bias, as the CPU's
// layernorm kernel computes it: llama.cpp's row_norm.wgsl with NORM (the notice above: the sum of the row, its mean,
// then the sum of the squares about the mean, a reduction of the workgroup each), in the frame of RMSNORM above: one
// workgroup a row, the tokens the dispatch's y. The weight's MUL and the bias's ADD, which llama.cpp's WebGPU runs as
// two dispatches of binary.wgsl after the NORM, are the last line of this one, as llama.cpp's Metal fuses NORM, MUL and
// ADD (kernel_norm_mul_add_f32: (y * scale) * f0 + f1, y = x - mean, the rounding in the same order as here and as the
// CPU's kernel); both are this layer's, from float at.
export const LAYER_NORM = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> norm: Norm;
@group(0) @binding(5) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) rows: vec3u, @builtin(local_invocation_index) t: u32) {
  if (id.y >= step.tokens) { return; }
  let row = (id.y * rows.x + id.x) * norm.size;
  var sum = 0.0;
  for (var i = t; i < norm.size; i += 64u) { sum += x[row + i]; }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(norm.size);
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) {
    let d = x[row + i] - mean;
    squares += d * d;
  }
  workgroupBarrier();
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var i = t; i < norm.size; i += 64u) { out[row + i] = weight[norm.at + i] * (s * (x[row + i] - mean)) + bias[norm.at + i]; }
}`;

// T153: a bias added to every token of a matrix's output (Qwen2's q, k and v), llama.cpp's binary.wgsl with OP_ADD and
// INPLACE (the notice above): y += bias, the bias (this layer's, from float at) the same for every token (b_ne1 1).
// Changed: the token is the dispatch's y and the element its x (as SWIGLU here), where llama.cpp numbers every element
// of the tensor along x and y and finds its place in either by strides
export const ADD = /* wgsl */ `
struct Bias { n: u32, at: u32, unused0: u32, unused1: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> shape: Bias;
@group(0) @binding(3) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.y >= step.tokens || id.x >= shape.n) { return; }
  y[id.y * shape.n + id.x] += bias[shape.at + id.x];
}`;

// RoPE on q and k, and the keys and values of every token into this layer's cache at its position (step.pos + the
// token): one workgroup per token. Pairs of neighbours turn (llama2.c's order), the first turned of every head (all
// of it but for GPT-NeoX, T154: whose converter puts the pairs of its rotary part in this order; none of it for GPT-2,
// whose positions are learned and added to the rows by the CPU: its keys go into the cache as they are); angles holds, per token, the cos of its headSize / 2 angles and then their sin. The cache
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

// T154: GELU (GPT-2's gelu_new and GPT-NeoX's, the tanh approximation), gate = gelu(gate) for every token (the dispatch's
// y), in place: llama.cpp's unary.wgsl with GELU (the notice above: its formula, the argument of tanh clamped as it
// has it). Changed: the element is the dispatch's x and the token its y (as SWIGLU); the bias of the projection before
// it is ADD's, as llama.cpp adds it (the CPU's gelu kernel adds it itself: the same sum, v + b, rounded once)
export const GELU = /* wgsl */ `
struct Size { n: u32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<uniform> size: Size;
@group(0) @binding(2) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let i = id.x * 64u + t;
  if (id.y >= step.tokens || i >= size.n) { return; }
  let at = id.y * size.n + i;
  let v = gate[at];
  gate[at] = 0.5 * v * (1.0 + tanh(clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -9.010913, 9.010913)));
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

// ---- T150: a layer of a generated token in five dispatches where the separate steps take fourteen (RMSNorm; q, k and
// v; RoPE and the cache; the attention; o; the residual add; RMSNorm; gate and up; SwiGLU; down; the residual add):
//   1. q, k and v as one matrix (their rows one after the other in one buffer), the norm on its read of the input, and
//      RoPE and the cache on its write: q turned into q, k turned and v as they are into the cache at step.pos
//   2. the attention (flashTile, as the prompt's)
//   3. o, its product added to the residual stream on its write
//   4. gate and up in one workgroup (gate's rows, then up's in the same buffer), the norm on the read, SwiGLU on the
//      write
//   5. down, added to the residual stream on its write
// For /benchmark/'s GPU section first (T150), and for the model's GPU worker when it generates on the GPU (T152).
// The matrix × vector is llama.cpp's mul_mat_vec in its widened form (T149's mulMatVec), and each fold takes its form
// from a public implementation:
//   - gate and up in one kernel with the GLU on the write: llama.cpp's CUDA mul_mat_vec_q with its fusion (mmvq.cu:
//     vgate beside vx, tmp_gate beside tmp, reduced alike, and result *= silu(gate) where a row is written; commit
//     2145525a). Here the two matrices are one buffer, up's rows `second` rows after gate's.
//   - the residual added on the write: the same kernel's x_bias (result += x_biases[j], an ADD after the MUL_MAT fused
//     into it), and MLC LLM's fused matmul and add (FuseDequantizeMatmulEwise, Apache-2.0)
//   - q, k and v (and gate and up) as one matrix: MLC LLM's Llama (qkv_proj and gate_up_proj, Apache-2.0)
//   - RoPE and the cache on the write: llama.cpp's CUDA fuses rope and set_rows (ggml-cuda.cu, rope_set_rows_ops) into
//     one kernel after the matrix; here into the matrix's own write, since a workgroup's 4 rows hold whole pairs (the
//     neighbours RoPE turns together, and the pairs of float16 the cache holds a u32). The turning is ROPE's above.
//   - RMSNorm on the read: the norm's scale 1 / sqrt(mean(x²) + eps) is one number for the whole row, so it comes out of
//     the sum: W·(g ⊙ x·s) = s × W·(g ⊙ x). Every workgroup reads all of x once anyway, and adds up x² beside the rows'
//     sums in the same reduction (FlashNorm, Graef et al. 2024, arXiv 2407.09577: the scale deferred past the matrix).
//     No public WGSL does this (llama.cpp's WebGPU fuses the norm with its weight only, rms_norm_mul.wgsl): the lines
//     are this project's, written as llama.cpp's rms_norm_mul computes it (eps inside the sqrt with the mean, as the
//     engine's rmsnorm and RMSNORM above). Checked (T150, Fable): the deferred scale rounds once at the end where the
//     separate form rounds every x·s, so both are within a few float32 ulp of float64 (4e-7 of the largest row at
//     worst, with channels at 1000× the rest too: the sum of x² adds the small terms among themselves before the tree
//     meets a large one). What it costs a workgroup: reading g (4·dim bytes from the cache, beside x) and 2·dim more
//     ALU next to the rows' 4·dim·1.125 bytes from DRAM; what it saves a layer: two dispatches and a vector written and
//     read. Which is cheaper is the device's (the layer table's fused row against the separate one).
//     A norm of a head (Qwen3's q and k, T153) cannot go on the write: a head's rows span headSize / 4 workgroups and
//     WGSL has no sum across them; the read side stays as it is, the write side gets a small dispatch after the matrix.
// Changed from mul_mat_vec besides: the sums are SUMS (the rows', gate's rows' too, and x²'s), reduced as llama.cpp
// reduces its rows' (the workgroup's tree, or subgroupAdd), into totals that the write (the epilogue) reads; where a
// row is written differs by the output. The weights' layout, the scales and Params' first four are T149's.
//
// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl and mul_mat_vec_acc.tmpl (MUL_ACC_Q8_0),
// and ggml/src/ggml-cuda/mmvq.cu (the fusion of gate, bias and GLU) (https://github.com/ggml-org/llama.cpp, commit
// 2145525a, 2026-09-26), under the MIT License:
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
//
// input: "norm" (x is the residual stream, normed on the read with norm_weight from params.normAt) or "plain".
// output: "rope" (rows of q, then kvRows of k, then of v: q into q, k and v into the cache at step.pos), "add" (added
// to dst: the residual stream), "swiglu" (silu(gate) × up into dst, up's rows params.second after gate's) or "write"
// (into dst: the classifier's logits, T151).
// Bindings: 0 the weights, 1 their scales, 2 x, 3 params; 4 norm_weight (norm); 5 dst (add, swiglu, write), or 5 q,
// 6 the keys, 7 the values, 8 the angles and 9 the step (rope). T151 (T150's review, (c)): the position is the Step's
// (a uniform of its own, which a run of tokens on the GPU copies from the sampler's state after each token), and the
// angles are the whole table of RoPE on the GPU, a row a position: the cos of its headSize / 2 angles, then their sin
// (the rows of the CPU's table: the model's own, Llama 3's scaling too).
const FUSED_PARAMS = /* wgsl */ `struct Params { rows: u32, words: u32, perRow: u32, second: u32, eps: f32, normAt: u32, qRows: u32, kvRows: u32,
                headSize: u32, turned: u32, unused0: u32, unused1: u32 }`;
// where the write of an output goes (bindings 5 on): T150's, and T175's DP4A form's alike
const fusedOutputs = (output) => (output === "rope" ? `@group(0) @binding(5) var<storage, read_write> q: array<f32>;
@group(0) @binding(6) var<storage, read_write> keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> values: array<u32>;
@group(0) @binding(8) var<storage, read> angles: array<f32>;
@group(0) @binding(9) var<uniform> step: Step;` : "@group(0) @binding(5) var<storage, read_write> dst: array<f32>;");
// The write (the epilogue), T150's and T175's DP4A form's alike: a workgroup's OUTPUTS_PER_WG rows from row_base on,
// their sums in totals (up's rows' after them, for SwiGLU), each times scale (the norm's, or 1), by thread thread_id
const fusedWrite = (output) => (output === "rope" ? `    // a pair of neighbouring rows a thread: q's turned into q; k's turned and v's as they are into the cache at pos
    if (thread_id < OUTPUTS_PER_WG / 2u) {
        let row = row_base + 2u * thread_id;
        if (row < params.rows) {
            let v0 = totals[2u * thread_id] * scale;
            let v1 = totals[2u * thread_id + 1u] * scale;
            let size = params.headSize;
            let half = size / 2u;
            let at = step.pos * size;  // the position's row of the table: its cos, then its sin
            if (row < params.qRows) {
                let in_head = row % size;
                if (in_head < params.turned) {
                    let c = angles[at + in_head / 2u];
                    let s = angles[at + half + in_head / 2u];
                    q[row] = v0 * c - v1 * s;
                    q[row + 1u] = v0 * s + v1 * c;
                } else {
                    q[row] = v0;
                    q[row + 1u] = v1;
                }
            } else if (row < params.qRows + params.kvRows) {
                let j = row - params.qRows;
                var key = vec2<f32>(v0, v1);
                let in_head = j % size;
                if (in_head < params.turned) {
                    let c = angles[at + in_head / 2u];
                    let s = angles[at + half + in_head / 2u];
                    key = vec2<f32>(key.x * c - key.y * s, key.x * s + key.y * c);
                }
                keys[step.pos * params.kvRows / 2u + j / 2u] = pack2x16float(key);
            } else {
                let j = row - params.qRows - params.kvRows;
                values[step.pos * params.kvRows / 2u + j / 2u] = pack2x16float(vec2<f32>(v0, v1));
            }
        }
    }` : `    if (thread_id < OUTPUTS_PER_WG) {
        let row = row_base + thread_id;
        if (row < params.rows) {
            let value = totals[thread_id] * scale;${output === "swiglu" ? `
            // llama.cpp's result *= silu(gate_value), as the GLU writes it (glu.wgsl's OP_SWIGLU)
            let gate = value;
            let up = totals[OUTPUTS_PER_WG + thread_id] * scale;
            dst[row] = gate / (1.0 + exp(-gate)) * up;` : output === "write" ? `
            dst[row] = value;` : `
            dst[row] = dst[row] + value;`}
        }
    }`);
export const fusedMatVec = ({ input, output, subgroups }) => {
  const norm = input === "norm", glu = output === "swiglu";
  const matrices = glu ? 2 : 1, sums = MUL_MAT_VEC_ROWS * matrices + (norm ? 1 : 0);
  return /* wgsl */ `${subgroups ? "enable subgroups;\nrequires subgroup_id;\n" : ""}
${FUSED_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
${norm ? "@group(0) @binding(4) var<storage, read> norm_weight: array<f32>;" : ""}
${fusedOutputs(output)}

const WG_SIZE = 256u;
const OUTPUTS_PER_WG = ${MUL_MAT_VEC_ROWS}u;
const BLOCK_SIZE = 32u;
const THREADS_PER_BLOCK = 4u;
const ELEMS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;
const MATRICES = ${matrices}u;
const SUMS = ${sums}u;
const SQUARES = OUTPUTS_PER_WG * MATRICES;  // where the sum of x² is, when the norm is on the read

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

// a row's sum at the thread's part of a block: llama.cpp's inner loop, for one row of either matrix
fn block_dot(weight_row: u32, block: u32, thread_within_block: u32, x_block: array<f32, ELEMS_PER_THREAD>) -> f32 {
    let d = scales[weight_row * params.perRow + block];
    let block_word_base = weight_row * params.words + block * (BLOCK_SIZE / 4u);
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
    return row_sum;
}

// the sums as the reduction takes them: the rows' (acc), then up's rows' (acc_up, beside acc as mmvq.cu's tmp_gate beside tmp), then x²
fn accumulate_vec_dot(thread_id: u32, row_base: u32) -> array<f32, SUMS> {
    var acc: array<f32, OUTPUTS_PER_WG>;${glu ? `
    var acc_up: array<f32, OUTPUTS_PER_WG>;` : ""}${norm ? `
    var squares = 0.0;` : ""}

    let num_blocks = params.perRow;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = src1[x_base + i];${norm ? `
            // the workgroup reads each value of x once: x² for the norm's scale, and the norm's weight now
            squares += x_block[i] * x_block[i];
            x_block[i] *= norm_weight[params.normAt + x_base + i];` : ""}
        }
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                acc[row] += block_dot(output_row, block, thread_within_block, x_block);${glu ? `
                // up's row, params.second rows after gate's (mmvq.cu's vgate beside vx)
                acc_up[row] += block_dot(output_row + params.second, block, thread_within_block, x_block);` : ""}
            }
        }
    }

    var sums: array<f32, SUMS>;
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        sums[row] = acc[row];${glu ? `
        sums[OUTPUTS_PER_WG + row] = acc_up[row];` : ""}
    }${norm ? `
    sums[SQUARES] = squares;` : ""}
    return sums;
}

// Flattened as [sum][thread] to keep each sum's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, SUMS * WG_SIZE>;
// what the reduction leaves: each sum of the workgroup, for the write
var<workgroup> totals: array<f32, SUMS>;

fn partial_index(sum: u32, thread: u32) -> u32 {
    return sum * WG_SIZE + thread;
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

    let acc = accumulate_vec_dot(thread_id, row_base);
${subgroups ? `
    for (var sum = 0u; sum < SUMS; sum++) {
        let subgroup_total = subgroupAdd(acc[sum]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(sum, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var sum = subgroup_id; sum < SUMS; sum += num_subgroups) {
        var sum_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            sum_acc += partial_sums[partial_index(sum, k)];
        }
        let sum_total = subgroupAdd(sum_acc);
        if (subgroup_invocation_id == 0) {
            totals[sum] = sum_total;
        }
    }` : `
    for (var sum = 0u; sum < SUMS; sum++) {
        partial_sums[partial_index(sum, thread_id)] = acc[sum];
    }

    workgroupBarrier();

    var stride = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var sum = 0u; sum < SUMS; sum++) {
                partial_sums[partial_index(sum, thread_id)] += partial_sums[partial_index(sum, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < SUMS) {
        totals[thread_id] = partial_sums[partial_index(thread_id, 0)];
    }`}

    workgroupBarrier();

    // the write: the norm's scale on every sum (s × W·(g ⊙ x)), then what the output does with a row
    let scale = ${norm ? "1.0 / sqrt(totals[SQUARES] / f32(params.perRow * BLOCK_SIZE) + params.eps)" : "1.0"};
${fusedWrite(output)}
}`;
};

// ---- T175: T150's layer on ONNX Runtime's DP4A for small M (T149's ortDp4aMatVec), the form that read Llama 3.2 1B's
// w1 at 96.8% of the buffer's reads on the owner's Android where llama.cpp's mul_mat_vec (T150's) read a layer at
// 18.5%. ORT runs it after a dispatch of its own that quantizes the vector (dp4a_matmul_nbits.cc:
// DP4AMatMulQuantizeProgram, then DP4AMatMulNBitsSmallMProgram), and adds the bias on the write. So here: the vector
// quantized first (QUANTIZE, T146's, as the CPU's quantize_x makes it), then the matrix, and on its write what
// T150's fusedMatVec does (fusedWrite: RoPE and the cache, the residual's add, SwiGLU, or the logits). Four
// quantizations a layer, one where a new vector comes in: the normed stream before q, k and v (one for the three),
// the attention's output before o, the normed stream before gate and up (one for both), silu(gate) × up before down;
// the classifier's normed stream a fifth, once a token. The norm cannot go on the matrix's read as in T150: the
// matrix reads the quantized vector, and a group's scale is of the values after the norm, so the norm goes with the
// quantizer (NORM_QUANTIZE), one dispatch where RMSNORM and QUANTIZE are two.
//
// RMSNorm with its output quantized as QUANTIZE quantizes it: one workgroup of 64 a token, the sum of x² as RMSNORM
// adds it, then a group of 32 a thread at a time, its values weight × (s × x) as RMSNORM writes them, the largest
// |value| / 127 its scale and each value rounded (half to even) and clamped to ±127, four to a u32 (QUANTIZE's). The
// form is vLLM's rms_norm_per_block_quant (csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu
// with layernorm_utils.cuh, https://github.com/vllm-project/vllm, commit 24c9772d, Apache-2.0): the row's rms in one
// block, then each group's scale from its normed values, then the values normed and quantized (compute_rms,
// compute_dynamic_per_token_scales, norm_and_quant). No lines are taken from it: these are RMSNORM's and QUANTIZE's
// in one workgroup (vLLM's block reduction, fp8 and residual paths are not here).
export const NORM_QUANTIZE = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;
@group(0) @binding(4) var<uniform> norm: Norm;
@group(0) @binding(5) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
fn packed(v: vec4<i32>) -> u32 {
  let b = bitcast<vec4<u32>>(v) & vec4<u32>(0xffu);
  return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}
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
  for (var g = t; g < norm.size / ${GROUP}u; g += 64u) {
    let at = g * ${GROUP}u;
    var largest = 0.0;
    for (var i = 0u; i < ${GROUP}u; i++) { largest = max(largest, abs(weight[norm.at + at + i] * (s * x[row + at + i]))); }
    let scale = largest / 127.0;
    let inverse = select(0.0, 1.0 / scale, scale > 0.0);
    for (var k = 0u; k < ${GROUP / 4}u; k++) {
      let i = at + 4u * k;
      let v = vec4<f32>(weight[norm.at + i] * (s * x[row + i]), weight[norm.at + i + 1u] * (s * x[row + i + 1u]),
                        weight[norm.at + i + 2u] * (s * x[row + i + 2u]), weight[norm.at + i + 3u] * (s * x[row + i + 3u]));
      xq[(row + i) / 4u] = packed(clamp(vec4<i32>(round(v * inverse)), vec4<i32>(-127), vec4<i32>(127)));
    }
    xs[token * (norm.size / ${GROUP}u) + g] = scale;
  }
}`;

// The matrix: ORT's DP4A for small M as ortDp4aMatVec has it (128 threads for 4 rows, 32 of them along the width,
// each a group of 32 as two vec4<u32> against the quantized vector in the workgroup's memory, the partial sums of a
// row reduced in inter_results), on T150's Params and bindings (the scales of the vector at 4, where T150's norm
// weight is), and T150's write on its rows. Changed from ortDp4aMatVec besides: for SwiGLU a thread multiplies up's row
// (params.second rows after gate's) beside gate's with the same registers of the vector, into rows of inter_results of
// its own (llama.cpp's CUDA mul_mat_vec_q fuses gate so, T150: vgate beside vx); the reduced sums go to totals, and
// the write reads them there (ORT writes a row where it reduces it, and RoPE needs a pair).
//
// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_small_m.wgsl.template (n_bits
// 8) with the parameters dp4a_matmul_nbits.cc gives it (https://github.com/microsoft/onnxruntime, commit 3756d4dc,
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
//
// output: as fusedMatVec's ("rope", "add", "swiglu" or "write"). Bindings: 0 the weights, 1 their scales, 2 the
// quantized vector (xq), 3 params, 4 its scales (xs); 5 on as fusedMatVec's.
export const fusedDp4aMatVec = ({ output }) => /* wgsl */ `requires packed_4x8_integer_dot_product;
${FUSED_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized vector (xq)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;       // its scales (xs), one a group of 32
${fusedOutputs(output)}
${sdp8ai}

const workgroup_size_x = 128u;
const tile_size = ${ORT_DP4A_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;
const MATRICES = ${output === "swiglu" ? 2 : 1}u;
// fusedWrite's name for the rows of a workgroup
const OUTPUTS_PER_WG = tile_size;

const double_tile_size_k_vec = 2 * tile_size_k_vec;

var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size * MATRICES>;
var<workgroup> tile_A : array<vec4<u32>, double_tile_size_k_vec>;
const scale_a_size_in_tile_a = double_tile_size_k_vec / 2;
var<workgroup> scale_A : array<f32, scale_a_size_in_tile_a>;
// what the reduction leaves: each row's sum (up's rows' after gate's), for the write
var<workgroup> totals: array<f32, tile_size * MATRICES>;

fn loadSHMA(kidx_v: u32, col: u32)
{
    let K16 = params.words / 4u;
    let k_offset = kidx_v + col;
    if (k_offset >= K16) {
    return;
    }

    tile_A[col] = a[k_offset];
    if (col < scale_a_size_in_tile_a && kidx_v / 2u + col < params.perRow)
    {
    // kidx_v - covers 16 values of k in input_a
    scale_A[col] = scales_a[kidx_v / 2u + col];
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
    let num_N_tile = (params.rows + tile_size - 1u) / tile_size;
    if (workgroup_idx >= num_N_tile) {
        return;
    }
    let K32 = params.perRow;
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
            if (b_global < params.rows && k_offset < K32)
            {
                let b_offset = b_global * K32 + k_offset;
                let own_scale_b = scales_b[b_global * params.perRow + block_idx];
                let own_b = b[b_offset * 2];
                let own_b1 = b[b_offset * 2 + 1];
                inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);${output === "swiglu" ? `
                // up's row, params.second rows after gate's (mmvq.cu's vgate beside vx), with the same own_a
                let up_global = b_global + params.second;
                let up_offset = up_global * K32 + k_offset;
                let up_scale_b = scales_b[up_global * params.perRow + block_idx];
                inter_results[tile_size + row_offset + local_row][local_col] += SDP8AI(own_a, b[up_offset * 2], own_a1, b[up_offset * 2 + 1], own_scale_a * up_scale_b);` : ""}
            }
        }
        workgroupBarrier();
    }

    if (local_idx < tile_size * MATRICES) {
      // Do reduce sum to get final output.
      var output_value = f32(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      totals[local_idx] = output_value;
    }

    workgroupBarrier();

    let thread_id = local_idx;
    let row_base = b_global_base;
    let scale = 1.0;  // the norm is the quantizer's (NORM_QUANTIZE)
${fusedWrite(output)}
}`;

// ---- T151: a run of generated tokens on the GPU, the ids read back once for all of them. Each token of the run is
// one compute pass: EMBED (the token's row of the embedding into the residual stream), the layers (fusedMatVec, T150),
// the final norm and the classifier (fusedMatVec, "write"), then SAMPLE (the repetition penalty, softmax, top-p and
// the draw), which writes the token into `chosen` and the state the next pass starts from; between passes the state's
// first four words are copied into the Step uniform that fusedMatVec and flashTile read (copyBufferToBuffer: a
// uniform is not a shader's to write). What the CPU gets back: chosen[0 .. sampled) and the state, after the run.
// For /benchmark/'s GPU section first (T151), and for the model's GPU worker when it generates on the GPU (T152).
//
// What the GPU does is the CPU's sampling (kernels/kernel.ts's penalize() and sample(), the engine's generate()),
// token for token: the same random number (the CPU draws them, in order, one a sampled token: randoms[sampled]) picks
// the same token but where a float32 sum in another order moves a border (sampleLikeCpu below is that CPU in
// JavaScript; tests/smoke.mjs holds it to the kernel, /benchmark/'s check holds SAMPLE to it). Their forms:
//   - the penalty: MLC LLM's apply_penalty_inplace (a thread a token of the window, the logit divided where positive
//     and multiplied where not; mlc_llm/compiler_pass/attach_logit_processor.py, Apache-2.0), with the CPU's window
//     (the latest REPETITION_WINDOW tokens of the history, the prompt and BOS in it) and its once for each distinct
//     token, where MLC counts them (its presence and frequency penalties are not the engine's)
//   - the largest logit and its first index, and softmax: llama.cpp's WebGPU argmax.wgsl (the pairs reduced in the
//     workgroup's memory; here the smaller index of two equal, as NumPy's argmax and ARGMAX above) and soft_max.wgsl
//     (exp(value - max), summed by the workgroup's tree), with kernel.ts's temperature and its floor of the nucleus
//     (a token under a ten millionth of the best one's probability is left out before exp())
//   - top-p without sorting: MLC LLM's top_p_pivot (mlc_llm/op/top_p_pivot.py, Apache-2.0): pivots between a bound
//     known to keep at least top-p of the mass and one known not to, the sum of the probabilities at or over each
//     pivot added up in one pass, the bounds moved to the pivots, until the smallest probability of the nucleus is
//     found. Changed: the pivots are spaced over the bits of the float (a positive float orders as its bits), so the
//     search ends on that probability itself, where MLC's ends within 1e-7 of it; the probabilities are those left
//     after the floor, gathered in the order of their index (below) where MLC reads the whole vocabulary each round
//   - the draw: where the CPU sorts the nucleus and walks it from the most probable token to the random number times
//     its mass (WebLLM's sample_with_top_p does the same after an argsort on the GPU: TVM's
//     sample_top_p_top_k_from_sorted_prob, Apache-2.0), SAMPLE finds the token by the same pivots: the probability of
//     the token the walk stops at is the largest p whose mass at or over it passes the random number (tokens of equal
//     probability are taken in the order of their index: the CPU's quicksort takes them in no set order). Without a
//     nucleus the CPU walks the tokens in the order of their index, and so does SAMPLE: llama.cpp's WebGPU
//     cumsum.wgsl (a thread a run of consecutive tokens, their sums scanned in the workgroup's memory), the first token
//     whose running sum passes the random number
//   - gathering the tokens over the floor in the order of their index (for the pivots, and the order of equals):
//     cumsum.wgsl's scan again, of the counts
//   - the token's row of the embedding: llama.cpp's WebGPU get_rows.wgsl (copy_elements of Q8_0: each weight times
//     its block's scale), for this project's int8 in groups of 32
// This project's: the state that carries the loop from one token to the next (the input token, the position, the
// window, the stop), and stopping: a token of settings.stop is written into chosen and ends the run (the passes after
// it change nothing: the position and the state stay, so a pass recomputes the same position and writes the same
// cache row), as the CPU's generate() breaks at a stop token.
//
// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/argmax.wgsl, soft_max.wgsl, cumsum.wgsl and get_rows.wgsl
// (https://github.com/ggml-org/llama.cpp, commit 95887577, 2026-09-26), under the MIT License:
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
//
// Adapted from MLC LLM, python/mlc_llm/op/top_p_pivot.py and python/mlc_llm/compiler_pass/attach_logit_processor.py
// (and, for T191's sampling in chunks below, python/mlc_llm/compiler_pass/attach_softmax_with_temperature.py)
// (https://github.com/mlc-ai/mlc-llm, commit 9fa644f5, 2026-08-17), and from the sampling of WebLLM
// (https://github.com/mlc-ai/web-llm, src/llm_chat.ts) and Apache TVM (python/tvm/relax/frontend/nn/op.py,
// https://github.com/apache/tvm, commit e0ed4aad), under the Apache License, Version 2.0. Changed as described above.
// Copyright (c) 2023-2025 by MLC LLM Contributors (MLC LLM's NOTICE)
// Copyright 2019-2023 The Apache Software Foundation (Apache TVM's NOTICE)
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// the engine's REPETITION_WINDOW (llama2_numpy.py): the penalty looks at this many of the latest tokens
export const REPETITION_WINDOW = 64;
// the most stop tokens settings hold (the list's models have up to 5, src/models.js)
export const STOPS_MOST = 8;
// State: Step's four words first (copied into the Step uniform after each token), then the loop's own
const STATE = /* wgsl */ `struct State {
  tokens: u32, pos: u32, unused0: u32, unused1: u32,
  token: u32,     // the input of the next pass: EMBED's row
  sampled: u32,   // the tokens sampled so far in this run: the index of the next random number and of chosen[]
  history: u32,   // how long the history is (BOS, the prompt, the sampled tokens): recent[history % WINDOW] is next
  stopped: u32,   // 1 once a stop token was sampled
  recent: array<u32, ${REPETITION_WINDOW}>,
}`;
export const STATE_BYTES = 32 + 4 * REPETITION_WINDOW;
/** T152's review (T160): float32 values into float16 bits, rounded to the nearest (ties to even), as the GPU's own keys
 * and values are (pack2x16float) where it rounds so: a float32 cache's keys and values going up to the GPU. out: a
 * Uint16Array as long as floats. Infinities and NaN stay so, what is past float16's largest becomes an infinity, and
 * what is below half its least subnormal 0. */
const floatWord = new Float32Array(1), floatBits = new Uint32Array(floatWord.buffer);
export function halvesOf(floats, out) {
  for (let i = 0; i < floats.length; i++) {
    floatWord[0] = floats[i];
    const bits = floatBits[0], sign = (bits >>> 16) & 0x8000, exponent = (bits >>> 23) & 0xff, fraction = bits & 0x7fffff;
    let half;
    if (exponent === 0xff) half = 0x7c00 | (fraction ? 0x200 : 0);
    else {
      const e = exponent - 112;  // float16's biased exponent
      if (e >= 0x1f) half = 0x7c00;
      else if (e <= 0) {
        // a subnormal (or 0): the fraction with its leading 1, shifted to units of 2^-24
        const shift = 14 - e;
        if (shift > 24) half = 0;
        else {
          const whole = fraction | 0x800000, rest = whole & ((1 << shift) - 1), middle = 1 << (shift - 1);
          half = whole >>> shift;
          if (rest > middle || (rest === middle && (half & 1))) half += 1;
        }
      } else {
        const rest = fraction & 0x1fff;
        half = (e << 10) | (fraction >>> 13);
        // (a carry into the exponent is right: to the next power of two, or to the infinity at the top)
        if (rest > 0x1000 || (rest === 0x1000 && (half & 1))) half += 1;
      }
    }
    out[i] = sign | half;
  }
  return out;
}
/** The state a run starts from: the token it feeds first, at pos, and the history before it (BOS, the prompt and
 * the tokens sampled before, the fed token last: the penalty's window is its latest REPETITION_WINDOW). length (T152:
 * the engine hands the latest of a longer history over): how long the history is, of which history is the end. */
export function samplingState({ token, pos, history, length = history.length }) {
  const words = new Uint32Array(STATE_BYTES / 4), first = length - history.length;
  words.set([1, pos, 0, 0, token, 0, length, 0]);
  for (let at = Math.max(first, length - REPETITION_WINDOW); at < length; at++) {
    words[8 + (at % REPETITION_WINDOW)] = history[at - first];
  }
  return words;
}
// Sampling, a uniform: the settings of the engine's generate()
const SAMPLING = /* wgsl */ `struct Sampling {
  vocab: u32, stops: u32, unused0: u32, unused1: u32,
  temperature: f32, topp: f32, penalty: f32, unused2: f32,
  stop: array<vec4<u32>, ${STOPS_MOST / 4}>,
}`;
export const SAMPLING_BYTES = 32 + 4 * STOPS_MOST;
/** Sampling's bytes: the engine's settings (temperature 0: the most likely token; top-p outside (0, 1): no nucleus;
 * penalty 1: none) and its stop tokens. */
export function samplingSettings({ vocab, temperature, topp, penalty = 1, stops = [] }) {
  if (stops.length > STOPS_MOST) throw new Error(`more than ${STOPS_MOST} stop tokens`);
  const bytes = new ArrayBuffer(SAMPLING_BYTES), words = new Uint32Array(bytes), values = new Float32Array(bytes);
  words.set([vocab, stops.length]);
  values.set([temperature, topp, penalty], 4);
  words.set(stops, 8);
  return new Uint8Array(bytes);
}

// the token's row of the embedding (int8, groups of 32 with a float32 scale each, 4 to a u32) into the stream x.
// Bindings: 0 the table, 1 its scales, 2 the state (its token), 3 x, 4 the row's width (x of a vec4)
export const EMBED = /* wgsl */ `
${STATE}
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> state: State;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = shape.x;
    let words = n / 4u;
    let row = state.token;
    for (var word = lid.x; word < words; word += 256u) {
        let q_packed = table[row * words + word];
        let d = scales[(row * n + word * 4u) / 32u];
        for (var k = 0u; k < 4u; k++) {
            dst[word * 4u + k] = f32(get_byte_i32(q_packed, k)) * d;
        }
    }
}`;

// What SAMPLE and the stages of the sampling in chunks (T191, below) share: the constants, the workgroup's memory of
// the reductions, and cumsum.wgsl's scan
const SAMPLER_COMMON = /* wgsl */ `
${STATE}
${SAMPLING}
const WG_SIZE = 256u;
const WINDOW = ${REPETITION_WINDOW}u;
// MLC's num_pivots: three pivots a round, the fourth lane of the sums unused
const PIVOTS = 3u;
// the bits of 1.0: the most probable token's probability is exp(0), and none is larger
const ONE_BITS = 0x3F800000u;
// kernel.ts's floor of the nucleus: ln 1e7
const NUCLEUS_FLOOR = 16.118095;
// the largest float below 1: a random number of 1.0 (a float64 just under 1 rounded) would pass every sum
const BELOW_ONE = 0.99999994;
const NONE = 0xFFFFFFFFu;

var<workgroup> best_value: array<f32, WG_SIZE>;
var<workgroup> best_index: array<u32, WG_SIZE>;
var<workgroup> shared_sum: array<f32, WG_SIZE>;
var<workgroup> uniform_word: u32;

// cumsum.wgsl's scan: (the sum of the values of the threads before t, the sum of all), in the same order every run
fn scan(value: f32, t: u32) -> vec2<f32> {
    shared_sum[t] = value;
    workgroupBarrier();
    // upsweep
    var offset = 1u;
    while (offset < WG_SIZE) {
        let idx = (t + 1u) * offset * 2u - 1u;
        if (idx < WG_SIZE) {
            shared_sum[idx] = shared_sum[idx] + shared_sum[idx - offset];
        }
        workgroupBarrier();
        offset <<= 1u;
    }
    let all = shared_sum[WG_SIZE - 1u];
    workgroupBarrier();
    // set last to 0 for exclusive sum
    if (t == 0u) {
        shared_sum[WG_SIZE - 1u] = 0.0;
    }
    workgroupBarrier();
    // downsweep
    offset = WG_SIZE >> 1u;
    while (offset > 0u) {
        let idx = (t + 1u) * offset * 2u - 1u;
        if (idx < WG_SIZE) {
            let x = shared_sum[idx - offset];
            shared_sum[idx - offset] = shared_sum[idx];
            shared_sum[idx] = shared_sum[idx] + x;
        }
        workgroupBarrier();
        offset >>= 1u;
    }
    let before = shared_sum[t];
    workgroupBarrier();
    return vec2<f32>(before, all);
}

// argmax.wgsl's pairs, reduced in the workgroup's memory: the largest value and its first index (of two equal, the
// smaller index)
struct Best { value: f32, at: u32 }
fn best_of(value: f32, at: u32, t: u32) -> Best {
    best_value[t] = value;
    best_index[t] = at;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            let b = best_value[t + offset];
            let bi = best_index[t + offset];
            if (b > best_value[t] || (b == best_value[t] && bi < best_index[t])) {
                best_value[t] = b;
                best_index[t] = bi;
            }
        }
        workgroupBarrier();
        offset >>= 1u;
    }
    let best = Best(best_value[0], best_index[0]);
    workgroupBarrier();
    return best;
}
`;

// The nucleus and the draw from the probabilities gathered over the floor (probs[0 .. count), their tokens in order[]
// in the order of their index), and the token into chosen and the state: SAMPLE's and the chunks' last stage's. The
// shader that takes it binds probs, order, state (read_write), chosen and settings
const SAMPLER_DRAW = /* wgsl */ `
var<workgroup> shared_sums: array<vec4<f32>, WG_SIZE>;
// the search's bounds, in bits, and the sum at lo: thread 0 writes them, every thread reads them by one
// workgroupUniformLoad a round (a barrier, and the value uniform for the loop's own barriers)
struct Bound { lo: u32, hi: u32, sum: f32 }
var<workgroup> bound: Bound;
var<workgroup> found: atomic<u32>;

// soft_max.wgsl's tree, four sums at once (a pivot's each)
fn total4(value: vec4<f32>, t: u32) -> vec4<f32> {
    shared_sums[t] = value;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            shared_sums[t] += shared_sums[t + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }
    let sums = shared_sums[0];
    workgroupBarrier();
    return sums;
}

// top_p_pivot's search, over the bits of the probabilities gathered (probs[0 .. count)): from lo, whose sum S(lo)
// (the sum of the probabilities at or over it) is lo_sum and passes, to hi, which does not, the largest bits whose
// sum passes: at or over limit (the nucleus: strict false), or over it (the draw: strict true). S only changes at a
// probability that is present, so the largest passing bits are one (and the draw's is where the CPU's walk stops:
// S(w) counts every token the sorted walk passes before w's last, and S of the next larger probability is what it
// passed before w's first). Each round narrows (lo, hi) to a quarter or so: 15 rounds from bits 0 to 1.0, 10 to 14
// for the draw from the nucleus's probability, each a read of the count gathered. Returns (those bits, that sum)
fn pivot(lo_start: u32, lo_sum_start: f32, limit: f32, strict: bool, count: u32, t: u32) -> vec2<f32> {
    if (t == 0u) {
        bound = Bound(lo_start, ONE_BITS + 1u, lo_sum_start);
    }
    var b = workgroupUniformLoad(&bound);
    while (b.hi - b.lo > 1u) {
        // three pivots spaced over (lo, hi), in bits: a positive float orders as its bits
        let spacing = max((b.hi - b.lo) / (PIVOTS + 1u), 1u);
        let u = min(vec3<u32>(b.lo + spacing, b.lo + 2u * spacing, b.lo + 3u * spacing), vec3<u32>(b.hi - 1u));
        let f = bitcast<vec3<f32>>(u);
        var acc = vec4<f32>(0.0);
        for (var k = t; k < count; k += WG_SIZE) {
            let p = probs[k];
            acc += select(vec4<f32>(0.0), vec4<f32>(p), vec4<bool>(p >= f.x, p >= f.y, p >= f.z, false));
        }
        let sums = total4(acc, t);
        if (t == 0u) {
            var next = b;
            for (var j = 0u; j < PIVOTS; j++) {
                let passes = select(sums[j] >= limit, sums[j] > limit, strict);
                if (passes && u[j] > next.lo) {
                    next.lo = u[j];
                    next.sum = sums[j];
                } else if (!passes && u[j] < next.hi) {
                    next.hi = u[j];
                }
            }
            bound = next;
        }
        b = workgroupUniformLoad(&bound);
    }
    return vec2<f32>(bitcast<f32>(b.lo), b.sum);
}

// the token into chosen and the state: the next pass's input at the next position, into the window; or the end
fn finish(token: u32) {
    let at = state.sampled;
    chosen[at] = token;
    state.sampled = at + 1u;
    var stop = false;
    for (var j = 0u; j < settings.stops; j++) {
        stop = stop || settings.stop[j / 4u][j % 4u] == token;
    }
    if (stop) {
        state.stopped = 1u;
    } else {
        state.token = token;
        state.pos = state.pos + 1u;
        state.recent[state.history % WINDOW] = token;
        state.history = state.history + 1u;
    }
}

// the nucleus of the count gathered (total: the mass of all over the floor), the draw in it for the random number r,
// and the token into chosen and the state (argmax where no token was found)
fn draw_nucleus(count: u32, total: f32, r: f32, argmax: u32, t: u32) {
    // the nucleus: the smallest probability p whose tokens at or over it hold topp of the mass
    let limit = settings.topp * total;
    let nucleus_found = pivot(0u, total, limit, false, count, t);
    let mass = nucleus_found.y;
    // the draw: the largest probability whose tokens at or over it hold more than r × the nucleus's mass
    let goal = r * mass;
    let drawn = pivot(bitcast<u32>(nucleus_found.x), mass, goal, true, count, t);
    let w = drawn.x;

    // the tokens of probability w, the mass over it, and which of the equal ones: the order of their index
    let run = (count + WG_SIZE - 1u) / WG_SIZE;
    let first = min(t * run, count);
    let last = min(first + run, count);
    var equal = 0u;
    var over = 0.0;
    for (var k = first; k < last; k++) {
        let p = probs[k];
        if (p == w) {
            equal++;
        } else if (p > w) {
            over += p;
        }
    }
    let equals = scan(f32(equal), t);
    let above = scan(over, t).y;
    let ties = u32(equals.y);
    let nth = min(u32(max(floor((goal - above) / w), 0.0)), max(ties, 1u) - 1u);
    var rank = u32(equals.x);
    for (var k = first; k < last; k++) {
        if (probs[k] == w) {
            if (rank == nth) {
                atomicStore(&found, order[k]);
            }
            rank++;
        }
    }
    workgroupBarrier();
    if (t == 0u) {
        let hit = atomicLoad(&found);
        finish(select(hit, argmax, hit == NONE));
    }
}
`;

// The penalty, softmax, top-p and the draw of one token, in one workgroup: see above. Bindings: 0 the logits (the
// penalty is applied to them in place, as the CPU does), 1 and 2 scratch of the vocabulary's size (probabilities and
// the indices of those gathered), 3 the state, 4 chosen, 5 the random numbers, 6 the settings.
export const SAMPLE = /* wgsl */ `
${SAMPLER_COMMON}
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(5) var<storage, read> randoms: array<f32>;
@group(0) @binding(6) var<uniform> settings: Sampling;
${SAMPLER_DRAW}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // a run that stopped changes nothing more
    if (t == 0u) {
        uniform_word = state.stopped;
        atomicStore(&found, NONE);
    }
    if (workgroupUniformLoad(&uniform_word) != 0u) {
        return;
    }

    // the repetition penalty (apply_penalty_inplace's thread a token): each distinct token of the window once
    let window = min(state.history, WINDOW);
    if (settings.penalty != 1.0 && t < window) {
        let token = state.recent[t];
        var seen = false;
        for (var j = 0u; j < t; j++) {
            seen = seen || state.recent[j] == token;
        }
        if (!seen && token < settings.vocab) {
            let value = logits[token];
            logits[token] = select(value * settings.penalty, value / settings.penalty, value > 0.0);
        }
    }
    storageBarrier();

    // a run of consecutive tokens a thread (cumsum.wgsl's): the largest logit and its first index (argmax.wgsl's pairs)
    let vocab = settings.vocab;
    let chunk = (vocab + WG_SIZE - 1u) / WG_SIZE;
    let begin = min(t * chunk, vocab);
    let end = min(begin + chunk, vocab);
    var value = -3.4e38;
    var at = NONE;
    for (var i = begin; i < end; i++) {
        let v = logits[i];
        if (v > value) {
            value = v;
            at = i;
        }
    }
    let largest = best_of(value, at, t);
    let best = largest.value;
    let argmax = largest.at;

    if (settings.temperature == 0.0) {
        if (t == 0u) {
            finish(argmax);
        }
        return;
    }

    // softmax's exp(value - max), at the temperature; with a nucleus only over the floor (kernel.ts)
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let inverse = 1.0 / settings.temperature;
    let lowest = select(-3.4e38, best - settings.temperature * NUCLEUS_FLOOR, nucleus);
    var sum = 0.0;
    var kept = 0u;
    for (var i = begin; i < end; i++) {
        let v = logits[i];
        if (v >= lowest) {
            let p = exp((v - best) * inverse);
            kept++;
            sum += p;
            if (!nucleus) {
                probs[i] = p;
            }
        } else if (!nucleus) {
            probs[i] = 0.0;
        }
    }
    let sums = scan(sum, t);
    let total = sums.y;
    let r = min(randoms[state.sampled], BELOW_ONE);

    if (!nucleus) {
        // the draw in the order of the index: the first token whose running sum passes r × total, else the last
        let goal = r * total;
        var running = sums.x;
        for (var i = begin; i < end; i++) {
            running += probs[i];
            if (running > goal) {
                atomicMin(&found, i);
                break;
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            let hit = atomicLoad(&found);
            finish(select(hit, vocab - 1u, hit == NONE));
        }
        return;
    }

    // the tokens over the floor gathered in the order of their index (their counts scanned): their indices into order,
    // then their probabilities into probs, the same exp() of the same logit as above
    let counts = scan(f32(kept), t);
    let count = u32(counts.y);
    var into = u32(counts.x);
    for (var i = begin; i < end; i++) {
        if (logits[i] >= lowest) {
            order[into] = i;
            into++;
        }
    }
    storageBarrier();
    for (var k = t; k < count; k += WG_SIZE) {
        probs[k] = exp((logits[order[k]] - best) * inverse);
    }
    storageBarrier();
    draw_nucleus(count, total, r, argmax, t);
}`;

// ---- T191: the sampling in chunks of the vocabulary, many workgroups. SAMPLE's one workgroup reads the vocabulary on
// one of the GPU's cores (the owner's Android: 5.9 ms of 11 a token of the CPU section's model, Llama 3's 128256
// tokens), and each of its threads a run of 501 consecutive logits. Here a workgroup a chunk of SAMPLE_CHUNK tokens,
// four consecutive ones a thread (a subgroup's loads side by side), in four dispatches: SAMPLE_MAX (the penalty on the
// window's tokens in the chunk, the chunk's largest logit and its first index), SAMPLE_SUM (every workgroup reduces
// the chunks' largest to the vocabulary's, then its chunk's softmax sum and how many tokens are over the floor; without
// a nucleus its probabilities into probs), SAMPLE_GATHER (with a nucleus: the chunk's tokens over the floor into
// order and probs, after those of the chunks before it, so in the order of the index as SAMPLE gathers them), and
// SAMPLE_PICK (one workgroup: SAMPLE's nucleus and draw on what was gathered, SAMPLER_DRAW; without a nucleus the draw
// in the order of the index, the chunk first by the chunks' sums, then the token in it). The same random number picks
// the same token as SAMPLE and the CPU (sampleLikeCpu) but where a float32 sum in another order moves a border.
// Their forms:
//   - the chunks, and each workgroup reducing all the chunks' partial results again before its own chunk: MLC LLM's
//     two-stage softmax (mlc_llm/compiler_pass/attach_softmax_with_temperature.py, Apache-2.0, above: chunk_lse, the
//     max and sum of each chunk of 4096, and softmax_with_chunked_sum, which merges the chunks' in every block), here
//     with the max of the vocabulary merged before the sums (SAMPLE's exp(value - max) and floor, whose max is the
//     vocabulary's) where MLC merges the sums' log-sum-exp
//   - the largest and its first index: argmax.wgsl's pairs, over a chunk and over the chunks; the sums, the counts and
//     the order of the index: soft_max.wgsl's tree and cumsum.wgsl's scan, as SAMPLE's
// This project's: where a chunk's tokens go in order[] (the tokens over the floor of the chunks before it, summed by
// every workgroup), the penalty applied by the workgroup whose chunk holds the token, and the draw without a nucleus in
// two steps (the chunk whose running sum passes, then the token in it; where the rounding of the two scans leaves none
// passing in that chunk, its last token).
// Bindings, of the same numbers as SAMPLE's (each stage binds those it reads: samplerStages): 0 the logits, 1 probs, 2
// order, 3 the state, 4 chosen, 5 the random numbers, 6 the settings, and 7 the chunks' partial results
// (samplePartsBytes: CHUNKS_COMMON's counted()).
export const SAMPLE_CHUNK = 1024;
export const sampleChunks = (vocab) => Math.ceil(vocab / SAMPLE_CHUNK);
export const samplePartsBytes = (vocab) => 8 * (1 + 2 * sampleChunks(vocab));
const CHUNKS_COMMON = /* wgsl */ `
${SAMPLER_COMMON}
const CHUNK = ${SAMPLE_CHUNK}u;
const EACH = CHUNK / WG_SIZE;

// a run that stopped changes nothing more (the state's word read by all: a barrier, and uniform)
fn stopped(t: u32) -> bool {
    if (t == 0u) {
        uniform_word = state.stopped;
    }
    return workgroupUniformLoad(&uniform_word) != 0u;
}

fn chunk_count() -> u32 {
    return (settings.vocab + CHUNK - 1u) / CHUNK;
}
// the partial results: [0] the vocabulary's largest logit and the floor (SAMPLE_SUM's), [1 + j] chunk j's largest logit
// and its index (SAMPLE_MAX's), [1 + chunks + j] how many of its tokens are over the floor and their sum (SAMPLE_SUM's)
fn counted(j: u32) -> vec2<u32> {
    return parts[1u + chunk_count() + j];
}

// the vocabulary's largest logit and its first index, from the chunks' (SAMPLE_MAX's)
fn largest(t: u32) -> Best {
    let chunks = chunk_count();
    var value = -3.4e38;
    var at = NONE;
    for (var j = t; j < chunks; j += WG_SIZE) {
        let part = parts[1u + j];
        let v = bitcast<f32>(part.x);
        if (v > value) {
            value = v;
            at = part.y;
        }
    }
    return best_of(value, at, t);
}

// soft_max.wgsl's tree: the sum of all the threads' values
fn sum_all(value: f32, t: u32) -> f32 {
    shared_sum[t] = value;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            shared_sum[t] += shared_sum[t + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }
    let all = shared_sum[0];
    workgroupBarrier();
    return all;
}
`;
const CHUNK_MAIN = "@compute @workgroup_size(WG_SIZE)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>)";
export const SAMPLE_MAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read_write> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    // the repetition penalty (apply_penalty_inplace's thread a token) on the window's tokens in this chunk: each
    // distinct token once, by the one workgroup whose chunk holds it
    let window = min(state.history, WINDOW);
    if (settings.penalty != 1.0 && t < window) {
        let token = state.recent[t];
        if (token >= begin && token < end) {
            var seen = false;
            for (var j = 0u; j < t; j++) {
                seen = seen || state.recent[j] == token;
            }
            if (!seen) {
                let value = logits[token];
                logits[token] = select(value * settings.penalty, value / settings.penalty, value > 0.0);
            }
        }
    }
    storageBarrier();
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var value = -3.4e38;
    var at = NONE;
    for (var i = first; i < last; i++) {
        let v = logits[i];
        if (v > value) {
            value = v;
            at = i;
        }
    }
    let best = best_of(value, at, t);
    if (t == 0u) {
        parts[1u + wid.x] = vec2<u32>(bitcast<u32>(best.value), best.at);
    }
}`;
export const SAMPLE_SUM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read_write> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    if (settings.temperature == 0.0) {
        return;
    }
    let best = largest(t).value;
    // SAMPLE's softmax: exp(value - max), at the temperature; with a nucleus only over the floor (kernel.ts)
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let inverse = 1.0 / settings.temperature;
    let lowest = select(-3.4e38, best - settings.temperature * NUCLEUS_FLOOR, nucleus);
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var sum = 0.0;
    var kept = 0u;
    for (var i = first; i < last; i++) {
        let v = logits[i];
        if (v >= lowest) {
            let p = exp((v - best) * inverse);
            kept++;
            sum += p;
            if (!nucleus) {
                probs[i] = p;
            }
        } else if (!nucleus) {
            probs[i] = 0.0;
        }
    }
    // the chunk's sum by the scan SAMPLE_PICK sums its probabilities with, in the same order
    let chunk_sum = scan(sum, t).y;
    let chunk_kept = sum_all(f32(kept), t);
    if (t == 0u) {
        parts[1u + chunk_count() + wid.x] = vec2<u32>(u32(chunk_kept), bitcast<u32>(chunk_sum));
        if (wid.x == 0u) {
            parts[0] = vec2<u32>(bitcast<u32>(best), bitcast<u32>(lowest));
        }
    }
}`;
export const SAMPLE_GATHER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    if (settings.temperature == 0.0 || !nucleus) {
        return;
    }
    // the largest and the floor as SAMPLE_SUM counted the chunks by
    let head = parts[0];
    let best = bitcast<f32>(head.x);
    let lowest = bitcast<f32>(head.y);
    let inverse = 1.0 / settings.temperature;
    // where this chunk's go: after the tokens over the floor of the chunks before it
    var before = 0.0;
    for (var j = t; j < wid.x; j += WG_SIZE) {
        before += f32(counted(j).x);
    }
    let offset = u32(sum_all(before, t));
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var kept = 0u;
    for (var i = first; i < last; i++) {
        if (logits[i] >= lowest) {
            kept++;
        }
    }
    var into = offset + u32(scan(f32(kept), t).x);
    for (var i = first; i < last; i++) {
        let v = logits[i];
        if (v >= lowest) {
            order[into] = i;
            probs[into] = exp((v - best) * inverse);
            into++;
        }
    }
}`;
export const SAMPLE_PICK = /* wgsl */ `
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(5) var<storage, read> randoms: array<f32>;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${SAMPLER_DRAW}
var<workgroup> found_chunk: atomic<u32>;
var<workgroup> chunk_before: f32;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t == 0u) {
        atomicStore(&found, NONE);
        atomicStore(&found_chunk, NONE);
    }
    if (stopped(t)) {
        return;
    }
    let vocab = settings.vocab;
    let chunks = (vocab + CHUNK - 1u) / CHUNK;
    let argmax = largest(t).at;
    if (settings.temperature == 0.0) {
        if (t == 0u) {
            finish(argmax);
        }
        return;
    }
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let r = min(randoms[state.sampled], BELOW_ONE);

    if (!nucleus) {
        // the draw in the order of the index: the chunk whose running sum passes r × total (a run of consecutive
        // chunks a thread, their sums scanned), then the token in it whose does, else the last
        let run = (chunks + WG_SIZE - 1u) / WG_SIZE;
        let lo_chunk = min(t * run, chunks);
        let hi_chunk = min(lo_chunk + run, chunks);
        var own = 0.0;
        for (var j = lo_chunk; j < hi_chunk; j++) {
            own += bitcast<f32>(counted(j).y);
        }
        let sums = scan(own, t);
        let goal = r * sums.y;
        var running = sums.x;
        for (var j = lo_chunk; j < hi_chunk; j++) {
            running += bitcast<f32>(counted(j).y);
            if (running > goal) {
                atomicMin(&found_chunk, j);
                break;
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            uniform_word = atomicLoad(&found_chunk);
        }
        let c = workgroupUniformLoad(&uniform_word);
        if (c == NONE) {
            if (t == 0u) {
                finish(vocab - 1u);
            }
            return;
        }
        // the sum of the chunks before it, from the thread whose run holds it
        if (c >= lo_chunk && c < hi_chunk) {
            var prefix = sums.x;
            for (var j = lo_chunk; j < c; j++) {
                prefix += bitcast<f32>(counted(j).y);
            }
            chunk_before = prefix;
        }
        let prefix = workgroupUniformLoad(&chunk_before);
        let begin = c * CHUNK;
        let end = min(begin + CHUNK, vocab);
        let first = min(begin + t * EACH, end);
        let last = min(first + EACH, end);
        var mine = 0.0;
        for (var i = first; i < last; i++) {
            mine += probs[i];
        }
        var inside = prefix + scan(mine, t).x;
        for (var i = first; i < last; i++) {
            inside += probs[i];
            if (inside > goal) {
                atomicMin(&found, i);
                break;
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            let hit = atomicLoad(&found);
            finish(select(hit, end - 1u, hit == NONE));
        }
        return;
    }

    // how many tokens the chunks gathered over the floor, and their mass
    var kept = 0.0;
    var mass = 0.0;
    for (var j = t; j < chunks; j += WG_SIZE) {
        let part = counted(j);
        kept += f32(part.x);
        mass += bitcast<f32>(part.y);
    }
    let count = u32(sum_all(kept, t));
    draw_nucleus(count, sum_all(mass, t), r, argmax, t);
}`;
/** The sampling in chunks (T191): its four stages in order, each with the bindings it takes (of SAMPLE's numbers and
 * 7 the partial results) and whether it runs a workgroup a chunk (else one workgroup) */
export const SAMPLER_STAGES = [
  { name: "max", code: SAMPLE_MAX, bindings: [0, 3, 6, 7], chunks: true },
  { name: "sum", code: SAMPLE_SUM, bindings: [0, 1, 3, 6, 7], chunks: true },
  { name: "gather", code: SAMPLE_GATHER, bindings: [0, 1, 2, 3, 6, 7], chunks: true },
  { name: "pick", code: SAMPLE_PICK, bindings: [1, 2, 3, 4, 5, 6, 7], chunks: false },
];

// The CPU's sampling in JavaScript (kernels/kernel.ts's penalize() and sample(), as the engine's generate() calls
// them): what SAMPLE is held to (/benchmark/'s check), and itself held to the kernel (tests/smoke.mjs). logits: a
// Float32Array, changed in place by the penalty as the kernel changes them. history: BOS, the prompt and the sampled
// tokens, the token fed last.
export function penalizeLikeCpu(logits, history, penalty) {
  if (penalty === 1) return;
  const f = Math.fround, p = f(penalty);
  for (const token of new Set(history.slice(-REPETITION_WINDOW))) {
    const value = logits[token];
    logits[token] = value > 0 ? f(value / p) : f(value * p);
  }
}
/** The token kernel.ts's sample() draws for random (in [0, 1)): float32 probabilities, float64 sums; the nucleus
 * sorted from the most probable, equal ones in the order of their index (the kernel's quicksort takes them in no set
 * order: either is its distribution). temperature 0: the first index of the largest logit (NumPy's argmax). */
export function sampleLikeCpu(logits, temperature, topp, random) {
  if (temperature === 0) return argmaxLikeCpu(finiteLikeCpu(logits));
  const { tokens, cumulative, mass } = walkLikeCpu(logits, temperature, topp);
  const goal = random * mass;
  for (let k = 0; k < tokens.length; k++) if (cumulative[k] > goal) return tokens[k];
  return tokens[tokens.length - 1];
}
export function argmaxLikeCpu(logits) {
  let best = -Infinity, first = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > best) [best, first] = [logits[i], i];
  return first;
}
/** T195: the engine draws no token when the largest logit is no finite number (a NaN anywhere, +inf anywhere, or
 * all -inf): the kernel returns -1 and NumPy's sample() raises, and the engine stops with an error. So does this: it
 * throws, and returns the logits otherwise. (argmaxLikeCpu alone passes over a NaN, as `>` is false for it.) SAMPLE
 * cannot be held to it: WGSL lets an implementation assume that no NaN nor infinity occurs (§15.7), so what it draws
 * from such logits is the device's; the engine has to find them some other way before it trusts SAMPLE (T152). */
export function finiteLikeCpu(logits) {
  let best = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (Number.isNaN(logits[i])) best = NaN;
    else if (logits[i] > best) best = logits[i];
  }
  if (!Number.isFinite(best)) throw new Error(`the largest logit is ${best}: no token is drawn from logits that are not finite (T195)`);
  return logits;
}
/** The tokens kernel.ts's sample() walks for a random number, in its order (the nucleus's, sorted; else the index's),
 * the running sum after each (float64) and the mass the random number is a share of. */
export function walkLikeCpu(logits, temperature, topp) {
  const f = Math.fround, n = logits.length, best = logits[argmaxLikeCpu(finiteLikeCpu(logits))];
  const nucleus = topp > 0 && topp < 1;
  // without a nucleus the kernel's floor is -f32.MAX_VALUE (and SAMPLE's -3.4e38): a -inf logit is left out, not
  // given the exp(-87) of vexp() below (T195: with a random number of 0 that drew a token at -inf)
  const lowest = nucleus ? f(best - f(f(temperature) * f(16.118095))) : -3.4028234663852886e38, inverse = f(1 / f(temperature));
  const probs = [], index = [];
  for (let i = 0; i < n; i++) {
    if (logits[i] >= lowest) {
      probs.push(f(f(logits[i] - best) * inverse));
      index.push(i);
    }
  }
  // kernel.ts's exp(): four at a time (vexp) holds x at -87 at least, the last count % 4 one by one (fexp) are 0 under
  // it. Only a random number of 0 draws such a token (tests/smoke.mjs found one: T151's review round)
  const simd = probs.length - (probs.length % 4);
  probs.forEach((x, k) => (probs[k] = x < -87 ? (k < simd ? f(Math.exp(-87)) : 0) : f(Math.exp(x))));
  let total = 0, top = 0;
  for (const p of probs) (total += p), (top = Math.max(top, p));
  let order = probs.map((_, k) => k), last = probs.length - 1;
  if (nucleus) {
    // the most probable token always stays (kernel.ts, T178)
    const cutoff = Math.min(((1 - f(topp)) / (probs.length > 1 ? probs.length - 1 : 1)) * total, top);
    order = order.filter((k) => probs[k] >= cutoff).sort((a, b) => probs[b] - probs[a] || index[a] - index[b]);
    const limit = f(topp) * total;
    let sum = 0;
    last = order.length - 1;
    for (let k = 0; k < order.length; k++) {
      sum += probs[order[k]];
      if (sum >= limit) {
        last = k;
        break;
      }
    }
  }
  const tokens = [], cumulative = [];
  let sum = 0;
  for (let k = 0; k <= last; k++) {
    sum += probs[order[k]];
    tokens.push(index[order[k]]);
    cumulative.push(sum);
  }
  return { tokens, cumulative, mass: sum };
}

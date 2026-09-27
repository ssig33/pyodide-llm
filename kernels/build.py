# Build the kernels as Emscripten side modules that Pyodide's ctypes.CDLL can load, without Emscripten:
# compile with AssemblyScript, then prepend the custom section "dylink.0" that Emscripten's loader looks for
# (WASM_DYLINK_MEM_INFO with memory size, memory alignment, table size and table alignment all zero).
#
#   python3 kernels/build.py <output directory>      ->  simdkernel.so, simdkernel_relaxed.wasmlib
#
# The relaxed module is NOT named *.so on purpose: Pyodide's package installer pre-loads every .so of a wheel,
# and a browser without relaxed SIMD fails to compile it.
import subprocess
import sys
from pathlib import Path

ASC = ["npx", "asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1"]


def side_module(source, out, features):
    temporary = Path(out).with_suffix(".tmp.wasm")
    subprocess.run(ASC + [str(source), "-o", str(temporary), "--enable", features], check=True)
    wasm = temporary.read_bytes()
    temporary.unlink()
    # No data section (id 11) may exist: nothing would relocate it, and it would overwrite Pyodide's own memory
    position, section_ids = 8, []
    while position < len(wasm):
        section_ids.append(wasm[position])
        position += 1
        size = shift = 0
        while True:
            byte = wasm[position]
            position += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        position += size
    assert 11 not in section_ids, f"{source} has a data segment: avoid std math, strings and asserts"
    name = b"dylink.0"
    payload = bytes([len(name)]) + name + bytes([1, 4, 0, 0, 0, 0])
    Path(out).write_bytes(wasm[:8] + bytes([0, len(payload)]) + payload + wasm[8:])
    print(out, Path(out).stat().st_size, "bytes")


def plain_module(source, out, features, shared=False, wasm64=False):
    """T93: the same kernels as plain WebAssembly, instantiated by public/forward.js on the memory that holds the
    weights (not Pyodide's). No dylink.0. shared: on a shared memory, for threads that share the weights (stage 2,
    and tests/threads-check.mjs). wasm64 (T101): on a 64-bit memory (kernels/wasm64.mjs), for models past 4 GiB."""
    flags = ["--sharedMemory", "--maximumMemory", "65536"] if shared else []
    if wasm64:
        flags += ["--transform", str(Path(__file__).parent / "wasm64.mjs")]
    subprocess.run(["npx", "asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory",
                    "--initialMemory", "1", *flags, str(source), "-o", str(out),
                    "--enable", features + (",threads" if shared else "") + (",memory64" if wasm64 else "")], check=True)
    print(out, Path(out).stat().st_size, "bytes")


if __name__ == "__main__":
    here, out = Path(__file__).parent, Path(sys.argv[1])
    side_module(here / "kernel.ts", out / "simdkernel.so", "simd")
    side_module(here / "kernel_relaxed.ts", out / "simdkernel_relaxed.wasmlib", "simd,relaxed-simd")
    plain_module(here / "kernel.ts", out / "simdkernel_plain.wasm", "simd")
    plain_module(here / "kernel_relaxed.ts", out / "simdkernel_relaxed_plain.wasm", "simd,relaxed-simd")
    plain_module(here / "kernel.ts", out / "simdkernel_shared.wasm", "simd,relaxed-simd", shared=True)
    plain_module(here / "kernel_relaxed.ts", out / "simdkernel_relaxed_shared.wasm", "simd,relaxed-simd", shared=True)
    # T101: the same on a 64-bit memory
    for shared in (False, True):
        kind = "shared" if shared else "plain"
        plain_module(here / "kernel.ts", out / f"simdkernel_{kind}64.wasm", "simd,relaxed-simd", shared=shared, wasm64=True)
        plain_module(here / "kernel_relaxed.ts", out / f"simdkernel_relaxed_{kind}64.wasm", "simd,relaxed-simd", shared=shared, wasm64=True)
    # T163: the CPU section's ceilings of /benchmark/ (loops, not kernels), on the memory of its made-up model: plain,
    # and shared where the page is cross-origin isolated (the reading threads)
    for shared in (False, True):
        kind = "shared" if shared else "plain"
        plain_module(here / "ceilings.ts", out / f"ceilings_{kind}.wasm", "simd", shared=shared)
        plain_module(here / "ceilings_relaxed.ts", out / f"ceilings_relaxed_{kind}.wasm", "simd,relaxed-simd", shared=shared)

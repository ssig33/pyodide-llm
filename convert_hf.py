# convert_hf.py
# Convert a Hugging Face Llama checkpoint into what llama2_numpy.py loads, with nothing but NumPy:
#   <out>.bin            llama2.c "legacy" checkpoint (7 int header, then the tensors): float32, float16 or int8
#   <out>.tokenizer.bin  llama2.c tokenizer format, from the sentencepiece model or the tokenizer.json
# It runs when the site is deployed, so no converted binary has to live in the repository. The conversion itself
# is public/llama2_convert.py, which the page uses too: this file adds what only the build needs, the files of a
# directory and PyTorch's pickle format.
#
#   python3 convert_hf.py <directory with config.json, pytorch_model.bin | model.safetensors | shards with an index,
#                          spiece.model | tokenizer.model | tokenizer.json> <out> [float32|float16|int8] [max seq_len]
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "public"))
from llama2_convert import (Arrays, Safetensors, Shards, bfloat16, checkpoint_form, checkpoint_header, checkpoint_size,  # noqa: E402
                            convert_weights, normalize, sentencepiece_charsmap, sentencepiece_pieces, tokenizer_bin,
                            tokenizer_json_charsmap, tokenizer_json_options, tokenizer_json_pieces)


# ------------------------------------------------------------------------------------------------ weights
def load_torch_pickle(path):
    """Read a PyTorch zip checkpoint without PyTorch. Only tensor-rebuilding globals are allowed to unpickle."""
    archive = zipfile.ZipFile(path)
    prefix = next(name for name in archive.namelist() if name.endswith("/data.pkl"))[:-len("data.pkl")]
    readers = {"FloatStorage": lambda raw: np.frombuffer(raw, dtype=np.float32),
               "HalfStorage": lambda raw: np.frombuffer(raw, dtype=np.float16), "BFloat16Storage": bfloat16}

    def rebuild_tensor(storage, storage_offset, size, stride, *unused):
        array = storage[storage_offset:storage_offset + int(np.prod(size))].reshape(size)
        assert tuple(stride) == tuple(s // array.itemsize for s in array.strides), "non-contiguous tensor"
        return array

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return dict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if module == "torch" and name in readers:
                return readers[name]
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

        def persistent_load(self, pid):
            _, reader, key, _, _ = pid
            return reader(archive.read(f"{prefix}data/{key}"))

    return Unpickler(archive.open(f"{prefix}data.pkl")).load()


def convert(directory, out_path, dtype, max_seq_len):
    config = json.loads((directory / "config.json").read_text())
    safetensors = directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if safetensors.exists():
        # read piece by piece, never as a whole
        data = np.memmap(safetensors, dtype=np.uint8, mode="r")
        source = Safetensors(lambda offset, length: data[offset:offset + length])
    elif index.exists():
        # split over several files (T105): the index says which; each is read like the one file above
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        maps = [np.memmap(directory / name, dtype=np.uint8, mode="r") for name in files]
        source = Shards([Safetensors(lambda offset, length, data=data: data[offset:offset + length]) for data in maps])
    else:
        source = Arrays(load_torch_pickle(directory / "pytorch_model.bin"))
    # a GPT-2 or a GPT-NeoX has more tensors than a Llama of the same header, a Qwen3 the norms of q and k and maybe
    # heads of another size (T124): the size needs what the header cannot say
    size = checkpoint_size(checkpoint_header(normalize(config), source, max_seq_len), dtype, checkpoint_form(config, source))
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(size,))
    convert_weights(source, config, dtype, max_seq_len, out)
    out.flush()
    return config["vocab_size"]


def convert_tokenizer(directory, out_path, vocab_size):
    model = next((p for p in (directory / "spiece.model", directory / "tokenizer.model") if p.exists()), None)
    if model:
        data = model.read_bytes()
        vocabulary = tokenizer_bin(sentencepiece_pieces(data), vocab_size, charsmap=sentencepiece_charsmap(data))
    else:
        # the page's conversion writes these the same (llama2_convert.Conversion)
        parsed = json.loads((directory / "tokenizer.json").read_text())
        vocabulary = tokenizer_bin(tokenizer_json_pieces(parsed), vocab_size, charsmap=tokenizer_json_charsmap(parsed),
                                   spaces=tokenizer_json_options(parsed)["tokenizer_kind"] != "bytebpe")
    Path(out_path).write_bytes(vocabulary)


if __name__ == "__main__":
    directory, out = Path(sys.argv[1]), sys.argv[2]
    dtype = np.dtype(sys.argv[3] if len(sys.argv) > 3 else "float32")
    max_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    vocab_size = convert(directory, f"{out}.bin", dtype, max_seq_len)
    convert_tokenizer(directory, f"{out}.tokenizer.bin", vocab_size)
    print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes ({dtype}), vocabulary {vocab_size}")

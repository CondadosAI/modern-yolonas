"""TensorRT engine building.

An engine is not a portable artifact. It is compiled for one GPU architecture, one
TensorRT version and one shape profile, and it will refuse to deserialise anywhere
else. That is why this project publishes ONNX and builds the engine on the target
machine, rather than shipping ``.engine`` files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["build_engine", "engine_metadata", "EngineRunner"]

# TensorRT names its dtypes after C, the rest of this project after the benchmark table.
_PRECISION_NAMES = {"float": "fp32", "float32": "fp32", "half": "fp16", "float16": "fp16", "int8": "int8"}


def engine_metadata() -> dict[str, str]:
    """What an engine built right now is tied to, for recording beside the file."""
    import tensorrt as trt
    import torch

    meta = {"tensorrt": trt.__version__, "torch_cuda": torch.version.cuda or "none"}
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["compute_capability"] = f"sm_{major}{minor}"
    return meta


def build_engine(
    onnx_path: str | Path,
    path: str | Path,
    workspace_gb: float = 4.0,
    tf32: bool = True,
    max_batch: int = 1,
    hardware_compatible: bool = False,
    version_compatible: bool = False,
    write_metadata: bool = True,
) -> Path:
    """Compile an ONNX file into a serialised TensorRT engine.

    There is no ``precision`` argument, and its absence is the API. TensorRT 11 builds
    strongly typed networks only — ``BuilderFlag.FP16`` and ``BuilderFlag.INT8`` no
    longer exist — so the engine's precision is whatever the ONNX graph says it is.
    Build an FP16 engine by exporting an FP16 ONNX (``export_onnx(..., half=True)``);
    build an INT8 one from a graph that already carries QuantizeLinear/DequantizeLinear
    nodes, which ``yolonas quantize`` produces.

    Args:
        onnx_path: Source ONNX. Its dtypes decide the engine's precision.
        path: Destination ``.engine``.
        workspace_gb: Tactic workspace ceiling. Too small silently removes fast
            kernels from consideration rather than failing.
        tf32: Allow TF32 for FP32 convolutions. On by default, as TensorRT has it; the
            flag exists so an exact-FP32 comparison can turn it off.
        max_batch: Upper batch bound for the optimisation profile, used only when the
            ONNX has a dynamic batch dimension.
        hardware_compatible: Build with ``HardwareCompatibilityLevel.AMPERE_PLUS``, so
            the engine loads on any Ampere-or-newer GPU (sm_80+) instead of only the
            one it was built on. This is what makes a *pre-built* engine worth
            publishing at all. It is not free: TensorRT drops the kernels that depend
            on a specific architecture, so the engine is slower than a native build
            on the same card — `docs/benchmarks/runtime_matrix.md` measures the gap.
            Requires sm_80+ at build time; a Turing card cannot produce one.
        version_compatible: Let the engine load under a later TensorRT minor version.
            A separate flag with a separate cost, kept apart from
            ``hardware_compatible`` so the two can be priced independently. A fully
            portable published engine wants both.
        write_metadata: Write ``<path>.json`` recording the GPU, compute capability
            and TensorRT version the engine was built against, so a later
            deserialisation failure reads as a mismatch rather than a corrupt file.

    Returns:
        The engine path written.

    Raises:
        RuntimeError: If the ONNX fails to parse, or the build produces no engine.
    """
    import tensorrt as trt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # No EXPLICIT_BATCH: TensorRT 11 removed the flag, implicit batch is gone and
    # every network is explicit-batch and strongly typed.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    # parse_from_file, not parse(bytes): only the file form resolves a `.onnx.data`
    # sidecar, and the byte form's failure reads as a corrupt initializer.
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    if hardware_compatible:
        import torch

        major, _ = torch.cuda.get_device_capability(0)
        if major < 8:
            raise RuntimeError(
                "AMPERE_PLUS engines can only be built on an Ampere-or-newer GPU (sm_80+); "
                f"this one is sm_{major}x"
            )
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

    if version_compatible:
        # Independent of the hardware level, and an independent slowdown: the two are
        # separate flags so the matrix can price them separately.
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)

    # A symbolic batch dimension needs a profile, or the build fails with
    # "input has dynamic shape but no optimization profile".
    inp = network.get_input(0)
    if any(d < 0 for d in inp.shape):
        profile = builder.create_optimization_profile()
        spatial = tuple(inp.shape[1:])
        profile.set_shape(inp.name, (1, *spatial), (1, *spatial), (max_batch, *spatial))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine; re-run with trt.Logger(trt.Logger.VERBOSE) for the reason")

    path.write_bytes(serialized)
    if write_metadata:
        meta = engine_metadata() | {
            "precision": _PRECISION_NAMES.get(str(network.get_input(0).dtype).rsplit(".", 1)[-1].lower(), "unknown"),
            "hardware_compatible": hardware_compatible,
            "version_compatible": version_compatible,
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def _make_output_allocator(trt, torch, device):
    """An allocator TensorRT calls back into for a data-dependent output shape.

    With NMS in the graph the detection count is not known until the kernel has run,
    and `get_tensor_shape` keeps reporting -1 before *and* after execution. The
    allocator callback is the only place TensorRT states the real shape.
    """

    class _Allocator(trt.IOutputAllocator):
        def __init__(self):
            super().__init__()
            self.buffer = None
            self.shape = None

        def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
            if self.buffer is None or self.buffer.numel() < size:
                # Kept as bytes; the caller views it as the tensor's dtype afterwards.
                self.buffer = torch.empty(int(size), dtype=torch.uint8, device=device)
            return self.buffer.data_ptr()

        # TensorRT picks whichever of the two it supports.
        def reallocate_output(self, tensor_name, memory, size, alignment):
            return self.reallocate_output_async(tensor_name, memory, size, alignment, None)

        def notify_shape(self, tensor_name, shape):
            self.shape = tuple(shape)

    return _Allocator()


class EngineRunner:
    """Run a serialised engine, with CUDA buffers owned by torch.

    Torch already holds a CUDA context and a caching allocator, so binding its
    tensors' ``data_ptr()`` straight into TensorRT avoids a second allocator (pycuda
    or cuda-python) and the two contexts fighting over the device.

    Args:
        path: Engine file.
        device: CUDA device the buffers live on.

    Raises:
        RuntimeError: If the engine cannot be deserialised — usually because it was
            built for a different GPU or TensorRT version. The ``.engine.json``
            written beside it records which.
    """

    def __init__(self, path: str | Path, device: str = "cuda:0"):
        import tensorrt as trt
        import torch

        self._trt = trt
        self.device = torch.device(device)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if engine is None:
            raise RuntimeError(
                f"could not deserialise {path}; an engine only loads on the GPU and TensorRT "
                f"version it was built for (see {Path(path).name}.json)"
            )
        self.engine = engine
        self.context = engine.create_execution_context()
        # A non-default stream: enqueueV3 on the default one makes TensorRT insert
        # extra cudaStreamSynchronize calls, which lands in the measured latency.
        self.stream = torch.cuda.Stream(device=self.device)

        self.input_name = None
        self.output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # One allocator per output, kept across calls so the buffer is reused.
        self._allocators = {
            name: _make_output_allocator(trt, torch, self.device) for name in self.output_names
        }

    def _torch_dtype(self, name):
        import torch

        return {
            self._trt.DataType.FLOAT: torch.float32,
            self._trt.DataType.HALF: torch.float16,
            self._trt.DataType.INT32: torch.int32,
        }[self.engine.get_tensor_dtype(name)]

    def __call__(self, images):
        """Run one batch. ``images`` is a CUDA float tensor ``[B, 3, S, S]``."""
        import torch

        images = images.to(self.device).contiguous()
        self.context.set_input_shape(self.input_name, tuple(images.shape))
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        # A graph with NMS in it has a data-dependent output shape, which reads as -1
        # before execution. TensorRT gives an upper bound for exactly this case; the
        # buffer is allocated to it and sliced back afterwards.
        buffers, allocators = {}, {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                allocators[name] = self._allocators[name]
                self.context.set_output_allocator(name, allocators[name])
            else:
                buffer = torch.empty(shape, dtype=self._torch_dtype(name), device=self.device)
                self.context.set_tensor_address(name, buffer.data_ptr())
                buffers[name] = buffer

        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()

        outputs = []
        for name in self.output_names:
            if name in allocators:
                allocator = allocators[name]
                dtype = self._torch_dtype(name)
                count = int(np.prod(allocator.shape)) if allocator.shape else 0
                flat = allocator.buffer[: count * dtype.itemsize].view(dtype)
                outputs.append(flat.view(allocator.shape))
            else:
                outputs.append(buffers[name])
        return tuple(outputs)

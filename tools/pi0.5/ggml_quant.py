"""
GGML K-quant quantization via ctypes.
Supports Q2_K, Q3_K, Q4_K, Q5_K, Q6_K with optional imatrix guidance.
"""

import ctypes
import numpy as np
from pathlib import Path
from gguf import GGMLQuantizationType


# C enum values from ggml.h (match ggml_type enum exactly)
_GGML_TYPE = {
    GGMLQuantizationType.Q2_K: 10,
    GGMLQuantizationType.Q3_K: 11,
    GGMLQuantizationType.Q4_K: 12,
    GGMLQuantizationType.Q5_K: 13,
    GGMLQuantizationType.Q6_K: 14,
}

# All K-quant types supported by this module
KQUANT_TYPES = set(_GGML_TYPE.keys())


class GGMLQuantizer:
    """Wrapper for ggml quantization functions via ctypes."""

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            search_paths = [
                Path(__file__).parent.parent.parent / "build" / "bin" / "libggml.so",
                Path(__file__).parent.parent.parent / "build_oft" / "bin" / "libggml.so",
                Path(__file__).parent.parent.parent / "build_openpi" / "bin" / "libggml.so",
                Path(__file__).parent.parent.parent / "build" / "ggml" / "src" / "libggml.so",
                Path(__file__).parent.parent.parent / "build" / "libggml.so",
                Path("/usr/local/lib/libggml.so"),
                Path("/usr/lib/libggml.so"),
            ]
            for p in search_paths:
                if p.exists():
                    lib_path = str(p)
                    break
            if lib_path is None:
                raise RuntimeError("libggml.so not found. Please specify lib_path or build ggml first.")

        self.lib = ctypes.CDLL(lib_path)
        self._bind_functions()

    def _bind_functions(self):
        # void ggml_quantize_init(enum ggml_type type)
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None

        # size_t ggml_quantize_chunk(
        #     enum ggml_type type,
        #     const float * src,
        #     void * dst,
        #     int64_t start,
        #     int64_t nrows,
        #     int64_t n_per_row,
        #     const float * imatrix   -- NULL if not used
        # )
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,                          # type
            ctypes.POINTER(ctypes.c_float),        # src
            ctypes.c_void_p,                       # dst
            ctypes.c_int64,                        # start
            ctypes.c_int64,                        # nrows
            ctypes.c_int64,                        # n_per_row
            ctypes.POINTER(ctypes.c_float),        # imatrix (can be NULL)
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

        # size_t ggml_row_size(enum ggml_type type, int64_t ne)
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_row_size.restype = ctypes.c_size_t

    def quantize(self, data: np.ndarray, qtype: GGMLQuantizationType,
                 imatrix: np.ndarray | None = None) -> bytes:
        """
        Quantize float32 data to a K-quant format.

        Args:
            data:     Input tensor as float32 numpy array (any shape, last dim = n_per_row)
            qtype:    Target quantization type (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
            imatrix:  Optional importance matrix of shape [n_per_row] (float32).
                      When provided, guides importance-weighted quantization.

        Returns:
            Quantized data as bytes
        """
        if qtype not in _GGML_TYPE:
            raise ValueError(f"GGMLQuantizer supports K-quant types only, got {qtype}. "
                             f"Supported: {list(_GGML_TYPE.keys())}")

        ggml_type = _GGML_TYPE[qtype]

        # Ensure contiguous float32
        data = np.ascontiguousarray(data, dtype=np.float32)

        # K-quants: last dim must be multiple of 256
        if data.shape[-1] % 256 != 0:
            raise ValueError(
                f"{qtype.name} requires last dimension divisible by 256, got {data.shape[-1]}"
            )

        # Flatten to 2D: [nrows, n_per_row]
        if data.ndim == 1:
            nrows = 1
            n_per_row = data.shape[0]
        else:
            nrows = int(np.prod(data.shape[:-1]))
            n_per_row = data.shape[-1]

        data_flat = data.reshape(nrows, n_per_row)

        # Classify imatrix: None | per-column [n_per_row] | per-element [nrows, n_per_row]
        per_element = False
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            if imatrix.ndim == 1 or (imatrix.ndim == 2 and imatrix.shape[0] == 1):
                flat = imatrix.reshape(-1)
                if flat.shape[0] != n_per_row:
                    raise ValueError(
                        f"per-column imatrix length {flat.shape[0]} != n_per_row {n_per_row}")
                imatrix = flat
            elif imatrix.ndim == 2:
                if imatrix.shape != (nrows, n_per_row):
                    raise ValueError(
                        f"per-element imatrix shape {imatrix.shape} != "
                        f"(nrows={nrows}, n_per_row={n_per_row})")
                per_element = True
            else:
                raise ValueError(f"imatrix must be 1-D or 2-D, got ndim={imatrix.ndim}")

        # Initialize quantization tables
        self.lib.ggml_quantize_init(ggml_type)

        # Calculate output size
        row_size = self.lib.ggml_row_size(ggml_type, n_per_row)
        total_size = row_size * nrows

        # Allocate output buffer
        dst = (ctypes.c_uint8 * total_size)()
        src_ptr = data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if per_element:
            # Per-weight Fisher: quantize row-by-row, each row gets its own
            # importance slice (mirrors quantize-vision.cpp / llama-quant.cpp
            # imatrix_is_per_weight path).
            dst_addr = ctypes.addressof(dst)
            written = 0
            for r in range(nrows):
                row_src = data_flat[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                row_im = np.ascontiguousarray(imatrix[r], dtype=np.float32)
                row_im_ptr = row_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                written += self.lib.ggml_quantize_chunk(
                    ggml_type, row_src,
                    ctypes.c_void_p(dst_addr + r * row_size),
                    0, 1, n_per_row, row_im_ptr,
                )
        else:
            imatrix_ptr = (imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                           if imatrix is not None else None)
            written = self.lib.ggml_quantize_chunk(
                ggml_type, src_ptr,
                ctypes.cast(dst, ctypes.c_void_p),
                0, nrows, n_per_row, imatrix_ptr,
            )

        if written != total_size:
            raise RuntimeError(
                f"Quantization size mismatch: expected {total_size}, got {written}"
            )

        return bytes(dst)

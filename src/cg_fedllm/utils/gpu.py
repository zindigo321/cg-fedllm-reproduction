"""GPU memory guard for Windows (WDDM).

Under WDDM the driver silently pages device allocations that exceed dedicated VRAM into shared system
memory instead of failing, which can slow a job down by more than 10x while it still "works". Capping the
PyTorch caching allocator at the dedicated memory that is currently free makes the allocator release its
cached blocks and retry before it would grow past the cap; a real excess then raises ``OutOfMemoryError``.
Numerics are unaffected.
"""

from __future__ import annotations

import gc

import torch


def cap_allocator_to_free_vram(margin_bytes: int = 256 * 2**20, device: int = 0) -> dict[str, int | float]:
    """Limit this process's caching allocator to the currently free dedicated VRAM minus ``margin_bytes``."""
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    cap = max(free - margin_bytes, 0)
    fraction = cap / total
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {
        "device_total_bytes": int(total),
        "free_at_start_bytes": int(free),
        "margin_bytes": int(margin_bytes),
        "allocator_cap_bytes": int(cap),
        "allocator_cap_fraction": round(fraction, 6),
    }

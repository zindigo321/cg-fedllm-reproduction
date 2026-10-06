"""Child process for the P5-A hard-kill tests: runs one synthetic invocation and ends it with ``os._exit``.

``os._exit`` ends the process at once: no ``finally`` block, no ``except`` handler, no ``atexit`` hook and no
interpreter cleanup runs, unlike ``SystemExit``. Usage: ``python -m tests.unit.p5a_hard_kill <root> <mode>``.

Modes (each a real P5-A boundary):
* ``during_first_record_write``: the temporary of ``ledger/invocation_1.json`` is written, then the process dies
  before the link (an abandoned first publication, v2.1 A2);
* ``during_start_write``: the temporary of the start record is written, then the process dies before the link;
* ``after_start``: the start record is finalized, then the process dies inside ``ae.to(device)``;
* ``after_end``: the end record is finalized, then the process dies before the checkpoint is linked.

Synthetic tensors only; CPU only; nothing outside ``<root>`` is written.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KILL_CODE = 86


def main(root: str, mode: str) -> None:
    from cg_fedllm.phase5 import p5a_run as run
    from cg_fedllm.utils.seeding import configure_determinism
    from tests.unit.p5a_fixtures import all_pops, context, loader, preflight

    # the same settings as the autouse fixture of tests/conftest.py, so the child's preflight evidence (Phi bytes)
    # is bitwise identical to what the parent test process reconstructs
    configure_determinism(True, 2)

    pops = all_pops()
    ctx = context(Path(root), pops)
    preflight(ctx, pops)
    real_writer, real_linker = ctx.store.writer, ctx.store.linker

    def writer(path, data):
        real_writer(path, data)
        if mode == "during_start_write" and path.name.startswith(".start_R2_single_snapshot"):
            os._exit(KILL_CODE)
        if mode == "during_first_record_write" and path.name.startswith(".invocation_1.json"):
            os._exit(KILL_CODE)  # ledger/ and the fsynced temporary exist; invocation_1.json was never linked

    def linker(src, dst):
        if mode == "after_end" and dst.name == "autoencoder_p5a_final.safetensors":
            os._exit(KILL_CODE)
        real_linker(src, dst)

    ctx.store.writer, ctx.store.linker = writer, linker
    if mode == "after_start":
        real_trainer = ctx.trainer

        def trainer(ae, xs, norm, **kw):
            class Dies(type(ae)):
                def to(self, *a, **k):  # dies exactly at the timing boundary
                    os._exit(KILL_CODE)

            ae.__class__ = Dies
            return real_trainer(ae, xs, norm, **kw)

        ctx.trainer = trainer
    run.run_invocation(ctx, loader(pops))
    os._exit(0)  # the kill point was not reached


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Opt-in SHARP pool-quality check; ACDS sharing and CART remain untouched.

The selected average is not guaranteed to preserve the loss of its best member
for a nonlinear network. Check the realized vector on the ranking batch and
fall back to that coherent member when averaging increases its measured CE.
This is a new selection variant, not a correction to the definition of ACDS.
"""
import numpy as np
from basil_core.sharp_protocol import link_vector


def selector(original, mode='checked'):
    if mode not in ('raw', 'checked', 'single'):
        raise ValueError('Unknown pool diagnostic mode.')

    def choose(cfg, state, worker, x, y, node, round_id, batch):
        pool, evidence = original(cfg, state, worker, x, y, node, round_id, batch)
        senders = evidence['selectedSenders']
        record = dict(mode=mode, originalSelectedCount=evidence['selectedCount'],
                      checked=False, fallback=False)
        if not senders:
            return pool, dict(evidence, poolQuality=record)
        sender = senders[0]
        best = link_vector(state['outbound'][sender], cfg['seed'], int(state['senderRounds'][sender]),
                           sender, node, cfg['sigma'])
        losses = np.asarray(worker.score_many(np.stack((pool, best)), x[batch], y[batch]), dtype=float)
        if losses.shape != (2,) or not np.isfinite(losses).all():
            raise FloatingPointError('Pool comparison requires two finite model losses.')
        fallback = bool(mode == 'single' or (mode == 'checked' and losses[0] > losses[1] + 1e-12))
        record.update(checked=True, poolCE=float(losses[0]), bestMemberCE=float(losses[1]),
            poolExcessCE=float(losses[0] - losses[1]), fallback=fallback,
            poolNorm=float(np.linalg.norm(pool.astype(np.float64))),
            bestMemberNorm=float(np.linalg.norm(best.astype(np.float64))))
        if fallback:
            evidence = dict(evidence, selectedSenders=[sender], selectedCount=1, targetCount=1)
        # selectedCount follows the actual vector, so the protocol's existing
        # RLA calculation uses q=1 when it uses a single incoming noisy copy.
        return (best if fallback else pool), dict(evidence, poolQuality=record)
    return choose

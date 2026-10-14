"""Isolated helper-threat diagnostics, NOT an executable research defense.

Coordinate-wise b-trimmed means constrain arbitrary outliers when n > 2b.
They do not authenticate identities, guarantee descent on heterogeneous data,
restore a withheld class, or implement our old class-mean scalar objective.
No training worker, queue, optimizer, EBM or cloud service is wired here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

REVISION = 'cart_untrusted_helpers_trimmed_diagnostic_v1'


def _owners(available_owners, bound):
    owners = tuple(available_owners)
    if (type(bound) is not int or bound < 0 or len(set(owners)) != len(owners) or
            any(type(c) is not int or not 0 <= c < 10 for c in owners)):
        raise ValueError('Distinct logical owners and nonnegative Byzantine bound required.')
    if len(owners) <= 2*bound:
        raise ValueError(f'Insufficient robust quorum: {len(owners)} responses; need at least {2*bound+1}.')
    return tuple(sorted(owners))


def trimmed_coordinates(rows, *, bound):
    """Discard b lowest/highest values per coordinate; no magnitude clipping.

With at most b arbitrary rows and more than 2b distinct authenticated senders,
each output coordinate lies within the honest coordinate range. That range
may be wide or point away from a useful multiclass descent direction.
"""
    import numpy as np
    values = np.asarray(rows, dtype=np.float64)
    if (type(bound) is not int or bound < 0 or values.ndim != 2 or
            values.shape[0] <= 2*bound or not values.shape[1]):
        raise ValueError('A nonempty coordinate matrix with n > 2b is required.')
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite helper payload; no silent repair or quorum reduction.')
    ordered = np.sort(values, axis=0)
    retained = ordered[bound:len(values)-bound] if bound else ordered
    # Divide first to avoid unnecessary overflow on large but finite payloads.
    mean = (retained/len(retained)).sum(axis=0)
    if not np.isfinite(mean).all():
        raise FloatingPointError('Non-finite robust aggregate.')
    return mean


def robust_owner_gradients(params, responses, *, receiver, round_id,
                           optimizer_step, available_owners, bound=4):
    """Diagnostic estimator only: exact identity/context checks, then trimming.

There is deliberately no attacker-ID argument and no oracle exclusion. The
caller must bind identities to authenticated transport; a Python owner field
is not authentication. Missing/duplicate responses fail closed.
"""
    import numpy as np
    from .cart_owner_gradients import OwnerGradientResponse
    from .research_protocol import params_hash
    owners = _owners(available_owners, bound)
    if type(receiver) is not int or receiver not in owners or any(
            type(v) is not int or v < 0 for v in (round_id, optimizer_step)):
        raise ValueError('Valid receiver and current round/step required.')
    responses = tuple(responses)
    if (any(not isinstance(r, OwnerGradientResponse) for r in responses) or
            len(responses) != len(owners) or {r.owner for r in responses} != set(owners)):
        raise ValueError('Missing, duplicate or unexpected gradient owner.')
    current = [np.asarray(p) for p in params]
    if not current or any(p.dtype != np.float32 or not p.size or not np.isfinite(p).all() for p in current):
        raise ValueError('Finite float32 current parameter tensors required.')
    digest = params_hash(current)
    ordered = sorted(responses, key=lambda r: r.owner)
    for r in ordered:
        if (r.receiver, r.round_id, r.optimizer_step, r.model_hash) != (receiver, round_id, optimizer_step, digest):
            raise ValueError('Stale or wrong-context gradient response.')
        if len(r.gradients) != len(current) or any(g.shape != p.shape for g, p in zip(r.gradients, current)):
            raise ValueError('Wrong gradient tensor layout.')
    gradients = tuple(trimmed_coordinates([r.gradients[j].ravel() for r in ordered], bound=bound)
                      .astype(np.float32).reshape(p.shape) for j, p in enumerate(current))
    return gradients, dict(revision=REVISION, bound=bound, availableOwners=list(owners),
        retainedValuesPerCoordinate=len(owners)-2*bound, modelHash=digest,
        authenticatedTransportImplemented=False, honestGradientVerified=False,
        equivalentToClassMeanObjective=False, researchReady=False)


@dataclass(frozen=True)
class SelectionScoreResponse:
    """One owner's score vector for the same five candidates and TRAIN request."""
    owner: int
    receiver: int
    round_id: int
    request_hash: str
    candidate_hashes: tuple[str, ...]
    losses: tuple[float, ...]

    def __post_init__(self):
        if (any(type(v) is not int or not 0 <= v < 10 for v in (self.owner, self.receiver)) or
                type(self.round_id) is not int or self.round_id < 0):
            raise ValueError('Valid owner, receiver and nonnegative round required.')
        if (type(self.candidate_hashes) is not tuple or len(self.candidate_hashes) != 5 or
                any(not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None
                    for h in (*self.candidate_hashes, self.request_hash))):
            raise ValueError('Exact request and five candidate hashes required.')
        if type(self.losses) is not tuple or len(self.losses) != 5 or any(
                not math.isfinite(v) or v < 0 for v in self.losses):
            raise ValueError('Five finite nonnegative CE claims required.')


def robust_selection(candidates, responses, *, receiver, round_id, request_hash,
                     available_owners, bound=4):
    """Trim score claims per candidate; retain nearest-predecessor tie breaking.

This is NOT BASIL selection or an honesty certificate. Binding hashes rejects
stale/swapped replies, not a malicious owner that lies about its own CE.
"""
    from .research_protocol import params_hash
    owners = _owners(available_owners, bound)
    if (type(receiver) is not int or receiver not in owners or type(round_id) is not int or round_id < 0 or
            not isinstance(request_hash,str) or re.fullmatch('[0-9a-f]{64}',request_hash) is None):
        raise ValueError('Valid current receiver/round/request required.')
    candidates, responses = tuple(candidates), tuple(responses)
    if len(candidates) != 5 or {s.sender_id for s in candidates} != {(receiver-d)%10 for d in range(1,6)}:
        raise ValueError('Five distinct counterclockwise candidate senders required.')
    for s in candidates:
        expected_round = round_id if s.sender_id < receiver else round_id-1
        if type(s.round_id) is not int or s.round_id != expected_round:
            raise ValueError('Wrong rolling snapshot round.')
    hashes = tuple(params_hash(s.params) for s in candidates)
    if (any(not isinstance(r,SelectionScoreResponse) for r in responses) or len(responses) != len(owners) or
            {r.owner for r in responses} != set(owners)):
        raise ValueError('Missing, duplicate or unexpected score owner.')
    for r in responses:
        if (r.receiver,r.round_id,r.request_hash,r.candidate_hashes) != (receiver,round_id,request_hash,hashes):
            raise ValueError('Stale, swapped or wrong-context candidate scores.')
    values = trimmed_coordinates([r.losses for r in sorted(responses,key=lambda r:r.owner)],bound=bound)
    minimum = float(min(values))
    chosen = min((i for i,v in enumerate(values) if v <= minimum+1e-8),
                 key=lambda i:(receiver-candidates[i].sender_id)%10)
    return candidates[chosen], dict(revision=REVISION, bound=bound,
        scores=[float(v) for v in values], selectedIndex=chosen, selectedSender=candidates[chosen].sender_id,
        retainedValuesPerCandidate=len(owners)-2*bound, honestScoresVerified=False,
        authenticatedTransportImplemented=False, researchReady=False)


def full_node_readiness():
    """No automatic production promotion from bounded-payload unit tests."""
    return dict(revision=REVISION, researchReady=False, queueStarted=False, trainingRunsStarted=0,
        requiredScope='full_node_threats', byzantineBound=4, minimumResponses=9,
        firstActivationAvailableResponses=1, postWarmupResponses=10,
        approvedAttackStartRound=20, trustedWarmupCanAvoidStartupQuorum=True,
        postWarmupRetainedCoordinates=2, honestOwnersIfFourWithhold=6,
        blockers=[
            'Always-on trimming cannot start at Round 0; a mean-to-robust switch after the honest R0-19 warmup is not validated.',
            'Four withheld responses leave six, below the 9-response quorum.',
            'Warmup snapshots may retain class knowledge, but there is no independent fresh source for four malicious/withheld class objectives.',
            'Coordinate range bounds do not establish multiclass descent or useful selection rankings.',
            'Authenticated identities and trustworthy request/probe provenance are not deployed.',
            'A Byzantine active trainer may ignore SGD; snapshot-only Hidden corruption does not simulate that.',
            'Helper-link noise variance/transport is unspecified; snapshot sigma does not define gradient noise.',
            'Trimmed gradients are not the derivative of the validated class-mean loss; EBM composition is unvalidated.',
        ])

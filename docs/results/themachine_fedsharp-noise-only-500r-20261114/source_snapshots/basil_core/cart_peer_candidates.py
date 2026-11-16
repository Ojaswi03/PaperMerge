"""Explicit coherent-peer eligibility under the SHARP survivor fallback."""


def eligible_senders(cfg,state,node,selection,*,guard_fallback=False):
    senders=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1)
        if state['senderRounds'][(node-d)%cfg['nodes']]>=0]
    screened=bool(selection.get('screenApplied'))
    passing=[s for j,s in enumerate(senders) if not screened or
        selection['peakRatios'][j]<=selection['peakThreshold']]
    fallback=bool(guard_fallback and screened and
        (selection.get('screenGuardFallback') or selection.get('screenAllFailFallback')))
    allowed=senders if fallback else passing
    ordered=sorted(allowed,key=lambda s:(-int(state['senderRounds'][s]),(node-s)%cfg['nodes']))
    return ordered,dict(peerFallbackApplied=fallback,screenPassingPeerCount=len(passing),
        eligiblePeerCount=len(ordered),peerEligibility='guard-consistent fallback' if guard_fallback else 'screen pass only')

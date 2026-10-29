"""Fixed public labels for comparison monitors; no scientific configuration edits."""
ARMS=dict(sharp='SHARP',cart='CART original',macro='CART macro',acds='SHARP+ACDS',
    prior_cart='CART+prior',prior_macro='Prior macro',prior_wide='Prior wide',
    prior_fallback='SHARP+CART',cart_fallback='CART fallback',balanced_commit='Balanced check',
    ingress_only='Ingress only',commit_only='Commit only',local_only='Local only',
    fresh_peer='Fresh peer',no_calibration='No calibration',no_repair='No head repair')
ENVIRONMENTS=dict(joint='attack+noise',clean='clean',noise_only='noise only',attack_only='attack only')


def comparison_label(cfg,*,include_environment=False):
    case=cfg.get('comparisonCase',{});arm=case.get('arm',cfg.get('comparisonLabel'))
    name=ARMS.get(arm)
    if name and include_environment:
        env=ENVIRONMENTS.get(case.get('environment'))
        if env:name+=' / '+env
    return name

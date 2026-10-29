"""Fixed public labels for comparison monitors; no scientific configuration edits."""
ARMS=dict(sharp='SHARP',cart='CART original',macro='CART macro',acds='SHARP+ACDS',
    prior_cart='CART+prior',prior_macro='Prior macro',prior_wide='Prior wide',
    prior_fallback='SHARP+CART',cart_fallback='CART fallback',balanced_commit='Balanced check',
    ingress_only='Ingress only',commit_only='Commit only',local_only='Local only',
    fresh_peer='Fresh peer',no_calibration='No calibration',no_repair='No head repair')
ENVIRONMENTS=dict(joint='attack+noise',clean='clean',noise_only='noise only',attack_only='attack only')
ARMS.update(cart_full='SHARP+CART', acds_checked='ACDS checked', sharp_checked='SHARP checked',
    cart_macro='CART macro', cart_ingress='CART ingress', cart_commit='CART commit',
    cart_no_prior='CART no prior', cart_no_calibration='CART no calib',
    cart_no_repair='CART no head', acds_single='ACDS single')


def comparison_label(cfg,*,include_environment=False):
    case=cfg.get('comparisonCase',{});arm=case.get('arm',cfg.get('comparisonLabel'))
    arm=case.get('studyVariant',{}).get('name',arm)
    name=ARMS.get(arm)
    if name and include_environment:
        env=ENVIRONMENTS.get(case.get('environment'))
        if env:name+=' / '+env
    return name

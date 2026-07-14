"""
Plotting for GUI experiment results.
Auto-discovers all experiments saved in experiments/results/gui/
and generates comparison plots per data split / dataset / attack type / approach.

Folder structure (results):
  experiments/results/gui/{split}/{dataset}/{attackKey}/{approach}/
  experiments/results2/gui/{split}/{dataset}/{attackKey}/{merged|cart}/
      acc_*.npy
      config_*.json
  Legacy results without split are still read from:
  experiments/results/gui/{dataset}/{attackKey}/{approach}/

Folder structure (plots):
  plots/images/gui/{split}/{dataset}/{attackKey}/{approach}/
  plots2/images/gui/{split}/{dataset}/{attackKey}/{merged|cart}/
  plots2/images/gui/{split}/{dataset}/{attackKey}/mitigation_sweep_comparison_avg.png
      experiments_avg.png
      experiments_avg_zoom.png
      grid_avg.png
      final_accuracy_avg.png
      improvement_over_no_mitigation_avg.png
      ablation_groups_avg.png

CONFIGURATION: Edit the variables below to customize your plots
"""
import os
import sys
import json
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR PLOTS
# ============================================================================

# Which datasets to plot? Options: "mnist", "cifar10", "nmnist"
# Set to None to auto-detect all available datasets
DATASETS_TO_PLOT = None

# Which metric? Options: "avg" only (worst is no longer saved)
METRIC = "avg"
DATA_SPLITS = ("nonIID", "IID")
PLOT_ROOT = "plots"
PLOT_ROOT_MERGED_CART = "plots2"
PLOTS2_APPROACHES = {"merged", "cart"}
RESULT_ROOT = os.path.join("experiments", "results", "gui")
RESULT_ROOT_MERGED_CART = os.path.join("experiments", "results2", "gui")

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')

# All distinct matplotlib markers
_ALL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p', 'H', '8', '+', 'x', '1', '2', '3', '4']
_CLEAN_COLORS = [
    '#111827',  # charcoal
    '#f97316',  # orange
    '#7c3aed',  # violet
    '#059669',  # emerald
    '#dc2626',  # red
    '#0891b2',  # cyan
    '#ca8a04',  # amber
]
_CLEAN_LINESTYLES = [
    (0, (8, 3)),
    (0, (3, 2, 1, 2)),
    (0, (1, 2)),
    (0, (6, 2, 2, 2)),
    (0, (10, 2)),
    (0, (4, 4)),
    (0, (2, 1)),
]

ENV_ORDER = ['Clean', 'Channel Noise Only', 'Byzantine Only', 'Byzantine + Channel Noise']
METHOD_ORDER = ['No Mitigation', 'SS', 'EBM', 'SS + EBM']
METHOD_COLORS = {
    'No Mitigation': '#6b7280',
    'SS': '#2563eb',
    'EBM': '#f97316',
    'SS + EBM': '#16a34a',
}


def getColors(n):
    # pick colormap based on how many colors are needed
    if n <= 0:
        return []
    if n <= 10:
        cmap = cm.get_cmap('tab10', n)
    elif n <= 20:
        cmap = cm.get_cmap('tab20', n)
    else:
        cmap = cm.get_cmap('hsv', n)
    return [cmap(i) for i in range(n)]


def getMarkers(n):
    # cycle through marker list to get n markers
    return [m for _, m in zip(range(n), itertools.cycle(_ALL_MARKERS))]


def splitFromConfig(config):
    return "nonIID" if config.get("nonIID", True) else "IID"


def _resultRoot(split=None, approach=None):
    base = RESULT_ROOT_MERGED_CART if approach in PLOTS2_APPROACHES else RESULT_ROOT
    return os.path.join(base, split) if split else base


def _allResultRoots(split=None):
    roots = []
    for base in (RESULT_ROOT, RESULT_ROOT_MERGED_CART):
        if split:
            roots.append(os.path.join(base, split))
        else:
            roots.append(base)
            roots.extend(os.path.join(base, sp) for sp in DATA_SPLITS)
    return roots


def _isUnder(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _plotRootForApproach(approach):
    return PLOT_ROOT_MERGED_CART if approach in PLOTS2_APPROACHES else PLOT_ROOT


def _plotRootForComparison(approaches):
    approaches = set(approaches or [])
    return PLOT_ROOT_MERGED_CART if approaches & PLOTS2_APPROACHES else PLOT_ROOT


def _plotDir(split, dataset, attackKey, approach, subdir=None):
    parts = [_plotRootForApproach(approach), "images", "gui", split, dataset, attackKey, approach]
    if subdir:
        parts.append(subdir)
    return os.path.join(*parts)


def _plotPath(split, dataset, attackKey, approach, filename, subdir=None):
    return os.path.join(_plotDir(split, dataset, attackKey, approach, subdir), filename)


def _experimentSourcePaths(experiments):
    paths = []
    for exp in experiments or []:
        for key in ('avgPath', 'configPath'):
            path = exp.get(key)
            if path and os.path.exists(path):
                paths.append(path)
    return paths


def _shouldSkipPlot(savePath, skipExisting=False, sourcePaths=None):
    if skipExisting and os.path.exists(savePath):
        sourcePaths = [p for p in (sourcePaths or []) if os.path.exists(p)]
        if sourcePaths:
            newestSource = max(os.path.getmtime(p) for p in sourcePaths)
            if newestSource > os.path.getmtime(savePath):
                return False
        print(f"Exists, skipping: {savePath}")
        return True
    return False


def noiseBucket(config):
    if not config.get('useChannelNoise', False):
        return "no_channel_noise"
    sigma = float(config.get('channelNoiseSigma', 0.0))
    return f"sigma_{sigma:.1f}".replace('.', '_')


def groupExperimentsByNoiseBucket(experiments):
    buckets = {}
    noNoise = [exp for exp in experiments if noiseBucket(exp['config']) == "no_channel_noise"]
    for exp in sorted(experiments, key=lambda exp: _noiseSortValue(exp['config'])):
        bucket = noiseBucket(exp['config'])
        if bucket == "no_channel_noise":
            continue
        buckets.setdefault(bucket, [])
        buckets[bucket].append(exp)
    if noNoise:
        buckets["no_channel_noise"] = noNoise
        for bucket in list(buckets):
            if bucket != "no_channel_noise":
                buckets[bucket] = noNoise + buckets[bucket]
    return {
        bucket: buckets[bucket]
        for bucket in sorted(buckets, key=_noiseBucketSortValue)
    }


def _iterConfigFiles(split=None):
    """Yield config paths from split-aware and legacy GUI result layouts."""
    roots = _allResultRoots(split)

    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for configPath in glob.glob(os.path.join(root, "**", "config_*.json"), recursive=True):
            if configPath in seen:
                continue
            seen.add(configPath)
            try:
                with open(configPath, 'r') as f:
                    config = json.load(f)
            except Exception:
                continue
            isResults2 = _isUnder(configPath, RESULT_ROOT_MERGED_CART)
            if config.get('approach') in PLOTS2_APPROACHES:
                if not isResults2:
                    continue
            elif isResults2:
                continue
            if split and splitFromConfig(config) != split:
                continue
            yield configPath, config


def _pathPartsAfterGui(path):
    normPath = os.path.normpath(path)
    for root in (RESULT_ROOT_MERGED_CART, RESULT_ROOT):
        normRoot = os.path.normpath(root)
        if _isUnder(normPath, normRoot):
            rel = os.path.relpath(normPath, normRoot)
            return rel.split(os.sep)
    rel = os.path.relpath(normPath, os.path.normpath(RESULT_ROOT))
    return rel.split(os.sep)


def _metadataFromConfigPath(configPath):
    parts = _pathPartsAfterGui(configPath)
    if parts and parts[0] in DATA_SPLITS:
        parts = parts[1:]
    if len(parts) < 4:
        return None
    return {
        'dataset': parts[0],
        'attackKey': parts[1],
        'approach': parts[2],
    }


def discoverDataSplits():
    splits = set()
    for split in DATA_SPLITS:
        if os.path.isdir(_resultRoot(split)):
            splits.add(split)
        if os.path.isdir(_resultRoot(split, approach='cart')):
            splits.add(split)
    for _, config in _iterConfigFiles():
        splits.add(splitFromConfig(config))
    return [sp for sp in DATA_SPLITS if sp in splits]


def discoverDatasets(split=None):
    # scan gui results directory for dataset subdirs
    # supports split-aware and legacy structures
    guiDir = _resultRoot(split)
    if split and not os.path.isdir(guiDir) and not os.path.isdir(_resultRoot()):
        print(f"No GUI results directory found at {guiDir}")
        return []

    datasets = set()
    for configPath, _ in _iterConfigFiles(split):
        meta = _metadataFromConfigPath(configPath)
        if meta:
            datasets.add(meta['dataset'])

    return sorted(datasets)


def discoverAttackTypes(dataset, split=None):
    # return attack type subfolders that have results (new or old structure)
    attackTypes = set()
    for configPath, _ in _iterConfigFiles(split):
        meta = _metadataFromConfigPath(configPath)
        if meta and meta['dataset'] == dataset:
            attackTypes.add(meta['attackKey'])

    return sorted(attackTypes)


def discoverApproaches(dataset, attackKey, split=None):
    # return approach subfolders (basil / noisy / merged) that have npy files
    approaches = set()
    legacyFlat = False
    for configPath, _ in _iterConfigFiles(split):
        meta = _metadataFromConfigPath(configPath)
        if not meta or meta['dataset'] != dataset or meta['attackKey'] != attackKey:
            continue
        approaches.add(meta['approach'])
        if meta['approach'].startswith("config_"):
            legacyFlat = True

    if legacyFlat:
        approaches.discard(next((ap for ap in approaches if ap.startswith("config_")), ""))
        approaches.add("_legacy")

    return sorted(approaches)


def discoverExperiments(dataset, attackKey=None, approach=None, split=None):
    # find config JSON files and pair each with its .npy accuracy file
    experiments = []
    seen = set()
    for configPath, config in _iterConfigFiles(split):
        meta = _metadataFromConfigPath(configPath)
        if not meta:
            continue
        if dataset and meta['dataset'] != dataset:
            continue
        if attackKey and meta['attackKey'] != attackKey:
            continue
        if approach and approach != "_legacy" and meta['approach'] != approach:
            continue
        if approach == "_legacy" and meta['approach'] not in ("", "_legacy"):
            continue

        resultDir = os.path.dirname(configPath)
        configName = os.path.basename(configPath)
        name = configName.replace("config_", "").replace(".json", "")

        avgPath = os.path.join(resultDir, f"acc_{name}.npy")
        if not os.path.exists(avgPath):
            continue
        if not _matchesCurrentConfig(config):
            continue

        dedupeKey = (splitFromConfig(config), meta['dataset'], meta['attackKey'], meta['approach'], name)
        if dedupeKey in seen:
            continue
        seen.add(dedupeKey)

        experiments.append({
            'name': name,
            'config': config,
            'configPath': configPath,
            'avgPath': avgPath,
            'label': buildLabel(config),
            'split': splitFromConfig(config),
        })

    return experiments


ATTACK_FLAGS = (
    'attackGaussian',
    'attackSignFlip',
    'attackHidden',
    'attackModelPoison',
    'attackScaling',
    'attackAlie',
    'attackIpm',
    'attackNoiseAmp',
)


def hasByzantineAttack(config):
    return any(bool(config.get(flag)) for flag in ATTACK_FLAGS)


def isCleanEnvironment(config):
    """True only for the clean reference: no Byzantine nodes and no channel noise."""
    return (
        not hasByzantineAttack(config)
        and not bool(config.get('useChannelNoise', False))
        and config.get('noiseMitigation', 'none') == 'none'
    )


def isCleanReferenceConfig(config):
    """True for the single clean threshold config used as the upper reference."""
    return (
        isCleanEnvironment(config)
        and config.get('experimentName', '').strip()
        == '0 - Byzantine Nodes + No Channel Noise + No Mitigation'
    )


def _currentConfigPath(config):
    expName = config.get('experimentName', '').strip()
    if not expName:
        return None
    split = splitFromConfig(config)
    approach = config.get('approach', 'basil')
    return os.path.join('gui', 'configs', split, approach, f'{expName}.json')


def _matchesCurrentConfig(config):
    """Return False when a saved result was produced from an obsolete config."""
    path = _currentConfigPath(config)
    if not path or not os.path.exists(path):
        return True
    try:
        with open(path, 'r') as f:
            current = json.load(f)
    except Exception:
        return True
    for key, val in current.items():
        if config.get(key) != val:
            return False
    return True


def buildLabel(config):
    # use custom experiment name if provided, otherwise auto-generate from config fields
    customName = config.get('experimentName', '').strip()
    if customName:
        return customName

    # Auto-generate descriptive label
    useNoise = config.get('useChannelNoise', False)
    mitigation = config.get('noiseMitigation', 'none')
    useBasil = config.get('useBasil', False)

    # Noise/environment description. Reserve "Clean Environment" for the one
    # true clean reference; Byzantine-only runs are not clean just because the
    # channel is noiseless.
    if isCleanEnvironment(config):
        noisePart = "Clean Environment"
    elif not useNoise:
        noisePart = "No Channel Noise"
    else:
        noiseStart = config.get('channelNoiseStart', 0)
        noisePart = f"Channel Noise@Round {noiseStart}" if noiseStart > 0 else "Channel Noise"

        # Mitigation
        if mitigation == 'ebm':
            noisePart += " + With EBM"
        else:
            noisePart += " + No EBM"

    # Attack description
    attacks = []
    if config.get('attackGaussian'):
        attacks.append(f"Gaussian@{config.get('attackGaussianStart', 0)}")
    if config.get('attackSignFlip'):
        attacks.append(f"SignFlip@{config.get('attackSignFlipStart', 0)}")
    if config.get('attackHidden'):
        attacks.append(f"Hidden@{config.get('attackHiddenStart', 0)}")
    if config.get('attackModelPoison'):
        attacks.append(f"ModelPoison@{config.get('attackModelPoisonStart', 0)}")
    if config.get('attackScaling'):
        attacks.append(f"Scaling@{config.get('attackScalingStart', 0)}")
    if config.get('attackAlie'):
        attacks.append(f"ALIE@{config.get('attackAlieStart', 0)}")
    if config.get('attackIpm'):
        attacks.append(f"IPM@{config.get('attackIpmStart', 0)}")
    if config.get('attackNoiseAmp'):
        attacks.append(f"NoiseAmp@{config.get('attackNoiseAmpStart', 0)}")

    if attacks:
        attackPart = " + ".join(attacks)
    else:
        attackPart = "No Byzantine Nodes"

    # Topology
    topo = "BASIL Ring" if useBasil else "Ring Topology"

    return f"{topo} ({noisePart} + {attackPart})"


def _approachTitle(approach):
    # human-readable approach label for plot titles
    titles = {
        'basil':   'BASIL Ring',
        'noisy':   'Noisy Channel (FedAvg)',
        'merged':  'Merged (BASIL + EBM)',
        'cart':    'CART (Class-Aware Ring)',
        '_legacy': 'Legacy',
    }
    return titles.get(approach, approach.title())


CLEAN_LW    = 2.0
CLEAN_ALPHA = 0.95


def loadCleanBaselines(dataset, approach, split=None):
    """Return only the true clean reference runs.

    The attack folder named "none" also contains channel-noise experiments.
    Those are not clean baselines and should not be labeled or plotted as the
    clean upper threshold.
    """
    return [
        exp for exp in discoverExperiments(dataset, 'none', approach, split=split)
        if isCleanReferenceConfig(exp['config']) and _matchesCurrentConfig(exp['config'])
    ]


def cleanOverlayStyle(idx):
    """Distinct styling for each 'none' baseline overlaid on attack plots."""
    return {
        'color': _CLEAN_COLORS[idx % len(_CLEAN_COLORS)],
        'linestyle': _CLEAN_LINESTYLES[idx % len(_CLEAN_LINESTYLES)],
        'marker': _ALL_MARKERS[idx % len(_ALL_MARKERS)],
    }


def datasetTitle(dataset):
    return {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }.get(dataset, dataset.upper())


def environmentLabel(config, attackKey):
    hasAttack = attackKey != 'none'
    hasNoise = bool(config.get('useChannelNoise', False))
    if hasAttack and hasNoise:
        return 'Byzantine + Channel Noise'
    if hasAttack:
        return 'Byzantine Only'
    if hasNoise:
        return 'Channel Noise Only'
    return 'Clean'


def noiseLevelLabel(config):
    if not config.get('useChannelNoise', False):
        return ''
    return f"σ={float(config.get('channelNoiseSigma', 0.0)):.1f}"


def methodLabel(config):
    ss = bool(config.get('useBasil', False))
    ebm = config.get('noiseMitigation', 'none') == 'ebm'
    if ss and ebm:
        return 'SS + EBM'
    if ss:
        return 'SS'
    if ebm:
        return 'EBM'
    return 'No Mitigation'


def _envOrderIndex(env):
    return ENV_ORDER.index(env) if env in ENV_ORDER else len(ENV_ORDER)


def _methodOrderIndex(method):
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def _noiseSortValue(config):
    if not config.get('useChannelNoise', False):
        return -1.0
    return float(config.get('channelNoiseSigma', 0.0))


def _noiseBucketSortValue(bucket):
    if bucket == "no_channel_noise":
        return -1.0
    try:
        return float(bucket.replace("sigma_", "").replace("_", "."))
    except ValueError:
        return 99.0


def _envNoiseDisplay(config, attackKey):
    env = environmentLabel(config, attackKey)
    noiseLabel = noiseLevelLabel(config)
    if noiseLabel:
        return f"{env} ({noiseLabel})"
    return env


def experimentSortKey(exp, attackKey):
    config = exp['config']
    env = environmentLabel(config, attackKey)
    method = methodLabel(config)
    return (
        _envOrderIndex(env),
        _noiseSortValue(config),
        _methodOrderIndex(method),
        exp['label'],
    )


def _finalAcc(exp):
    if not os.path.exists(exp['avgPath']):
        return None
    acc = np.load(exp['avgPath'])
    if len(acc) == 0:
        return None
    return float(acc[-1])


def plotDatasetExperiments(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    # overlay all experiment curves on one axes and save the figure
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "experiments_avg.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cleanExperiments = loadCleanBaselines(dataset, approach, split=split) if attackKey != 'none' else []
    colors = getColors(len(experiments) + len(cleanExperiments))
    markers = getMarkers(len(experiments))

    for idx, exp in enumerate(experiments):
        if not os.path.exists(exp['avgPath']):
            continue

        acc = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))

        ax.plot(rounds, acc,
                label=exp['label'],
                color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=5,
                markevery=max(1, len(rounds) // 10))

    # Overlay clean baselines as dashed reference lines. Use distinct colors
    # for each clean run so multiple "none" folder overlays are distinguishable.
    cleanMarkers = getMarkers(len(cleanExperiments))
    for cleanIdx, exp in enumerate(cleanExperiments):
        if not os.path.exists(exp['avgPath']):
            continue
        acc    = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))
        style = cleanOverlayStyle(cleanIdx)
        ax.plot(rounds, acc, label=f"[Clean Environment] {exp['label']}",
                color=style['color'], linewidth=CLEAN_LW,
                linestyle=style['linestyle'], alpha=CLEAN_ALPHA,
                marker=cleanMarkers[cleanIdx], markersize=4,
                markevery=max(1, len(rounds) // 10))

    title = datasetTitles.get(dataset, dataset.upper())
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'{title} | {attackTitle} | {approachTitle} — Average Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDatasetExperimentsZoom(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    """Zoomed line plot for comparing mitigation curves without the clean scale."""
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    if len(experiments) <= 1:
        return

    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "experiments_avg_zoom.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = getColors(len(experiments))
    markers = getMarkers(len(experiments))
    yVals = []

    for idx, exp in enumerate(experiments):
        if not os.path.exists(exp['avgPath']):
            continue
        acc = np.load(exp['avgPath'])
        if len(acc) == 0:
            continue
        rounds = np.arange(len(acc))
        yVals.extend(float(x) for x in acc)
        ax.plot(rounds, acc,
                label=exp['label'],
                color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=5,
                markevery=max(1, len(rounds) // 10))

    if not yVals:
        plt.close()
        return

    yMin = max(0.0, min(yVals) - 0.03)
    yMax = min(1.0, max(yVals) + 0.06)
    if yMax - yMin < 0.12:
        center = (yMax + yMin) / 2
        yMin = max(0.0, center - 0.06)
        yMax = min(1.0, center + 0.06)

    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'Zoomed Accuracy | {datasetTitle(dataset)} | {attackTitle} | {approachTitle}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([yMin, yMax])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDatasetGrid(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    # one subplot per experiment arranged in a grid layout
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    if len(experiments) <= 1:
        return

    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "grid_avg.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    nExps = len(experiments)
    nCols = min(nExps, 3)
    nRows = (nExps + nCols - 1) // nCols

    fig, axes = plt.subplots(nRows, nCols, figsize=(6 * nCols, 5 * nRows))
    if nRows == 1 and nCols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = datasetTitles.get(dataset, dataset.upper())
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    fig.suptitle(f'{title} | {attackTitle} | {approachTitle} — Average Accuracy',
                 fontsize=16, fontweight='bold')

    colors  = getColors(len(experiments))
    markers = getMarkers(len(experiments))

    cleanExperiments = loadCleanBaselines(dataset, approach, split=split) if attackKey != 'none' else []

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        if not os.path.exists(exp['avgPath']):
            continue

        acc    = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))

        ax.plot(rounds, acc, color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=4, markevery=1)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Average Accuracy', fontsize=10)
        ax.set_title(exp['label'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Overlay every clean run from the 'none' folder with unique styling.
        for cleanIdx, cleanExp in enumerate(cleanExperiments):
            if not os.path.exists(cleanExp['avgPath']):
                continue
            cleanAcc = np.load(cleanExp['avgPath'])
            if len(cleanAcc) == 0:
                continue
            cleanRounds = np.arange(len(cleanAcc))
            style = cleanOverlayStyle(cleanIdx)
            ax.plot(cleanRounds, cleanAcc,
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=CLEAN_LW, alpha=CLEAN_ALPHA,
                    marker=style['marker'], markersize=3,
                    markevery=max(1, len(cleanRounds) // 10),
                    label=f"[Clean Environment] {cleanExp['label']}")

        if cleanExperiments:
            ax.legend(fontsize=7, loc='lower right')

        # annotate the final accuracy value on the last point
        if len(acc) > 0:
            ax.annotate(f'{acc[-1]:.3f}', xy=(len(acc) - 1, acc[-1]),
                       fontsize=9, fontweight='bold',
                       xytext=(-30, 10), textcoords='offset points')

    # hide unused subplots
    for idx in range(nExps, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotImprovementOverNoMitigation(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    """Final-accuracy gain relative to matching no-mitigation environment."""
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "improvement_over_no_mitigation_avg.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    baselines = {}
    rows = []

    for exp in experiments:
        final = _finalAcc(exp)
        if final is None:
            continue
        config = exp['config']
        env = environmentLabel(config, attackKey)
        bucket = noiseBucket(config)
        method = methodLabel(config)
        if method == 'No Mitigation':
            baselines[(env, bucket)] = max(final, baselines.get((env, bucket), -1.0))
        rows.append((env, bucket, method, exp['label'], final, config))

    bestRows = {}
    for env, bucket, method, label, final, config in rows:
        if (env, bucket) not in baselines:
            continue
        baseline = baselines[(env, bucket)]
        displayEnv = _envNoiseDisplay(config, attackKey)
        displayLabel = f"{displayEnv}\n{method}\nbase {baseline:.3f}"
        rawGain = 0.0 if method == 'No Mitigation' else final - baseline
        gain = max(0.0, rawGain)
        key = (env, bucket, method)
        sortKey = (
            _envOrderIndex(env),
            _noiseSortValue(config),
            _methodOrderIndex(method),
        )
        if key not in bestRows or final > bestRows[key][5]:
            bestRows[key] = (sortKey, displayLabel, gain, rawGain, method, final)

    plotRows = sorted(bestRows.values(), key=lambda row: row[0])

    if not plotRows:
        return

    _, labels, gains, rawGains, methods, finals = zip(*plotRows)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 6))
    colors = [METHOD_COLORS.get(method, '#6b7280') for method in methods]
    bars = ax.bar(range(len(labels)), gains, color=colors, width=0.65)
    ax.axhline(0.0, color='#111827', linewidth=1.5)

    for bar, gain in zip(bars, gains):
        offset = 0.005
        annotation = f'{gain:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, gain + offset,
                annotation, ha='center', va='bottom', fontsize=9, fontweight='bold')

    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_ylabel('Final Accuracy Gain Over Matched Baseline', fontsize=12)
    ax.set_title(f'Mitigation Gain (negative gains shown as 0) | {datasetTitle(dataset)} | {attackTitle} | {approachTitle}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.text(0.01, 0.97,
            'Matched baseline: No Mitigation with the same attack/noise setting',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#d1d5db', alpha=0.85, boxstyle='round,pad=0.3'))
    yMax = max(gains) if gains else 0.0
    ax.set_ylim([0, max(0.05, yMax + 0.04)])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotAblationGroups(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    """Grouped final-accuracy bars by environment and mitigation method."""
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "ablation_groups_avg.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    values = {}
    groupSort = {}

    for exp in experiments:
        final = _finalAcc(exp)
        if final is None:
            continue
        config = exp['config']
        env = environmentLabel(config, attackKey)
        method = methodLabel(config)
        group = _envNoiseDisplay(config, attackKey)
        groupSort[group] = (_envOrderIndex(env), _noiseSortValue(config), group)
        key = (group, method)
        if key not in values or final > values[key][0]:
            values[key] = final

    groups = sorted(groupSort, key=lambda group: groupSort[group])
    methods = [m for m in METHOD_ORDER if any((group, m) in values for group in groups)]
    if not groups or not methods:
        return

    x = np.arange(len(groups))
    width = min(0.18, 0.8 / max(1, len(methods)))
    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 2.4), 6))
    colors = [METHOD_COLORS.get(method, '#6b7280') for method in methods]

    for idx, method in enumerate(methods):
        offsets = x + (idx - (len(methods) - 1) / 2) * width
        heights = [values.get((group, method), 0.0) for group in groups]
        bars = ax.bar(offsets, heights, width=width, label=method, color=colors[idx])
        for bar, height in zip(bars, heights):
            if height <= 0:
                continue
            annotation = f'{height:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
                    annotation, ha='center', va='bottom', fontsize=8, rotation=90)

    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_ylabel('Final Average Accuracy', fontsize=12)
    ax.set_title(f'Ablation Groups | {datasetTitle(dataset)} | {attackTitle} | {approachTitle}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10, rotation=15, ha='right')
    maxValue = max(values.values())
    ax.set_ylim([0, min(1.0, maxValue + 0.12)])
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotFinalAccuracyBar(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    # bar chart of each experiment's final accuracy value
    experiments = sorted(experiments, key=lambda exp: experimentSortKey(exp, attackKey))
    split = split or (experiments[0].get('split') if experiments else 'nonIID')
    savePath = _plotPath(split, dataset, attackKey, approach, "final_accuracy_avg.png", plotSubdir)
    if _shouldSkipPlot(savePath, skipExisting, _experimentSourcePaths(experiments)):
        return

    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    labels = []
    finalAccs = []
    noiseLabels = []

    for exp in experiments:
        if not os.path.exists(exp['avgPath']):
            continue
        acc = np.load(exp['avgPath'])
        labels.append(exp['label'])
        finalAccs.append(acc[-1] if len(acc) > 0 else 0)
        noiseLabels.append(noiseLevelLabel(exp['config']))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    colors = getColors(len(labels))
    bars   = ax.bar(range(len(labels)), finalAccs, color=colors, width=0.6)

    for bar, acc in zip(bars, finalAccs):
        annotation = f'{acc:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                annotation, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Overlay each clean run's final accuracy from the 'none' folder.
    if attackKey != 'none':
        cleanExperiments = loadCleanBaselines(dataset, approach, split=split)
        for cleanIdx, cleanExp in enumerate(cleanExperiments):
            if not os.path.exists(cleanExp['avgPath']):
                continue
            cleanAcc = np.load(cleanExp['avgPath'])
            if len(cleanAcc) == 0:
                continue
            style = cleanOverlayStyle(cleanIdx)
            finalCleanAcc = float(cleanAcc[-1])
            ax.axhline(y=finalCleanAcc,
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=CLEAN_LW, alpha=CLEAN_ALPHA,
                       label=f"[Clean Environment] {cleanExp['label']}: {finalCleanAcc:.3f}",
                       zorder=5)
        if cleanExperiments:
            ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    title = datasetTitles.get(dataset, dataset.upper())
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'Final Accuracy | {title} | {attackTitle} | {approachTitle}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotExperimentSet(dataset, attackKey, approach, experiments, split=None, plotSubdir=None, skipExisting=False):
    plotDatasetExperiments(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)
    plotDatasetExperimentsZoom(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)
    if len(experiments) > 1:
        plotDatasetGrid(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)
    plotFinalAccuracyBar(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)
    plotImprovementOverNoMitigation(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)
    plotAblationGroups(dataset, attackKey, approach, experiments, split=split, plotSubdir=plotSubdir, skipExisting=skipExisting)


def plotMitigationSweepComparison(dataset, attackKey, split=None, skipExisting=False):
    """Compare approaches/methods on one sigma sweep for the same attack."""
    split = split or 'nonIID'
    approaches = [
        approach for approach in discoverApproaches(dataset, attackKey, split=split)
        if approach not in ('_legacy',)
    ]
    preferred = [approach for approach in ('merged', 'cart') if approach in approaches]
    others = [approach for approach in approaches if approach not in preferred]
    approaches = preferred + others
    if len(approaches) < 2:
        return

    saveDir = os.path.join(_plotRootForComparison(approaches), "images", "gui", split, dataset, attackKey)
    savePath = os.path.join(saveDir, "mitigation_sweep_comparison_avg.png")

    values = {}
    sigmas = set()
    sourcePaths = []
    for approach in approaches:
        for exp in discoverExperiments(dataset, attackKey, approach, split=split):
            final = _finalAcc(exp)
            if final is None:
                continue
            sourcePaths.extend(_experimentSourcePaths([exp]))
            config = exp['config']
            sigma = _noiseSortValue(config)
            if sigma < 0:
                sigma = 0.0
            method = methodLabel(config)
            key = (approach, method, sigma)
            values[key] = max(final, values.get(key, -1.0))
            sigmas.add(sigma)

    if not values or not sigmas:
        return
    if _shouldSkipPlot(savePath, skipExisting, sourcePaths):
        return

    xVals = sorted(sigmas)
    approachStyles = {
        'cart': '-',
        'merged': '--',
        'basil': ':',
        'noisy': '-.',
    }
    methodMarkers = {
        'No Mitigation': 'o',
        'SS': 's',
        'EBM': '^',
        'SS + EBM': 'D',
    }

    fig, ax = plt.subplots(figsize=(max(10, len(xVals) * 1.9), 6))

    for approach in approaches:
        for method in METHOD_ORDER:
            yVals = [
                values.get((approach, method, sigma), np.nan)
                for sigma in xVals
            ]
            if all(np.isnan(y) for y in yVals):
                continue
            isTarget = approach == 'cart' and method == 'SS + EBM'
            label = f"{_approachTitle(approach)} | {method}"
            ax.plot(
                xVals, yVals,
                label=label,
                color=METHOD_COLORS.get(method, '#6b7280'),
                linestyle=approachStyles.get(approach, '-'),
                marker=methodMarkers.get(method, 'o'),
                linewidth=3.4 if isTarget else 2.0,
                markersize=7 if isTarget else 5,
                alpha=1.0 if isTarget else 0.78,
                zorder=5 if isTarget else 3,
            )

    for approach in approaches:
        cleanExperiments = loadCleanBaselines(dataset, approach, split=split)
        if not cleanExperiments:
            continue
        cleanFinals = [
            _finalAcc(exp) for exp in cleanExperiments
            if _finalAcc(exp) is not None
        ]
        if not cleanFinals:
            continue
        cleanFinal = max(cleanFinals)
        ax.axhline(
            cleanFinal,
            color='#111827',
            linestyle=approachStyles.get(approach, ':'),
            linewidth=1.4,
            alpha=0.45,
            label=f"Clean reference | {_approachTitle(approach)} {cleanFinal:.3f}",
        )

    targetPoints = [
        (sigma, values[('cart', 'SS + EBM', sigma)])
        for sigma in xVals
        if ('cart', 'SS + EBM', sigma) in values
    ]
    if targetPoints:
        bestSigma, bestAcc = max(targetPoints, key=lambda item: item[1])
        ax.annotate(
            f"CART SS+EBM peak: σ={bestSigma:.1f}, {bestAcc:.3f}",
            xy=(bestSigma, bestAcc),
            xytext=(10, 14),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#d1d5db', alpha=0.9, boxstyle='round,pad=0.3'),
            arrowprops=dict(arrowstyle='->', color='#111827', linewidth=1.0),
        )

    attackTitle = attackKey.replace("_", " + ").title()
    ax.set_xlabel('Channel Noise σ (0.0 = no channel noise)', fontsize=12)
    ax.set_ylabel('Final Average Accuracy', fontsize=12)
    ax.set_title(f'Approach Sweep Comparison | {datasetTitle(dataset)} | {attackTitle}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(xVals)
    ax.set_xticklabels([f"{sigma:.1f}" for sigma in xVals])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout()
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateGuiPlots():
    # discover datasets → attack types → approaches, then generate plots per combination
    print("\n" + "=" * 80)
    print("GENERATING GUI EXPERIMENT PLOTS")
    print("=" * 80)

    splits = discoverDataSplits()
    if not splits:
        print("No GUI experiment results found in experiments/results/gui/")
        print("Run experiments via the GUI first (python runGui.py)")
        return

    print(f"Data splits found: {splits}")
    print("=" * 80)

    for split in splits:
        datasets = DATASETS_TO_PLOT if DATASETS_TO_PLOT else discoverDatasets(split=split)
        if not datasets:
            print(f"\nNo datasets found for split {split}, skipping...")
            continue

        print(f"\nSplit: {split} - Datasets: {datasets}")

        for dataset in datasets:
            attackTypes = discoverAttackTypes(dataset, split=split)
            if not attackTypes:
                print(f"\nNo attack type subfolders found for {split}/{dataset}, skipping...")
                continue

            print(f"\nDataset: {dataset.upper()} [{split}] - Attack types: {attackTypes}")

            for attackKey in attackTypes:
                approaches = discoverApproaches(dataset, attackKey, split=split)
                if not approaches:
                    continue

                print(f"\n  Attack: {attackKey} - Approaches: {approaches}")

                for approach in approaches:
                    experiments = discoverExperiments(dataset, attackKey, approach, split=split)
                    if not experiments:
                        continue

                    print(f"\n    Approach: {approach} ({len(experiments)} experiment(s))")
                    for exp in experiments:
                        print(f"      - [{exp['split']}] {exp['label']}")

                    plotExperimentSet(dataset, attackKey, approach, experiments, split=split)
                    for bucket, bucketExperiments in groupExperimentsByNoiseBucket(experiments).items():
                        print(f"      Bucket: {bucket} ({len(bucketExperiments)} experiment(s))")
                        plotExperimentSet(dataset, attackKey, approach, bucketExperiments,
                                          split=split, plotSubdir=bucket)

                plotMitigationSweepComparison(dataset, attackKey, split=split)

    print("\nAll GUI plots generated!")
    print("Plots saved to: plots/images/gui/ and plots2/images/gui/ for merged/cart")


if __name__ == "__main__":
    print("""
=========================================================================
                      GUI EXPERIMENT PLOTTING SCRIPT

   Auto-discovers and plots all experiments from experiments/results/gui

   Organized by: dataset / attack type / approach
=========================================================================
    """)

    generateGuiPlots()
    print("\nDone!")

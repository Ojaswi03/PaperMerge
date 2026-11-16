"""TRAIN-only input loading and audited partitions for 10 or 100 classes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tarfile
import urllib.request

from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')

ROOT = Path(__file__).resolve().parents[1]


def digest(array):
    a=np.asarray(array)
    h=hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode())
    for start in range(0,len(a),1024):
        h.update(np.ascontiguousarray(a[start:start+1024]).tobytes())
    return h.hexdigest()


def prepare_cache(dataset):
    folder=ROOT/'experiments/cache'/dataset
    paths=[folder/'x_train_float32.npy',folder/'y_train_int32.npy']
    if all(p.is_file() for p in paths):
        return
    folder.mkdir(parents=True,exist_ok=True)
    if dataset=='mnist':
        archive=Path.home()/'.keras/datasets/mnist.npz'
        if not archive.is_file():
            archive=folder/'mnist.npz'
            urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',archive)
        with np.load(archive,allow_pickle=False) as source:
            # The archive also contains TEST. Only these two TRAIN keys are read.
            x=source['x_train'].astype(np.float32)[...,None]/255.
            y=source['y_train'].astype(np.int32)
    elif dataset in ('cifar10','cifar100'):
        if dataset=='cifar10':
            raw=Path.home()/'.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py'
            batches=[]
            if not raw.is_dir():
                raise FileNotFoundError('Prepare the existing CIFAR-10 TRAIN cache before SHARP.')
            for i in range(1,6):
                with (raw/f'data_batch_{i}').open('rb') as handle:
                    batches.append(pickle.load(handle,encoding='bytes'))
            images=np.concatenate([b[b'data'] for b in batches]);labels=sum([b[b'labels'] for b in batches],[])
        else:
            archive=folder/'cifar-100-python.tar.gz'
            if not archive.is_file():
                temporary=archive.with_suffix('.download')
                urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',temporary)
                temporary.replace(archive)
            # Official CIFAR-100 archive checksum (also specified by Keras).
            if hashlib.md5(archive.read_bytes()).hexdigest()!='eb9058c3a382ffc7106e4002c42a8d85':
                raise ValueError('CIFAR-100 source archive checksum mismatch.')
            with tarfile.open(archive,'r:gz') as handle:
                source=handle.extractfile('cifar-100-python/train')
                batch=pickle.load(source,encoding='bytes')
            images=batch[b'data'];labels=batch[b'fine_labels']
        x=images.reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.float32)/255.
        y=np.asarray(labels,np.int32)
    else:
        raise ValueError('Unknown dataset.')
    for path,array in zip(paths,(x,y)):
        temporary=path.with_suffix('.tmp.npy');np.save(temporary,array,allow_pickle=False);temporary.replace(path)
    (folder/'sharp_train_cache.json').write_text(json.dumps(dict(dataset=dataset,
        trainingOnly=True,officialTestLoaded=False,samples=len(y),imagesSha256=digest(x),labelsSha256=digest(y)),indent=2)+'\n')


def training_data(dataset):
    folder=ROOT/'experiments/cache'/dataset
    x=np.load(folder/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(folder/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    classes=100 if dataset=='cifar100' else 10
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    expected=60000 if dataset=='mnist' else 50000
    if x.shape!=(expected,*shape) or y.shape!=(expected,) or x.dtype!=np.float32 or y.dtype!=np.int32:
        raise ValueError('Unexpected original TRAIN cache shape or dtype.')
    if not np.array_equal(np.unique(y),np.arange(classes)) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError('Expected normalized finite TRAIN images and all class labels.')
    return x,y


def prepare_partition(labels,cfg):
    from basil_core.research_protocol import keyed_rng
    y=np.asarray(labels);classes=cfg['classes'];seed=cfg['seed'];n=cfg['nodes']
    heldout=[];eligible=[]
    for c in range(classes):
        # One shared development holdout for all seeds/conditions of a dataset.
        members=np.flatnonzero(y==c)
        order=keyed_rng(2025,'sharp_train_holdout',cfg['dataset'],c).permutation(members)
        amount=1000//classes
        validation=order[:amount]
        if cfg['validationPerClass'] is not None:validation=validation[:cfg['validationPerClass']]
        heldout.extend(validation.tolist())
        candidates=order[amount:]
        if cfg['trainPerClass'] is not None:
            if cfg['trainPerClass']>len(candidates):raise ValueError('Requested subset exceeds available TRAIN examples.')
            candidates=keyed_rng(seed,'sharp_train_subset',c).choice(candidates,cfg['trainPerClass'],replace=False)
        eligible.extend(candidates.tolist())
    eligible=np.asarray(eligible,np.int64)
    if len(eligible)<2*n:raise ValueError('Need at least two fitting samples per node; enlarge the diagnostic subset.')
    attempts=0
    if cfg['partition']=='iid':
        chunks=[[] for _ in range(n)]
        for c in range(classes):
            members=keyed_rng(seed,'sharp_iid',c).permutation(eligible[y[eligible]==c])
            # Rotate remainder assignment so small subsets do not starve high IDs.
            offset=int(keyed_rng(seed,'sharp_iid_offset',c).integers(n))
            for j,piece in enumerate(np.array_split(members,n)):chunks[(j+offset)%n].extend(piece.tolist())
    else:
        from basil_core.data.cifar import _dirichletPartition
        for attempts in range(100):
            allocation=_dirichletPartition(y[eligible],n,cfg['dirichletAlpha'],keyed_rng(seed,'sharp_dirichlet',attempts))
            chunks=[eligible[a.astype(np.int64)].tolist() for a in allocation]
            if min(map(len,chunks))>=2:break
        else:raise ValueError('Dirichlet split has empty/tiny clients after 100 deterministic attempts.')
    chunks=[np.asarray(c,np.int64) for c in chunks]
    if min(map(len,chunks))<2:raise ValueError('IID diagnostic subset is too small for this node count.')
    flat=np.concatenate(chunks);heldout=np.asarray(heldout,np.int64)
    if len(np.unique(flat))!=len(flat) or set(flat)!=set(eligible) or np.intersect1d(flat,heldout).size:
        raise ValueError('TRAIN partition is not disjoint and exhaustive, or overlaps evaluation.')
    hist=np.asarray([np.bincount(y[c],minlength=classes) for c in chunks])
    identity=dict(partitionHash=hashlib.sha256(b''.join(c.astype('<i8').tobytes() for c in chunks)).hexdigest(),
        counts=[len(c) for c in chunks],classCounts=hist.tolist(),holdoutHash=digest(heldout),
        eligibleHash=digest(eligible),holdoutSamples=len(heldout),fitSamples=len(flat),
        dirichletResampleAttempt=attempts,minimumLocalSamples=int(hist.sum(axis=1).min()),
        classesWithAtLeastTwoSamples=(hist>=2).sum(axis=1).tolist(),officialTestLoaded=False)
    return chunks,heldout,identity

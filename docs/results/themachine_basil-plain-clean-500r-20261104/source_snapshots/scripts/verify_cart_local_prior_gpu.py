"""CPU/GPU numerical check of the training-only prior-adjusted CART objective."""
def verify(cfg):
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_local_prior import LocalPriorWorker,log_prior,objective_gradients
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.research_protocol import keyed_rng
    old=SharpWorker(cfg,'CPU');new=LocalPriorWorker(cfg,'GPU');flat=old.export()
    np.testing.assert_array_equal(flat,new.export())
    x,y=training_data(cfg['dataset']);chunks,_,_=prepare_partition(y,cfg)
    indices=chunks[0];images=x[indices];labels=y[indices]
    first=keyed_rng(cfg['seed'],'sharp_batch',0,0,0).choice(len(labels),min(cfg['batchSize'],len(labels)),replace=False)
    prior=log_prior(labels,cfg['classes']);checks=[]
    for route,coefficient in [('fresh',.001),('own',0.)]:
        old.load(flat);new.admitted_route=route;new.admitted_has_channel=route!='own'
        anchor=tuple(tf.convert_to_tensor(v) for v in old.tensors(flat));eta=cfg['learningRate']
        records=[]
        for step in range(cfg['localSteps']):
            b=first if not step else keyed_rng(cfg['seed'],'sharp_batch',0,0,step).choice(
                len(labels),min(cfg['batchSize'],len(labels)),replace=False)
            with tf.device('/CPU:0'):
                gradients,values=objective_gradients(old.model,images[b],labels[b],anchor,
                    tf.constant(coefficient,tf.float32),tf.constant(cfg['proximalMu'],tf.float32),tf.constant(prior))
                for v,g in zip(old.model.trainable_variables,gradients):v.assign_sub(tf.constant(eta,tf.float32)*g)
            records.append([float(v) for v in values])
        result,row=new.train(flat,images,labels,0,0,.001,first)
        error=float(np.max(np.abs(result-old.export())))
        np.testing.assert_allclose(result,old.export(),rtol=2e-5,atol=2e-6)
        actual=[[r[k] for k in ('ce','gradientSquared','objective','proximalPenalty','updateGradientNorm')] for r in row['steps']]
        np.testing.assert_allclose(actual,records,rtol=5e-5,atol=2e-5)
        np.testing.assert_array_equal(row['trainingLogPrior'],prior.astype(np.float64))
        if row['optimizerSteps']!=cfg['localSteps']:raise AssertionError('Prior worker changed SGD count.')
        checks.append(dict(dataset=cfg['dataset'],route=route,priorObjectiveCpuGpuVerified=True,
            maximumWeightDifference=error,optimizerSteps=cfg['localSteps']))
    return checks

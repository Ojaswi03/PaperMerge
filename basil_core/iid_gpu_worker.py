"""GPU kernels with CPU-keyed preprocessing; shares the existing training loop."""
import tensorflow as tf
import numpy as np
from basil_core.iid_study import IidWorker
from basil_core.research_protocol import gradient_norm_objective_gradients
from basil_core.research_audit import augment_batch, ProtocolAudit, tensor_stats
from basil_core.trainer import lossFn


class GpuIidWorker(IidWorker):
    def _batch_gradients(self,x,y,key,mode,semantics,sigma,legacy_lambda):
        if semantics!='paper_absolute_gaussian' or mode not in {'none','gradient_norm_objective'}:
            raise ValueError('GPU IID execution supports CE and source EBM only.')
        # Per-image map/seed/crop operations are expensive tiny CUDA launches.
        # The original keyed transform stays byte-identical on the CPU.
        with tf.device('/CPU:0'):
            images=augment_batch(x,key)
        with tf.device('/GPU:0'):
            for variable in self.model.trainable_variables:
                tf.debugging.assert_all_finite(variable,'Non-finite incoming parameter')
            if mode=='gradient_norm_objective':
                gradients,values=gradient_norm_objective_gradients(self.model,images,y,sigma**2,details=True)
            else:
                with tf.GradientTape() as tape:
                    logits=self.model(images,training=True)
                    tf.debugging.assert_all_finite(logits,'Non-finite training logits')
                    ce=lossFn(y,logits)
                gradients=tape.gradient(ce,self.model.trainable_variables)
                if any(g is None for g in gradients):raise ValueError('Disconnected CE gradient.')
                norm=tf.linalg.global_norm(gradients)
                accuracy=tf.reduce_mean(tf.cast(tf.argmax(logits,axis=1,output_type=tf.int32)==y,tf.float32))
                values=(ce,norm**2,ce,tf.constant(0.),norm,norm,tf.constant(0.,tf.float64),accuracy)
            for name,value in zip(('CE','gradient norm squared','objective','coefficient',
                'base norm','applied norm','EBM correction norm','accuracy'),values):
                tf.debugging.assert_all_finite(value,f'Non-finite {name}')
            for gradient in gradients:tf.debugging.assert_all_finite(gradient,'Non-finite applied gradient')
            return gradients,values


class GpuProtocolAudit(ProtocolAudit):
    def begin_batch(self,*args,**kwargs):
        with tf.device('/CPU:0'):
            super().begin_batch(*args,**kwargs)

    def failure(self, worker, error):
        if not isinstance(error, tf.errors.ResourceExhaustedError):
            return super().failure(worker, error)
        # An OOM must not launch the base audit's GPU-intensive graph replay.
        pending=self.pending
        self.write('failure.json',{**self.current,'exception':str(error),
            'category':'resource_exhausted','graphReplaySkipped':True,
            'parameterStatsBeforeBatch':[tensor_stats(p) for p in pending['params']] if pending else [],
            'recovery':'Restart from the last completed activation checkpoint after freeing memory.'})
        if pending:
            np.savez_compressed(self.directory/'failure_batch.npz',x=pending['x'],y=pending['y'],key=pending['key'],
                **{f'weight_{i}':p for i,p in enumerate(pending['params'])})

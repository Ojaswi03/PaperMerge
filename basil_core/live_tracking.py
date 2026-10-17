"""Optional observation-only W&B logging. No model/RNG/optimizer dependencies."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT/'gui/worker_state/live_tracking.json'
LOG = logging.getLogger(__name__)
CONFIG_FIELDS = ('experimentProtocol','conditionId','nRounds','nNodes','dataset','partitionStrategy',
    'seed','modelArchitecture','localEpochs','batchSize','optimizer','learningRate',
    'learningRateSchedule','momentum','basilMemorySize','snapshotSelectionRule','aggregationMode',
    'resolvedAttackerIds','attackType','attackStartRound','attackHidden','useChannelNoise',
    'channelNoiseSemantics','channelNoiseSigmaAbsolute','localObjective','ebmMode','researchValid')


def read_settings(path=SETTINGS_PATH):
    if os.environ.get('PAPERMERGE_LIVE_TRACKING')=='disabled':
        return {'enabled': False}
    if not Path(path).exists():
        return {'enabled': False}
    return json.loads(Path(path).read_text())


class LiveTracker:
    """Only receives existing scalar telemetry; never uploads code/data/checkpoints."""
    def __init__(self, config, manifest, *, settings=None, sdk=None, notify=None, prepare_only=False):
        self.run = None
        self.notify = notify or (lambda payload: None)
        self.metadata = {'status': 'disabled'}
        self.last_round = -1
        self.last_activation = -1
        self.namespace = config.get('trackingNamespace', 'IID RUN')
        try:
            settings = read_settings() if settings is None else settings
            if not settings.get('enabled', False):
                return
            if settings.get('mode', 'online') not in ('online','offline'):
                raise ValueError('Tracking mode must be online or offline.')
            entity, project = settings.get('entity',''), settings.get('project','')
            if not entity or not project:
                raise ValueError('Live tracking needs explicit W&B entity and project names.')
            if sdk is None:
                import wandb as sdk
            previous = manifest.get('liveTracking', {}) or settings.get('preparedRuns',{}).get(config.get('conditionId'),{})
            resume = bool(previous.get('runId'))
            if resume and (previous.get('entity'),previous.get('project')) != (entity,project):
                raise ValueError('Cannot silently move resumed tracking to another account/project.')
            # W&B's ID generator does not touch the scientific keyed RNG streams.
            run_id = previous.get('runId') or sdk.util.generate_id()
            group = previous.get('group') or settings.get('group') or config.get('iidCampaignId',config['experimentProtocol'])
            self.last_round = previous.get('lastLoggedRound',-1)
            self.last_activation = previous.get('lastLoggedActivation',-1)
            logged_config = {k: config[k] for k in CONFIG_FIELDS if k in config}
            logged_config.update(initialModelHash=manifest.get('initialModelHash'),
                iidPartitionHash=manifest.get('iidPartitionHash'),device=manifest.get('device'),
                ebmCoefficient=config.get('channelNoiseSigmaAbsolute',0.)**2 if config.get('ebmMode')=='gradient_norm_objective' or config.get('localObjective')=='source_ebm' else 0.,
                legacyLambdaUsed=False)
            if config.get('partitionStrategy') == 'dirichlet':
                logged_config.update(partitionHash=manifest.get('partitionHash'),
                    dirichletAlpha=config.get('dirichletAlpha'),snapshotAnchorMu=config.get('snapshotAnchorMu'),
                    logitCalibrationMethod=config.get('logitCalibrationMethod'),logitCalibrationTau=config.get('logitCalibrationTau'))
            self.run = sdk.init(entity=entity,project=project,id=run_id,
                name=f"{self.namespace} — {config.get('conditionId','IID')} — {config['localObjective']} — {config['nRounds']} rounds",group=group,
                job_type='noniid-basil-research' if config.get('partitionStrategy')=='dirichlet' else 'iid-basil-research',tags=[config.get('conditionId','IID'),f"{config['nRounds']}rounds"],
                config=logged_config,mode=settings.get('mode','online'),
                resume='allow' if resume and settings.get('mode','online')=='online' else None,
                dir=str(ROOT/'gui/worker_state'),
                settings=sdk.Settings(disable_git=True,save_code=False,console='off',
                    x_disable_stats=True,init_timeout=15))
            self.run.define_metric('round')
            self.run.define_metric('round/*',step_metric='round')
            self.run.define_metric(self.namespace+'/*',step_metric='round')
            self.run.define_metric('activation')
            self.run.define_metric('node/*',step_metric='activation')
            self.run.summary['research_training_started'] = not prepare_only
            self.run.summary['research_status'] = 'prepared' if prepare_only else 'running'
            self.run.summary['research_rounds_completed'] = self.last_round + 1
            self.metadata = dict(status='active',entity=entity,project=project,runId=run_id,
                url=self.run.url,mode=settings.get('mode','online'),group=group,
                lastLoggedRound=self.last_round,lastLoggedActivation=self.last_activation)
            self._announce()
        except Exception as error:
            self._disable(error)

    def _announce(self):
        # Status must not depend on a widget or an active GUI event loop.
        try:
            self.notify(dict(event='tracking_status', **self.metadata))
        except Exception:
            LOG.warning('Tracking status notification failed',exc_info=True)

    def _disable(self, error):
        self.metadata.update(status='unavailable',error=str(error))
        LOG.warning('W&B unavailable; local research evidence remains authoritative: %s', error)
        if self.run is not None:
            try:
                self.run.finish(exit_code=1)
            except Exception:
                pass
        self.run = None
        self._announce()

    def _log(self, metrics):
        if self.run is None:
            return False
        clean = {key:value for key,value in metrics.items()
            if isinstance(value,(float,int)) and math.isfinite(value)}
        try:
            self.run.log(clean)
            return True
        except Exception as error:
            self._disable(error)
            return False

    def activation(self, record):
        index = int(record['round'])*10+int(record['nodeId'])
        if index <= self.last_activation:
            return
        metrics = {'activation':index,'node/round':record['round'],'node/id':record['nodeId']}
        for source,target in (('globalTestAccuracy','accuracy'),('localTrainingLoss','local_ce'),
            ('modelL2Norm','model_norm'),('selectedCandidateLoss','selected_local_ce'),
            ('inputSourceNode','selected_sender'),('attackActive','attack_active')):
            if source in record:
                metrics['node/'+target] = record[source]
        if self._log(metrics):
            self.last_activation = index
            self.metadata['lastLoggedActivation'] = index

    def round(self, row):
        index = int(row['round'])
        if index <= self.last_round:
            return
        primary = {}
        if row.get('full_test_accuracy') is not None:primary[self.namespace+'/Accuracy']=row['full_test_accuracy']
        if row.get('full_test_loss') is not None:primary[self.namespace+'/Loss']=row['full_test_loss']
        if self._log({'round':index,**primary,**{'round/'+k:v for k,v in row.items() if k!='round'}}):
            self.last_round = index
            self.metadata['lastLoggedRound'] = index
            try:
                self.run.summary['research_rounds_completed'] = index + 1
            except Exception as error:
                self._disable(error)

    def finish(self, status):
        if self.run is not None:
            try:
                self.run.summary['research_status'] = status
                self.run.finish(exit_code=0 if status in {'completed','prepared'} else 1)
                self.metadata['status'] = status
            except Exception as error:
                self.metadata.update(status='unavailable',error=str(error))
                LOG.warning('W&B finish failed; local results remain intact: %s',error)
            finally:
                self.run = None

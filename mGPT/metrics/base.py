from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2t import M2TMetrics
from .m2m import PredMetrics


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, metrics_dict=None, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints
        data_name = datamodule.name

        # Get the list of metrics to initialize from config
        metrics_to_init = metrics_dict if metrics_dict else cfg.METRIC.TYPE

        # Only initialize metrics that are configured
        if 'TM2TMetrics' in metrics_to_init and data_name in ["humanml3d", "kit"]:
            self.TM2TMetrics = TM2TMetrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        if 'M2TMetrics' in metrics_to_init and data_name in ["humanml3d", "kit"]:
            self.M2TMetrics = M2TMetrics(
                cfg=cfg,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)

        if 'MMMetrics' in metrics_to_init and data_name in ["humanml3d", "kit"]:
            self.MMMetrics = MMMetrics(
                cfg=cfg,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        if 'MRMetrics' in metrics_to_init:
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        if 'PredMetrics' in metrics_to_init:
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )

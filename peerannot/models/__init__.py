from .aggregation.MV import MV
from .aggregation.NaiveSoft import NaiveSoft
from .aggregation.DS import DawidSkene
from .aggregation.DS_clust import DawidSkeneClust
from .aggregation.GLAD import GLAD
from .aggregation.WDS import WDS
from .aggregation.twothird import TwoThird
from .aggregation.plantnet import PlantNet

from .aggregation.Wawa import Wawa
from .aggregation.IWMV import IWMV

from .identification.WAUM_perworker import WAUM_perworker
from .identification.WAUM import WAUM
from .identification.AUM import AUM
from .identification.trace_confusion import Trace_confusion
from .identification.Spam_score import Spam_Score
from .identification.entropy import Entropy
from .identification.krippendorff_alpha import Krippendorff_Alpha
from .agg_deep.CoNAL import CoNAL
from .agg_deep.Crowdlayer import Crowdlayer

agg_strategies = {
    "MV": MV,
    "NaiveSoft": NaiveSoft,
    "DS": DawidSkene,
    "DSWC": DawidSkeneClust,
    "GLAD": GLAD,
    "WDS": WDS,
    "PlantNet": PlantNet,
    "TwoThird": TwoThird,
    "IWMV": IWMV,
    "Wawa": Wawa,
}

agg_deep_strategies = {
    "CoNAL": CoNAL,
    "CrowdLayer": Crowdlayer,
}

identification_strategies = {
    "AUM": AUM,
    "WAUM_perworker": WAUM_perworker,
    "WAUM": WAUM,
    "entropy": Entropy,
    "trace_confusion": Trace_confusion,
    "spam_score": Spam_Score,
    "krippendorffAlpha": Krippendorff_Alpha,
}

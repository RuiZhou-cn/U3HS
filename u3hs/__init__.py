# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_u3hs_config
from .dataset_mapper import PanopticDeeplabDatasetMapper
from .u3hs_model import (
    U3HS,
    INS_EMBED_BRANCHES_REGISTRY,
    build_ins_embed_branch,
)

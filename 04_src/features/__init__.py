"""
特徴量モジュール

各種Block（特徴量変換器）を提供。
"""

from features.base import BaseBlock, set_seed
from features.pipeline import FeaturePipeline, BlockInfo

# Encoding Blocks
from features.blocks.encoding import (
    LabelEncodingBlock,
    CountEncodingBlock,
    TargetEncodingBlock,
    OneHotEncodingBlock,
    TopNCategoryLEBlock,
    MultiKeyTEBlock,
)

# Dimension Reduction Blocks
from features.blocks.dimension_reduction import (
    DimensionReductionBlock,
    SVDBlock,
    PCABlock,
    UMAPBlock,
)

# Text Blocks
from features.blocks.text import TfidfBlock

# Multi-hot / One-hot + SVD Blocks
from features.blocks.multi_hot import (
    MultiHotSVDBlock,
    MultiColumnMultiHotSVDBlock,
    MultiColumnOneHotSVDBlock,
)

# Utility Blocks
from features.blocks.rename import RenameBlock

# Numeric/Temporal/Aggregation Blocks (将来追加予定)
# from features.blocks.numeric import ...
# from features.blocks.temporal import ...
# from features.blocks.aggregation import ...

__all__ = [
    # Base
    "BaseBlock",
    "set_seed",
    # Pipeline
    "FeaturePipeline",
    "BlockInfo",
    # Encoding
    "LabelEncodingBlock",
    "CountEncodingBlock",
    "TargetEncodingBlock",
    "OneHotEncodingBlock",
    "TopNCategoryLEBlock",
    # Dimension Reduction
    "DimensionReductionBlock",
    "SVDBlock",
    "PCABlock",
    "UMAPBlock",
    # Text
    "TfidfBlock",
    # Multi-hot
    "MultiHotSVDBlock",
    "MultiColumnMultiHotSVDBlock",
    # Utility
    "RenameBlock",
]

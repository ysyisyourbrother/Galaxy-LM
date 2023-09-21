import torch
import torch.nn as nn


class ParallelMLP(nn.Module):
    """Parallel Transformer class."""
    def __init__(self, config):
        super(ParallelMLP, self).__init__()


class ParallelAttention(nn.Module):
    """Parallel Transformer class."""
    def __init__(self, config):
        super(ParallelAttention, self).__init__()


class ParallelTransformerLayer(nn.Module):
    """Parallel Transformer Layer class.
    Arguments:
        config: global configuration
        attn_mask_type: 
    """
    def __init__(self, config,
                 attn_mask_type):
        super(ParallelTransformerLayer, self).__init__()


class ParallelTransformer(nn.Module):
    """Parallel Transformer class."""
    def __init__(self, config):
        super(ParallelTransformer, self).__init__()

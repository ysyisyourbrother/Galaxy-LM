import torch

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_PARALLEL_GROUP = None


def get_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_PARALLEL_GROUP


def get_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    # global _MPU_TENSOR_MODEL_PARALLEL_RANK
    # if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
    #     return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank()


def get_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    # global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    # if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
    #     return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size()

def initialize_model_parallel():
    """ initialize _TENSOR_PARALLEL_GROUP """
    pass
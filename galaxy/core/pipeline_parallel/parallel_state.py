import torch


def get_pipeline_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    # global _MPU_TENSOR_MODEL_PARALLEL_RANK
    # if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
    #     return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank()


def get_pipeline_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    # global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    # if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
    #     return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size()

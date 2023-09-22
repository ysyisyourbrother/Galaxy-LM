import torch
import torch.distributed as dist

from galaxy.core.model_parallel.parallel_state import (
    get_model_parallel_world_size,
    get_model_parallel_rank
)


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    dist.all_reduce(input_)

    return input_


def _split_along_seq_dim(input_, seq_scatter_list):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    # Bypass the function if we are using only 1 GPU.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input_
    rank = get_model_parallel_rank()

    # Split along sequence dimension.
    dim_size = input_.size()[1]
    # seq_scatter_list 长度要等于参与模型并行的设备总数
    assert (len(seq_scatter_list) == world_size) , "seq_scatter_list should be equal to world_size"

    local_dim_size = seq_scatter_list[rank]
    dim_offset = sum(seq_scatter_list[:rank])

    output = input_[:, dim_offset : dim_offset + local_dim_size, :].contiguous()

    return output


def _gather_along_seq_dim(input_, seq_scatter_list):
    """Gather tensors and concatinate along the seq dimension.
    Arguments:
        input_: [bs, seq_len, hidden_size]
        seq_scatter_list: sequence len on different device [20,12]
    """

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    
    # seq_scatter_list 长度要等于参与模型并行的设备总数
    assert (len(seq_scatter_list) == world_size) , "seq_scatter_list should be equal to world_size"

    # 由于 dist.all_gather 不支持不同大小的张量聚合，因此我们要采取先填充，再聚合，后裁剪的策略
    # Step 1: Preallocate space for all_gather
    max_seq_len = max(seq_scatter_list)
    dim_size[1] = max_seq_len
    gathered_tensors = []
    for _ in range(world_size):
        tmp_tensor = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        gathered_tensors.append(tmp_tensor)
    
    # Step 2: If necessary, pad tensor to match max size
    if input_.size(1) < max_seq_len:
        dim_size[1] = max_seq_len - input_.size(1)
        pad_input = torch.cat([input_, torch.zeros(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())], dim=1)
    else:
        pad_input = input_

    # Step 3: Perform all_gather
    dist.all_gather(gathered_tensors, pad_input.contiguous())

    # Step 4: Remove padding and concatenate
    gathered_tensors = [t[:, :seq_len, :] for t, seq_len in zip(gathered_tensors, seq_scatter_list)]
    concatenated_tensor = torch.cat(gathered_tensors, dim=1)

    return concatenated_tensor


def _reduce_scatter_along_seq_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[1] % world_size == 0
    ), "Sequence dimension of the tensor should be divisible by sequence parallel size"

    dim_size[1] = dim_size[1] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    dist.reduce_scatter(output, [input_.contiguous()])
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, seq_scatter_list):
        ctx.seq_scatter_list = seq_scatter_list
        return _split_along_seq_dim(input_, ctx.seq_scatter_list)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_seq_dim(grad_output, ctx.seq_scatter_list), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""
    @staticmethod
    def forward(ctx, input_, is_tensor_parallel_follow, seq_scatter_list):
        """
        Arguments:
            - is_tensor_parallel_follow: all-gather完成后紧接着执行张量并行
            用来判断反向传播的梯度处理
        """
        ctx.is_tensor_parallel_follow = is_tensor_parallel_follow
        ctx.seq_scatter_list = seq_scatter_list
        return _gather_along_seq_dim(input_, seq_scatter_list)

    @staticmethod
    def backward(ctx, grad_output):
        # 如果all-gather后面紧跟着张量并行，则需要先做一个grad reduce再scatter。
        # 如果不是，则直接切分回去即可
        if ctx.is_tensor_parallel_follow:
            # 因为forward方法有多个输入，因此需要返回多个值的梯度，不需要梯度的返回None
            raise NotImplementedError('''_reduce_scatter_along_seq_dim for heterogeneous seq_size 
                                        hasn't been implemented yet''')
            return _reduce_scatter_along_seq_dim(grad_output), None, None
        else:
            # TODO: 支持不规则划分
            return _split_along_seq_dim(grad_output, ctx.seq_scatter_list), None, None


"""
Function.apply将Function作用到输入上，Pytorch会根据自定义的Forward、Backward功能修改计算图。
"""
def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_, seq_scatter_list):
    """ 将 sequence 划分到不同设备上 """
    return _ScatterToSequenceParallelRegion.apply(input_, seq_scatter_list)


def gather_from_sequence_parallel_region(input_, is_tensor_parallel_follow, seq_scatter_list):
    return _GatherFromSequenceParallelRegion.apply(input_, is_tensor_parallel_follow, seq_scatter_list)

"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""


import os
from pydoc import Helper
import threading
import torch
import torch.distributed as dist
import sys

from galaxy.core.pipeline_parallel import threadsafe_counter,threadsafe_queue
from galaxy.core.pipeline_parallel import parallel_state


class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.rank = parallel_state.get_pipeline_parallel_rank()
        self.world_size = parallel_state.get_pipeline_parallel_world_size()
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.tensor_tag = {"forward": 0, "backward": 1}
        self.tensor_shape = {"forward": (config.batch_size, config.pad_size, config.hidden_size), 
                             "backward": (config.batch_size, config.pad_size, config.hidden_size)}  
        self.setup_queue()
        # TODO 修改num_iterations值大小
        self.start_helper_threads(num_iterations=10000)

    def setup_queue(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        # 目前只考虑stage间只有单个input/output
        self.forward_receive_queues = threadsafe_queue.Queue()
        self.backward_receive_queues = threadsafe_queue.Queue()
        self.forward_send_queues = threadsafe_queue.Queue()
        self.backward_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self, num_iterations):
        # 启动 send forward helper thread 
        self.start_helper_thread(func=send_helper_thread, 
                                args=(self.forward_send_queues, 
                                self.next_rank,
                                num_iterations,
                                self.tensor_tag["forward"]))
        # 启动 send backward helper thread 
        self.start_helper_thread(func=send_helper_thread, 
                                 args=(self.backward_send_queues, 
                                       self.pre_rank,
                                       num_iterations,
                                       self.tensor_tag["backward"]))
        # 启动 recv forward helper thread 
        self.start_helper_thread(func=recv_helper_thread, 
                                 args=(self.forward_receive_queues, 
                                       self.tensor_shape["forward"], 
                                       self.pre_rank, 
                                       num_iterations,
                                       self.tensor_tag["forward"]))
        # 启动 recv backward helper thread 
        self.start_helper_thread(func=recv_helper_thread, 
                                 args=(self.backward_receive_queues,
                                       self.tensor_shape["backward"], 
                                       self.next_rank, 
                                       num_iterations,
                                       self.tensor_tag["backward"]))


    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()


    def send(self, tensor, backward=False):
        if backward:
            self.backward_send_queues.add(tensor)
        else:
            self.forward_send_queues.add(tensor)

    
    def recv(self, backward=False):
        if backward:
            tensor = self.backward_receive_queues.remove()
        else:
            tensor = self.forward_receive_queues.remove()
            tensor = tensor.requires_grad_()
        return tensor


def recv_helper_thread(recv_queue, tensor_shape, src_rank, num_iterations, tag):
    """负责接收张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    for i in range(num_iterations):
        tensor = _recv(tensor_shape, src_rank, tag)
        print(f"recv tensor from {src_rank}")
        recv_queue.add(tensor)


def send_helper_thread(send_queue, dst_rank, num_iterations, tag):
    """负责发送张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    for i in range(num_iterations):
        # 当send_queue为空时，队列阻塞
        tensor = send_queue.remove()
        print(f"send tensor to {dst_rank}")
        _send(tensor, dst_rank, tag)


def _send(tensor, dst_rank, tag):
    tensor = tensor.cpu()
    dist.send(tensor=tensor, dst=dst_rank, tag=tag)


def _recv(tensor_shape, src_rank, tag):
    tensor = torch.zeros(tensor_shape)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor




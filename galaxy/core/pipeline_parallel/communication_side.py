"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""

import threading
import torch
import torch.distributed as dist
from galaxy.core.pipeline_parallel import threadsafe_queue
from .communication import recv_helper_thread,send_helper_thread
class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank =  config.is_first_stage
        self.if_last_rank =  config.is_last_stage
        self.tensor_tag = {"forward": 0,  "forward_side": 1,"backward": 2}
        side_hidden_size =  int (config.hidden_size // config.side_reduction_factor)
        self.tensor_shape = {"forward": (config.batch_size, config.pad_size, config.hidden_size), 
                            "forward_side": (config.batch_size, config.pad_size,  side_hidden_size),
                            "backward": (config.batch_size, config.pad_size, side_hidden_size)}   
        self.setup_queue()
        # TODO: 修改num_iterations值大小
        self.start_helper_threads(num_iterations=1000)

    def setup_queue(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        self.forward_receive_queues = threadsafe_queue.Queue()
        self.backward_receive_queues = threadsafe_queue.Queue()
        self.forward_send_queues = threadsafe_queue.Queue()
        self.backward_send_queues = threadsafe_queue.Queue()
        # side 的 queue
        self.forward_side_receive_queues = threadsafe_queue.Queue()
        self.forward_side_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self, num_iterations):
        if not self.if_first_rank:
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
            # 启动 recv forward side helper thread 
            self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.forward_side_receive_queues, 
                                        self.tensor_shape["forward_side"], 
                                        self.pre_rank, 
                                        num_iterations,
                                        self.tensor_tag["forward_side"]))
        if not self.if_last_rank:
            # 启动 send forward helper thread
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["forward"]))
            # 启动 send forward side helper thread
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_side_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["forward_side"]))
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


    def send(self, tensor, backward=False, side=False):
        if backward:
            self.backward_send_queues.add(tensor)
        elif side:
            self.forward_side_send_queues.add(tensor)
        else:
            self.forward_send_queues.add(tensor)

    
    def recv(self, backward=False, side = False):
        if backward:
            tensor = self.backward_receive_queues.remove()
        elif side:
            tensor = self.forward_side_receive_queues.remove()
            tensor = tensor.requires_grad_()
        else:
            tensor = self.forward_receive_queues.remove()
            # tensor = tensor.requires_grad_() #TODO: 差别这么大
        return tensor



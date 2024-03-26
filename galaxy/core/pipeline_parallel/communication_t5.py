"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""

import threading
import torch
import torch.distributed as dist

from galaxy.core.pipeline_parallel import threadsafe_counter,threadsafe_queue
from galaxy.core.pipeline_parallel import parallel_state

class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.config = config
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank =  config.is_first_stage
        self.if_last_rank =  config.is_last_stage
        self.is_encoder = config.is_encoder
        self.is_encoder_first =  config.is_encoder_first
        self.is_encoder_last =config.is_encoder_last    
        self.is_decoder = config.is_decoder
        self.is_decoder_first = config.is_decoder_first
        self.is_decoder_last = config.is_decoder_last
        
        self.tensor_tag = {"forward": 0, "backward": 1, "encoder_output": 2}
         # encoder最后一层 decoder要不断传输encoder_outputs
        self.tensor_shape = {"forward": (config.batch_size, config.pad_size, config.d_model), 
                             "encoder_output": (config.batch_size, config.pad_size, config.d_model),
                             "backward": (config.batch_size, config.pad_size, config.d_model)}  
        self.setup_queue()
        # TODO 修改num_iterations值大小
        self.start_helper_threads(num_iterations=1000)

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
        
        # 由于decoder 额外需要 encoder outputs ，所以需要额外的queue
        self.forward_encoder_output_receive_queues = threadsafe_queue.Queue()
        self.forward_encoder_output_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self, num_iterations):
        if self.config.is_encoder  : # encoder only
            # 不是第一层encoder, 需要 接受encoder forward的结果
            if not self.config.is_encoder_first  :
                self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.forward_receive_queues, 
                                        self.tensor_shape["forward"], 
                                        self.pre_rank, 
                                        num_iterations,
                                        self.tensor_tag["forward"]))
            # 不是最后一个encoder 需要发送encoder forward的结果
            if not  self.config.is_encoder_last:
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["forward"]))
            # 如果是包含最后一层encoder， 需要传输encoder_outputs的queue
            if self.config.is_encoder_last  :
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_encoder_output_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["encoder_output"]))
            
                
        elif self.config.is_decoder : # decoder only
            # 都需要接受前面的encoder_outputs
            self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.forward_encoder_output_receive_queues, 
                                self.tensor_shape["encoder_output"], 
                                self.pre_rank, 
                                num_iterations,
                                self.tensor_tag["encoder_output"]))
            if not self.config.is_decoder_first :
                #  不包含第一层decoder,  接受decoder的输出
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.forward_receive_queues, 
                                self.tensor_shape["forward"], 
                                self.pre_rank, 
                                num_iterations,
                                self.tensor_tag["forward"]))
            if not self.config.is_decoder_last: 
                # 不包含最后一层decoder, 发送decoder的输出
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["forward"]))
                # 不包含最后一层decoder, 发送decoder的输出， 以及encoder_outputs
                self.start_helper_thread(func=send_helper_thread, 
                                        args=(self.forward_encoder_output_send_queues, 
                                        self.next_rank,
                                        num_iterations,
                                        self.tensor_tag["encoder_output"]))
            # backward  
            # 不是第一层 就要往前传backward
            if self.config.left_layer_index != 0:
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.backward_send_queues, 
                                        self.pre_rank,
                                        num_iterations,
                                        self.tensor_tag["backward"]))
            # 不是第一层 从后面接收
            if self.config.right_layer_index != self.config.num_layers*2:
                self.start_helper_thread(func=recv_helper_thread, 
                                        args=(self.backward_receive_queues,
                                            self.tensor_shape["backward"], 
                                            self.next_rank, 
                                            num_iterations,
                                        self.tensor_tag["backward"]))
                


    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()


    def send(self, tensor, backward=False, encoder_output =False):
        if backward:
            self.backward_send_queues.add(tensor)
        elif encoder_output :
            self.forward_encoder_output_send_queues.add(tensor)
        else:
            self.forward_send_queues.add(tensor)
            print("Stage {} forward send queue add tensor". format(self.config.stage))

    
    def recv(self, backward=False, encoder_output =False):
        if backward:
            tensor = self.backward_receive_queues.remove()
        elif encoder_output :
            tensor = self.forward_encoder_output_receive_queues.remove()
            tensor = tensor.requires_grad_() #TODO:
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
        print(f"recv tensor from {src_rank} ({tag}:{i}/{num_iterations})")
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
        print(f"send tensor to {dst_rank} ({tag}:{i}/{num_iterations})")
        _send(tensor, dst_rank, tag)

def _send(tensor, dst_rank, tag):
    tensor = tensor.cpu()
    dist.send(tensor=tensor, dst=dst_rank, tag=tag)


def _recv(tensor_shape, src_rank, tag):
    tensor = torch.zeros(tensor_shape)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor




"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""

import threading
import torch
import torch.distributed as dist

from galaxy.core.pipeline_parallel import threadsafe_counter,threadsafe_queue
from galaxy.core.pipeline_parallel import parallel_state
from .communication import recv_helper_thread,send_helper_thread

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

        self.tensor_tag = {"encoder_forward": 0, 
                           "decoder_forward":1,
                           "encoder_output_forward": 2,
                           "encoder_backward": 3,
                           "decoder_backward": 4,
                           "encoder_output_backward": 5
                            }
         # encoder最后一层 decoder要不断传输encoder_outputs
        # encoder 和 decoder 的forward_seq_len 不一样
        self.tensor_shape = {"encoder_forward": (config.batch_size, config.pad_size, config.d_model), 
                             "decoder_forward": (config.batch_size, 1, config.d_model),
                             "encoder_output_forward": (config.batch_size, config.pad_size, config.d_model),
                             
                             "encoder_backward": (config.batch_size, config.pad_size, config.d_model),
                             "decoder_backward": (config.batch_size, 1, config.d_model),
                             "encoder_output_backward": (config.batch_size,  config.pad_size, config.d_model)}  
        self.setup_queue()
        # TODO 修改num_iterations值大小
        self.start_helper_threads(num_iterations=1000)

    def setup_queue(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        #############################################################################
        ## forward
        if self.config.is_encoder:
            if not self.config.is_encoder_first:
                self.forward_receive_queues = threadsafe_queue.Queue()
            if not self.config.is_encoder_last:
                self.forward_send_queues = threadsafe_queue.Queue()
            if self.config.is_encoder_last:
                self.forward_encoder_output_send_queues = threadsafe_queue.Queue()
        if self.config.is_decoder:
            #所有decoder 都要接受encoder output
            self.forward_encoder_output_receive_queues = threadsafe_queue.Queue()
            if not self.config.is_encoder_last : #不是最后一层 都要向后传encoder output, 和 decoder forward
                self.forward_encoder_output_send_queues = threadsafe_queue.Queue()
                self.forward_send_queues = threadsafe_queue.Queue()
            if not self.config.is_decoder_first : #不是第一层 接受forward
                self.forward_receive_queues = threadsafe_queue.Queue()
        ##############################################################################
        ## backward
        if self.config.is_encoder:
            if self.config.is_encoder_last:
                self.backward_encoder_output_receive_queues = threadsafe_queue.Queue()
            if not self.config.is_encoder_last:
                self.backward_receive_queues = threadsafe_queue.Queue()
            if not self.config.is_encoder_first:
                self.backward_send_queues = threadsafe_queue.Queue()
                
        if self.config.is_decoder:
            self.backward_encoder_output_send_queues = threadsafe_queue.Queue()
            if not self.config.is_decoder_last:
                self.backward_encoder_output_receive_queues = threadsafe_queue.Queue()
                self.backward_receive_queues = threadsafe_queue.Queue()
            if not self.config.is_decoder_first:
                self.backward_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self, num_iterations):
        ########################################################
        # forward
        # encoder
        if self.config.is_encoder  : # encoder only
            # 不是第一层encoder, 需要 接受encoder forward的结果
            if not self.config.is_encoder_first  :
                self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.forward_receive_queues, 
                                        self.tensor_shape["encoder_forward"], 
                                        self.pre_rank, 
                                        num_iterations,
                                        self.tensor_tag["encoder_forward"]))
            # 不是最后一个encoder 需要发送encoder forward的结果
            if not  self.config.is_encoder_last:
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["encoder_forward"]))
            # 如果是包含最后一层encoder， 需要传输encoder_outputs的queue
            if self.config.is_encoder_last  :
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_encoder_output_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["encoder_output_forward"]))
            
        # decoder
        if self.config.is_decoder : # decoder only
            # 都需要接受前面的encoder_outputs
            self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.forward_encoder_output_receive_queues, 
                                self.tensor_shape["encoder_output_forward"], 
                                self.pre_rank, 
                                num_iterations,
                                self.tensor_tag["encoder_output_forward"]))
            if not self.config.is_decoder_first :
                #  不包含第一层decoder,  接受decoder的输出
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.forward_receive_queues, 
                                self.tensor_shape["decoder_forward"], 
                                self.pre_rank, 
                                num_iterations,
                                self.tensor_tag["decoder_forward"]))
            if not self.config.is_decoder_last: 
                # 不包含最后一层decoder, 发送decoder的输出
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    num_iterations,
                                    self.tensor_tag["decoder_forward"]))
                # 不包含最后一层decoder, 发送 encoder_outputs
                self.start_helper_thread(func=send_helper_thread, 
                                        args=(self.forward_encoder_output_send_queues, 
                                        self.next_rank,
                                        num_iterations,
                                        self.tensor_tag["encoder_output_forward"]))
        #################################################################################
        # backward 
        
        if self.config.is_encoder:
            if  self.config.is_encoder_last:
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.backward_encoder_output_receive_queues, 
                                self.tensor_shape["encoder_output_backward"], 
                                self.next_rank, 
                                num_iterations,
                                self.tensor_tag["encoder_output_backward"]))
            if not self.config.is_encoder_last:
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.backward_receive_queues, 
                                self.tensor_shape["encoder_backward"], 
                                self.next_rank, 
                                num_iterations,
                                self.tensor_tag["encoder_backward"]))
            if not self.config.is_encoder_first :
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.backward_send_queues, 
                                        self.pre_rank,
                                        num_iterations,
                                        self.tensor_tag["encoder_backward"]))
                
        
        if self.config.is_decoder:
            self.start_helper_thread(func=send_helper_thread, 
                                args=(self.backward_encoder_output_send_queues, 
                                    self.pre_rank,
                                    num_iterations,
                                    self.tensor_tag["encoder_output_backward"]))
            if not self.config.is_decoder_last:
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.backward_encoder_output_receive_queues, 
                                self.tensor_shape["encoder_output_backward"], 
                                self.next_rank, 
                                num_iterations,
                                self.tensor_tag["encoder_output_backward"]))
                self.start_helper_thread(func=recv_helper_thread, 
                            args=(self.backward_receive_queues, 
                                self.tensor_shape["decoder_backward"], 
                                self.next_rank, 
                                num_iterations,
                                self.tensor_tag["decoder_backward"]))
            if not self.config.is_decoder_first:
                self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.backward_send_queues, 
                                        self.pre_rank,
                                        num_iterations,
                                        self.tensor_tag["decoder_backward"]))
                
    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()


    def send(self, tensor, backward=False, encoder_output =False):
        if backward:
            if not encoder_output:
                self.backward_send_queues.add(tensor)
            else:
                self.backward_encoder_output_send_queues.add(tensor)
        else:            
            if encoder_output :
                self.forward_encoder_output_send_queues.add(tensor)
            else:
                self.forward_send_queues.add(tensor)

    
    def recv(self, backward=False, encoder_output =False):
        if backward:
            if  encoder_output:
                tensor = self.backward_encoder_output_receive_queues.remove()
            else:
                tensor = self.backward_receive_queues.remove()
        else:
            if encoder_output :
                tensor = self.forward_encoder_output_receive_queues.remove()
                tensor = tensor.requires_grad_() #TODO:
            else:
                tensor = self.forward_receive_queues.remove()
                tensor = tensor.requires_grad_()
        return tensor






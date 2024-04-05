import torch
import contextlib
import itertools
import time
from galaxy.core.pipeline_parallel.communication_t5  import CommunicationHandler
from galaxy.global_vars import get_args
import datetime

class PipelineRuntime():
    def __init__(self, config, model, loss_func, train_iter, optimizer, lr, if_cuda):
        self.config = config
        self.args = get_args()
        self.model = model
        self.stage = self.config.stage
        self.total_stage = self.config.total_stage
        self.comm_handler = CommunicationHandler(config)

        self.if_cuda = if_cuda
        self.tensors = []                   # 每一个元素是一个字典，字典里记录输入和输出位置的张量值,目前只支持单输入单输出
        self.loss_func = loss_func
        self.train_iter = train_iter        # training dataloader
        self.optimizer = optimizer(list(self.parameters()), lr=lr)

        self.num_microbatches = config.num_microbatches # 一个sync-round内micro-batch总数
        self.num_forward_micro_batch = 0    # 统计执行了多少个micro-batch的前向传播
        self.num_backward_micro_batch = 0

        self.training_iteration = 0     # 统计一共执行了多少次sync-round
        self.forward_time_total = 0.0
        self.backward_time_total = 0.0
        self.record = False
    def parameters(self):
        parameter_iterators = []
        parameter_iterators.append(self.model.parameters())
        return itertools.chain(*parameter_iterators)
    def set_record(self):
        self.record = True
        self.forward_time_total = 0.0
        self.backward_time_total= 0.0
    def send_tensors_forward(self ):
        # 如果是最后一个stage，返回
        if self.stage == self.total_stage-1:
            return 
        if self.config.is_encoder :  # encoder-only
            if not  self.config.is_encoder_last: # 不是最后一层encoder,传encoder_hidden_states
                tensor = self.tensors[-1]["fw_out"]
            # 发送tensor到下一个stage
                self.comm_handler.send(tensor, backward=False,encoder_output=False)
            if self.config.is_encoder_last: # 最后一层encoder
                # 发送encoder outputs
                encoder_outputs_tensor = self.tensors[-1]["fw_out_encoder_output"]
                self.comm_handler.send(encoder_outputs_tensor, backward=False, encoder_output=True)
        elif  self.config.is_decoder : # decoder-only
            if not  self.config.is_decoder_last: # 不是最后一层decoder
                tensor = self.tensors[-1]["fw_out"]
                self.comm_handler.send(tensor, backward=False,encoder_output=False)
                encoder_outputs_tensor = self.tensors[-1]["fw_out_encoder_output"]
                self.comm_handler.send(encoder_outputs_tensor, backward=False, encoder_output=True)
            

    def receive_tensors_forward(self, input_sample=None):
        
        # 如果是encoder 且第一层 获得  input_sample
        # 如果是encoder 且非第一层 获得    hidden states
        # 如果是decoder 且第一层，获得encoder output 
        # 如果是decoder 且非第一层，获得encoder output 和 hidden states
        self.tensors.append({})
        self.tensors[-1]["fw_in"]  = None
        self.tensors[-1]["fw_in_encoder_output"] = None
        if self.config.is_encoder  :  # encoder-only
            if self.config.is_encoder_first: #  第一层encoder
                if input_sample is not None:
                    inputs, _ = input_sample
                    self.tensors[-1]["fw_in"] = inputs 
                    return self.tensors[-1]["fw_in"], None # sample 不需要cuda
                else:
                    raise Exception("Missing input.")
            else: #  从前面encoder 获得 hidden_states
                tensor = self.comm_handler.recv(backward=False, encoder_output=False )
                self.tensors[-1]["fw_in"]  = tensor
                if self.if_cuda: 
                    self.tensors[-1]["fw_in"] = self.tensors[-1]["fw_in"].cuda()
                return self.tensors[-1]["fw_in"], None
        elif self.config.is_decoder: # decoder-only
            if not self.config.is_decoder_first: # 不是第一层decoder
                tensor = self.comm_handler.recv(backward=False, encoder_output=False ) # 从前面decoder 获得 hidden_states， 否则None
                self.tensors[-1]["fw_in"] = tensor
            # 全部都要decoder接受encoder output
            encoder_output_tensor = self.comm_handler.recv(backward=False, encoder_output=True )
            self.tensors[-1]["fw_in_encoder_output"] = encoder_output_tensor
            if self.if_cuda:
                if self.tensors[-1]["fw_in"] is not None:
                    self.tensors[-1]["fw_in"] = self.tensors[-1]["fw_in"].cuda()
                if self.tensors[-1]["fw_in_encoder_output"] is not None:
                    self.tensors[-1]["fw_in_encoder_output"] = self.tensors[-1]["fw_in_encoder_output"].cuda()
            
            return self.tensors[-1]["fw_in"], self.tensors[-1]["fw_in_encoder_output"]


    def send_tensors_backward(self, gradients, encoder_output_gradients ):
        # 如果stage为0，则返回
        if self.config.is_encoder:
            if self.config.is_encoder_first:
                return
            else:
                self.comm_handler.send(gradients, backward=True, encoder_output=False)
        if self.config.is_decoder:
            if self.config.is_decoder_first:
                self.comm_handler.send(encoder_output_gradients, backward=True, encoder_output=True)
            else:
                self.comm_handler.send(gradients, backward=True, encoder_output=False)
                self.comm_handler.send(encoder_output_gradients, backward=True, encoder_output=True)


    def receive_tensors_backward(self):
        # 最后一个stage，创建空字典
        gradients = None
        encoder_output_gradients = None
        if self.config.is_encoder: # encoder only
            if self.config.is_encoder_last:
                gradients = self.comm_handler.recv(backward=True,encoder_output=True)
            else:
                gradients = self.comm_handler.recv(backward=True,encoder_output=False)
        else: # decoder only 
            if self.config.is_decoder_last:
                pass 
            else:
                gradients = self.comm_handler.recv(backward=True,encoder_output=False)
                encoder_output_gradients = self.comm_handler.recv(backward=True,encoder_output=True)
            
        if self.if_cuda:
            if gradients is not None:
                gradients = gradients.cuda()
            if encoder_output_gradients is not None:
                encoder_output_gradients = encoder_output_gradients.cuda()
        return gradients,encoder_output_gradients
    
    def run_forward(self, input_sample=None):
        self.num_forward_micro_batch += 1
        print(f"start forward of microbatch {self.num_forward_micro_batch}")
        # 获取前向传播需要的数据
        fw_input, fw_in_encoder_output= self.receive_tensors_forward(input_sample)
        if self.config.is_encoder  :  # encoder-only
            torch.cuda.synchronize()
            forward_start = time.time()
            #####################################################################
            if self.config.is_encoder_first: #  第一层encoder
                encoder_outputs  = self.model(fw_input) # input_sample
            else:
                encoder_outputs  = self.model([fw_input])
            self.tensors[-1]["fw_out"] = encoder_outputs
            if self.config.is_encoder_last  : # 最后一层encoder
                self.tensors[-1]["fw_out_encoder_output"] = encoder_outputs
            #########################################################################
            torch.cuda.synchronize()
            forward_end = time.time()
            self.forward_time_total += (forward_end - forward_start)
            self.send_tensors_forward()   
        else: #decoder
            if self.config.is_decoder_last: # 最后一层decoder
                torch.cuda.synchronize()
                forward_start = time.time()
                ##############################################################
                out = self.model([fw_input, fw_in_encoder_output])
                labels = torch.ones(out.shape[0]).cuda().long()
                loss = self.loss_func(out, labels)
                self.tensors[-1]["loss"] = loss 
                ##############################################################
                torch.cuda.synchronize()
                forward_end = time.time()
                self.forward_time_total += (forward_end - forward_start)
            else : # 不是最后一层decoder
                torch.cuda.synchronize()
                forward_start = time.time()
                ################################################################
                decoder_outputs = self.model([fw_input, fw_in_encoder_output])
                self.tensors[-1]["fw_out"] = decoder_outputs
                self.tensors[-1]["fw_out_encoder_output"]  = fw_in_encoder_output
                ##################################################################
                torch.cuda.synchronize()
                forward_end = time.time()
                self.forward_time_total += (forward_end - forward_start)
                self.send_tensors_forward()
        print(f"finish forward of microbatch {self.num_forward_micro_batch}")

    def run_backward(self):
        self.num_backward_micro_batch += 1
        print(f"start backward of microbatch {self.num_backward_micro_batch}")
        # 先进行BP的micro-batch一定是先执行FP的
        tensors = self.tensors.pop(0)
        # 获取stage model 输入张量(input)和输出张量(output)
        # 及对应梯度input_gradient,output_gradient
        input_gradient = None
        input_encoder_output_gradient = None
        # 接收下一个stage传来的梯度,
        # 如果是decoder 有两部分
        # 如果是encoder 只有一部分
        output_gradient, output_gradient_for_encoder_outputs = self.receive_tensors_backward()

        if self.config.is_encoder: # encoder 一个输入 一个输出
            output_tensor = tensors["fw_out"]
            input_tensor = tensors["fw_in"]
        else: # decoder 
            if self.config.is_decoder_last: # decoder输入有两个
                output_tensor = tensors["loss"]
                input_tensor = tensors["fw_in"]
                input_encoder_output = tensors["fw_in_encoder_output"]
            else:
                output_tensor = tensors["fw_out"]
                input_tensor = tensors["fw_in"]
                input_encoder_output = tensors["fw_in_encoder_output"]
        # register_hook会在反向传播过程中被触发，并且传入参数为梯度
        def hook_wrapper():
            def hook(gradient):
                nonlocal input_gradient
                print("Hook called!")
                input_gradient = gradient
            return hook
        def hook_encoder_output_wrapper():
            def  hook_encoder_output(gradient):
                nonlocal input_encoder_output_gradient
                print("hook_encoder_output_wrapper called!")
                input_encoder_output_gradient = gradient
            return hook_encoder_output
        torch.cuda.synchronize()
        backward_start = time.time()
        ###################################################################
        if self.config.is_encoder:
            # stage0的fw_input不用计算梯度，
            if not self.config.is_encoder_first:
                input_tensor.requires_grad_()
                input_tensor.register_hook(hook_wrapper())
            
            if self.config.is_encoder_last:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))
            else:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))
        else:
            input_encoder_output.requires_grad_()
            input_encoder_output.register_hook(hook_encoder_output_wrapper())
            if not self.config.is_decoder_first:  #第一个encoder， input也不需要计算梯度
                input_tensor.requires_grad_()
                input_tensor.register_hook(hook_wrapper())
            
            if self.config.is_decoder_last:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=None)
            else:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))
        ##############################################################################################
        torch.cuda.synchronize()
        backward_end= time.time()
        self.backward_time_total += (backward_end - backward_start)
        # 发送梯度到上一个stage
        if input_encoder_output_gradient is not None:
            print("send input_encoder_output_gradient shape {} type {}".format(input_encoder_output_gradient.shape, input_encoder_output_gradient.dtype))
        self.send_tensors_backward(input_gradient,input_encoder_output_gradient)
        print(f"finish backward of microbatch {self.num_backward_micro_batch}")


    def forward_backward_pipelining(self):
        """
        Gpipe流水线并行(没有all-reduce和1F1B)
        """
        # 同时注入多个micro-batch进入pipeline
        for mb_id in range(self.config.num_microbatches):
            # 如果是第一个stage需要生成数据
            if self.stage == 0: 
                input_sample = next(self.train_iter)
                self.run_forward(input_sample)
            else:
                self.run_forward()
            
        # 清空梯度
        self.optimizer.zero_grad()
        # 完成所有micro-batch的FP，开始反向传播
        for mb_id in range(self.config.num_microbatches):
            self.run_backward()
        self.optimizer.step()
        self.training_iteration += 1
        print(f"Finish {self.training_iteration}-th iteration!")
        
    def run_iteration(self,num_iteration):
        for i in range(num_iteration):
            self.forward_backward_pipelining()

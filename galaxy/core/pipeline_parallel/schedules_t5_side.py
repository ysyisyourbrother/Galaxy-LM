import torch
import contextlib
import itertools
import time
from galaxy.core.pipeline_parallel.communication_t5_side import CommunicationHandler
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

    def parameters(self):
        parameter_iterators = []
        parameter_iterators.append(self.model.parameters())
        return itertools.chain(*parameter_iterators)

    def send_tensors_forward(self ):
        # 如果是最后一个stage，返回
        if self.stage == self.total_stage-1:
            return 
        if self.config.is_encoder :  # encoder-only
            if not self.config.is_encoder_last: # 不是最后一层encoder,传encoder_hidden_states
                tensor = self.tensors[-1]["fw_out"]
                side_tensor = self.tensors[-1]["side_fw_out"]
                self.comm_handler.send(tensor, backward=False,encoder_output=False,side=False)
                self.comm_handler.send(side_tensor, backward=False,encoder_output=False,side=True)
                print("Stage {} send forward hidden_states shape {} type {}".format(self.stage , tensor.shape, tensor.dtype))
                print("Stage {} send side forward hidden_states shape {} type {}".format(self.stage , side_tensor.shape, side_tensor.dtype))
            if self.config.is_encoder_last: # 最后一层encoder
                # 发送encoder outputs
                encoder_outputs_tensor = self.tensors[-1]["fw_out_encoder_output"]
                side_encoder_outputs_tensor = self.tensors[-1]["side_fw_out_encoder_output"]            
                self.comm_handler.send(encoder_outputs_tensor, backward=False, encoder_output=True,side=False)
                self.comm_handler.send(side_encoder_outputs_tensor, backward=False, encoder_output=True,side=True)
                print("Stage {} send  fw_out_encoder_output shape {} type {}".format(self.stage , encoder_outputs_tensor.shape, encoder_outputs_tensor.dtype))
                print("Stage {} send  side fw_out_encoder_output shape {} type {}".format(self.stage , side_encoder_outputs_tensor.shape, side_encoder_outputs_tensor.dtype))

        elif  self.config.is_decoder : # decoder-only
            if not  self.config.is_decoder_last: # 不是最后一层decoder
                tensor = self.tensors[-1]["fw_out"]
                side_tensor = self.tensors[-1]["side_fw_out"]
                self.comm_handler.send(tensor, backward=False,encoder_output=False)
                self.comm_handler.send(side_tensor, backward=False,encoder_output=False,side=True)
                print("Stage {} send forward tensor shape {} type {}".format(self.stage , tensor.shape, tensor.dtype))
                print("Stage {} send side forward hidden_states shape {} type {}".format(self.stage , side_tensor.shape, side_tensor.dtype))

                encoder_outputs_tensor = self.tensors[-1]["fw_out_encoder_output"]
                side_encoder_outputs_tensor = self.tensors[-1]["side_fw_out_encoder_output"]            
                self.comm_handler.send(encoder_outputs_tensor, backward=False, encoder_output=True,side=False)
                self.comm_handler.send(side_encoder_outputs_tensor, backward=False, encoder_output=True,side=True)
                print("Stage {} send  fw_out_encoder_output shape {} type {}".format(self.stage , encoder_outputs_tensor.shape, encoder_outputs_tensor.dtype))
                print("Stage {} send  side fw_out_encoder_output shape {} type {}".format(self.stage , side_encoder_outputs_tensor.shape, side_encoder_outputs_tensor.dtype))

        
        
        

    def receive_tensors_forward(self, input_sample=None):
        
        # 如果是encoder 且第一层 获得  input_sample
        # 如果是encoder 且非第一层 获得    hidden states
        # 如果是decoder 且第一层，获得encoder output 
        # 如果是decoder 且非第一层，获得encoder output 和 hidden states
        self.tensors.append({})
        self.tensors[-1]["fw_in"]  = None
        self.tensors[-1]["side_fw_in"]  = None
        self.tensors[-1]["fw_in_encoder_output"] = None
        self.tensors[-1]["side_fw_in_encoder_output"] = None
        if self.config.is_encoder  :  # encoder-only
            if self.config.is_encoder_first: #  第一层encoder
                if input_sample is not None:
                    inputs, _ = input_sample
                    self.tensors[-1]["fw_in"] = inputs 
                    print("Stage {} receive input_sample ".format(self.stage  ))
                    return self.tensors[-1]["fw_in"], None , None , None # sample 不需要cuda
                else:
                    raise Exception("Missing input.")
            else: #  从前面encoder 获得 hidden_states 以及side_hidden_states
                tensor = self.comm_handler.recv(backward=False, encoder_output=False ,side=False)
                side_tensor = self.comm_handler.recv(backward=False,encoder_output=False,side=True)
                print("Stage {} receive hidden_states size {}  ".format(self.stage, tensor.shape  ))
                print("Stage {} receive side hidden_states size {}  ".format(self.stage, side_tensor.shape  ))
                self.tensors[-1]["fw_in"]  = tensor
                self.tensors[-1]["side_fw_in"]  = side_tensor
                if self.if_cuda: 
                        self.tensors[-1]["fw_in"] = self.tensors[-1]["fw_in"].cuda()
                        self.tensors[-1]["side_fw_in"] = self.tensors[-1]["side_fw_in"].cuda()
                return self.tensors[-1]["fw_in"], self.tensors[-1]["side_fw_in"], None, None
        elif self.config.is_decoder: # decoder-only
            if not self.config.is_decoder_first: # 不是第一层decoder
                tensor = self.comm_handler.recv(backward=False, encoder_output=False, side=False) # 从前面decoder 获得 hidden_states， 否则None
                side_tensor = self.comm_handler.recv(backward=False,encoder_output=False,side=True)
                self.tensors[-1]["fw_in"] = tensor
                self.tensors[-1]["side_fw_in"]  = side_tensor
            # 全部都要decoder接受encoder output
            encoder_output_tensor = self.comm_handler.recv(backward=False, encoder_output=True , side=False)
            side_encoder_output_tensor = self.comm_handler.recv(backward=False, encoder_output=True , side=True)
            self.tensors[-1]["fw_in_encoder_output"] = encoder_output_tensor
            self.tensors[-1]["side_fw_in_encoder_output"] = side_encoder_output_tensor
            if self.if_cuda:
                if self.tensors[-1]["fw_in"] is not None:
                    self.tensors[-1]["fw_in"] = self.tensors[-1]["fw_in"].cuda()
                if self.tensors[-1]["side_fw_in"] is not None:
                    self.tensors[-1]["side_fw_in"] = self.tensors[-1]["side_fw_in"].cuda()
                if self.tensors[-1]["fw_in_encoder_output"] is not None:
                    self.tensors[-1]["fw_in_encoder_output"] = self.tensors[-1]["fw_in_encoder_output"].cuda()
                if self.tensors[-1]["side_fw_in_encoder_output"] is not None:
                    self.tensors[-1]["side_fw_in_encoder_output"] = self.tensors[-1]["side_fw_in_encoder_output"].cuda()
            
            return self.tensors[-1]["fw_in"], self.tensors[-1]["side_fw_in"],self.tensors[-1]["fw_in_encoder_output"],self.tensors[-1]["side_fw_in_encoder_output"]


    def send_tensors_backward(self, side_gradients, side_encoder_output_gradients ):
        # 如果stage为0，则返回
        if self.config.is_encoder:
            if self.config.is_encoder_first:
                return
            else:
                self.comm_handler.send(side_gradients, backward=True, encoder_output=False,side=True)
        if self.config.is_decoder:
            if self.config.is_decoder_first:
                print(side_encoder_output_gradients.shape)
                self.comm_handler.send(side_encoder_output_gradients, backward=True, encoder_output=True,side=True)
            else:
                self.comm_handler.send(side_gradients, backward=True, encoder_output=False,side=True)
                self.comm_handler.send(side_encoder_output_gradients, backward=True, encoder_output=True,side=True)
        # 发送gradients到上一个stage
        if side_gradients is not None:
            print("Stage {} send backward side_gradients shape {} type {}".format(self.stage , side_gradients.shape, side_gradients.dtype))
        if side_encoder_output_gradients is not None:
            print("Stage {} send backward side_encoder_output_gradients shape {} type {}".format(self.stage , side_encoder_output_gradients.shape, side_encoder_output_gradients.dtype))
    


    def receive_tensors_backward(self):
        # 最后一个stage，创建空字典
        side_gradients = None
        side_encoder_output_gradients = None
        if self.config.is_encoder: # encoder only
            if self.config.is_encoder_last:
                side_encoder_output_gradients = self.comm_handler.recv(backward=True,encoder_output=True,side=True)
            else:
                side_gradients = self.comm_handler.recv(backward=True,encoder_output=False,side=True)
        else: # decoder only 
            if self.config.is_decoder_last:
                pass 
            else:
                side_gradients = self.comm_handler.recv(backward=True,encoder_output=False,side=True)
                side_encoder_output_gradients = self.comm_handler.recv(backward=True,encoder_output=True,side=True)
            
        if self.if_cuda:
            if side_gradients is not None:
                side_gradients = side_gradients.cuda()
            if side_encoder_output_gradients is not None:
                side_encoder_output_gradients = side_encoder_output_gradients.cuda()
        return side_gradients,side_encoder_output_gradients

    def run_forward(self, input_sample=None):
        self.num_forward_micro_batch += 1
        print(f"start forward of microbatch {self.num_forward_micro_batch}")
        # 获取前向传播需要的数据
        fw_input, side_fw_input,fw_in_encoder_output, side_fw_in_encoder_output= self.receive_tensors_forward(input_sample)
        print("Stage {} receive fw_input  ".format(self.stage  ))
        if fw_input is not None and not self.config.is_encoder_first:
            print("Stage {} fw_input is Shape {}".format(self.stage, fw_input.shape))  
        if fw_in_encoder_output is not None:
            print("Stage {} fw_in_encoder_output is Shape {}".format(self.stage, fw_in_encoder_output.shape))
        if self.config.is_encoder  :  # encoder-only
            if self.config.is_encoder_first: #  第一层encoder
                encoder_outputs ,side_encoder_outputs = self.model(fw_input) # input_sample
            else:
                encoder_outputs,side_encoder_outputs  = self.model([fw_input, side_fw_input])
            self.tensors[-1]["fw_out"] = encoder_outputs
            self.tensors[-1]["side_fw_out"]  = side_encoder_outputs
            if self.config.is_encoder_last  : # 最后一层encoder
                self.tensors[-1]["fw_out_encoder_output"] = encoder_outputs
                self.tensors[-1]["side_fw_out_encoder_output"] = side_encoder_outputs
            self.send_tensors_forward()   
        else: #decoder
            if self.config.is_decoder_last: # 最后一层decoder
                out = self.model([fw_input, 
                                side_fw_input,
                                fw_in_encoder_output,
                                side_fw_in_encoder_output])
                labels = torch.ones(out.shape[0]).cuda().long()
                loss = self.loss_func(out, labels)
                self.tensors[-1]["loss"] = loss 
            else : # 不是最后一层decoder
                decoder_outputs,side_decoder_outputs = self.model([fw_input, 
                                                                   side_fw_input,
                                                                   fw_in_encoder_output,
                                                                   side_fw_in_encoder_output])
                self.tensors[-1]["fw_out"] = decoder_outputs
                self.tensors[-1]["side_fw_out"] = side_decoder_outputs
                self.tensors[-1]["fw_out_encoder_output"]  = fw_in_encoder_output
                self.tensors[-1]["side_fw_out_encoder_output"]  = side_fw_in_encoder_output
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
        if output_gradient != None:
            print("output_gradient:", output_gradient.shape, output_gradient.dtype)
        if output_gradient_for_encoder_outputs != None:
            print("output_gradient_for_encoder_outputs:", output_gradient_for_encoder_outputs.shape, output_gradient_for_encoder_outputs.dtype)

        if self.config.is_encoder: # encoder 一个输入 一个输出
            output_tensor = tensors["side_fw_out"]
            input_tensor = tensors["side_fw_in"]
        else: # decoder 
            if self.config.is_decoder_last : # decoder输入有两个
                output_tensor = tensors["loss"]
                input_tensor = tensors["side_fw_in"]
                input_encoder_output = tensors["side_fw_in_encoder_output"]
            else:
                output_tensor = tensors["side_fw_out"]
                input_tensor = tensors["side_fw_in"]
                input_encoder_output = tensors["side_fw_in_encoder_output"]
        # register_hook会在反向传播过程中被触发，并且传入参数为梯度
        def hook_wrapper():
            def hook(gradient):
                nonlocal input_gradient
                print("Hook called!")  # Debug print to check if the hook is executed
                input_gradient = gradient
            return hook
        def hook_encoder_output_wrapper():
            def hook(gradient):
                nonlocal input_encoder_output_gradient
                input_encoder_output_gradient = gradient
            return hook
        
        # 除了其他stage的fw_input都要计算梯度，并保存在input_gradients
        if self.config.is_encoder:
            # stage0的fw_input不用计算梯度，
            if not self.config.is_encoder_first:
                input_tensor.requires_grad_()
                input_tensor.register_hook(hook_wrapper())
                if self.config.is_encoder_last:
                    torch.autograd.backward(tuple([output_tensor]),  grad_tensors=tuple([output_gradient_for_encoder_outputs]))
                else:
                    torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))
        else: # decoder only 
            input_encoder_output.requires_grad_()
            input_encoder_output.register_hook(hook_encoder_output_wrapper())
            if not self.config.is_decoder_first:  #第一个encoder， input也不需要计算梯度
                input_tensor.requires_grad_()
                input_tensor.register_hook(hook_wrapper())
            ##########################################################
            if self.config.is_decoder_last:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=None)
            else:
                torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))
        # 发送梯度到上一个stage
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
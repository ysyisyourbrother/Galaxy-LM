import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import contextlib
import itertools

from galaxy.core.pipeline_parallel.communication import CommunicationHandler
from galaxy.global_vars import get_args


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

    def send_tensors_forward(self):
        # 如果是最后一个stage，返回
        if self.stage == self.total_stage-1:
            return 

        tensor = self.tensors[-1]["fw_out"]
        # 发送tensor到下一个stage
        self.comm_handler.send(tensor, backward=False)
        

    def receive_tensors_forward(self, input_sample=None):
        # 如果是第一个stage，从data_loader中获得input
        self.tensors.append({})
        if self.stage == 0:
            if input_sample is not None:
                inputs, target = input_sample
                self.tensors[-1]["fw_in"] = inputs 
                # TODO: 如何将target发送到最后一个设备计算loss?
                # self.tensors[-1]["target"] = target 
            else:
                raise Exception("Missing input.")
        else:
            # 从上一台机器接收tensor
            tensor = self.comm_handler.recv(backward=False)
            if self.if_cuda:
                tensor = tensor.cuda()
            self.tensors[-1]["fw_in"] = tensor 

        return self.tensors[-1]["fw_in"]


    def send_tensors_backward(self, gradients):
        # 如果stage为0，则返回
        if self.stage == 0:
            return  

        # 发送gradients到上一个stage
        self.comm_handler.send(gradients, backward=True)


    def receive_tensors_backward(self):
        # 最后一个stage，创建空字典
        if self.stage == self.total_stage - 1:
            return None
        else:
            gradients = self.comm_handler.recv(backward=True)
            if self.if_cuda:
                gradients = gradients.cuda()
            return gradients

    def run_forward(self, input_sample=None):
        self.num_forward_micro_batch += 1
        print(f"start forward of microbatch {self.num_forward_micro_batch}")
        # 获取前向传播需要的数据
        fw_input = self.receive_tensors_forward(input_sample)

        # 最后一个stage，执行loss function
        if self.stage == self.total_stage - 1:
            # forward pass
            output = self.model(fw_input)

            # TODO: 怎么把label传送到最后一个stage
            labels = torch.ones(output.shape[0]).cuda().long()
            loss = self.loss_func(output, labels)
            self.tensors[-1]["loss"] = loss 

        # 不是最后一个stage，正常执行前向传播
        else:
            output = self.model(fw_input)
            self.tensors[-1]["fw_out"] = output
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
        # 接收下一个stage传来的梯度
        output_gradient = self.receive_tensors_backward()
        if self.stage == self.total_stage - 1:
            # 最后一个stage的output为loss
            output_tensor = tensors["loss"]
            input_tensor = tensors["fw_in"]
        else:
            output_tensor = tensors["fw_out"]
            input_tensor = tensors["fw_in"]

        # register_hook会在反向传播过程中被触发，并且传入参数为梯度
        def hook_wrapper():
            def hook(gradient):
                nonlocal input_gradient
                input_gradient = gradient
            return hook

        # 除了stage0的fw_input不用计算梯度，其他stage的fw_input都要计算梯度，并保存在input_gradients
        if self.stage != 0:
            input_tensor.requires_grad_()
            input_tensor.register_hook(hook_wrapper())

        if self.stage == self.total_stage - 1:
            torch.autograd.backward(tuple([output_tensor]), grad_tensors=None)
        else:
            torch.autograd.backward(tuple([output_tensor]), grad_tensors=tuple([output_gradient]))

        # 发送梯度到上一个stage
        self.send_tensors_backward(input_gradient)

        print(f"finish backward of microbatch {self.num_backward_micro_batch}")


    def forward_backward_pipelining(self):
        """Gpipe流水线并行(没有all-reduce和1F1B)
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

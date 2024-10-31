from galaxy.activations.utils import get_activation
import torch.nn as nn
from galaxy.loralib.utils import mark_only_lora_as_trainable

def get_parameter_number_of_name(model, name):
        # mlp para:
    num = 0
    total_num = sum(p.numel() for p in model.parameters())
    for n, p in model.named_parameters():
        if  name  in n:
            num +=  p.numel()
    print("{} in model: {} , {} of totel para".format(name,num, num/total_num*100 ))

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 

class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)
class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(config, 'hidden_size'):
            self.input_dim = config.hidden_size
        elif hasattr(config, 'd_model'):
            self.input_dim = config.d_model
        else:
            raise NotImplementedError
        self.down_sample_size = self.input_dim // config.adapter_reduction_dim
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 
    
def mark_only_adapter_as_trainable(model: nn.Module)  -> None:
    print("mark only adapter as trainable....")
    for n, p in model.named_parameters():
        if 'adapter' not in n:
            p.requires_grad = False
            
def mark_only_side_as_trainable(model: nn.Module)  -> None:
    print("mark only side as trainable....")
    for n, p in model.named_parameters():
        if 'side' not in n:
            p.requires_grad = False


    
def modify_model_for_peft(model: nn.Module, config)  -> None:
    if config.full_model:
        pass
    elif config.use_lora:
        mark_only_lora_as_trainable(model)
    elif config.use_adapter:
        mark_only_adapter_as_trainable(model)
    elif config.use_side:
        mark_only_side_as_trainable(model)
    elif config.use_side_only:
        mark_only_side_as_trainable(model)
    else:
        raise NotImplementedError
    print('number of model parameters:', get_parameter_number(model))
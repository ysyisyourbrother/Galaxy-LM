from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoTokenizer)



def get_pruning_dict(args, qst_config, data_type,):
    if "t5" in args.model_name_or_path or "T5" in args.model_name_or_path:
        from  .pruning_methods_t5 import pruning_T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained( args.model_name_or_path,torch_dtype = data_type)
        method = pruning_T5ForConditionalGeneration
        print(model.state_dict()["decoder.block.8.layer.1.EncDecAttention.k.weight"].shape)
        print(method.__name__)
        pruned_state_dict = method(model,  reduction_factor=qst_config.r, importance_measure= None)
        print(pruned_state_dict ["decoder.block.8.layer.1.EncDecAttention.k.weight"].shape)
    elif "bart" in args.model_name_or_path or "BART" in args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained( args.model_name_or_path,torch_dtype = data_type)
        print(model)
        print(model.state_dict()["model.decoder.layers.8.self_attn.out_proj.weight"].shape)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        from .pruning_methods_bart import pruning_without_residual
        method = pruning_without_residual
        pruned_state_dict = method(model, tokenizer, qst_config.r )
        print(pruned_state_dict ["model.decoder.layers.8.self_attn.out_proj.weight"].shape)
        # print(pruned_state_dict.keys())
    else:
        raise NotImplementedError
    return pruned_state_dict
def init_side_network_bart(args, qst_config, model):
    if "bart" not in args.model_name_or_path:
        raise NotImplementedError
    if args.init_side_method == "none":
        print("Not init side_network")
        return model
    elif args.init_side_method == "zero":
        print("Init side_network by zero")
        for name, param in model.named_parameters():
            if 'side' in name:
                param.data.fill_(0)  # 将参数值赋为0
        return model
    elif args.init_side_method == "pruning":
        print("Init side_network by pruning")
        pruned_state_dict = get_pruning_dict(args, qst_config,data_type=model.config.torch_dtype)
        # model is QSTBartForSequenceClassification
        for n, p in model.named_parameters():
            if "side_layers" in n:
                infer_n = n.split(".")
                infer_n[2] = "layers"
                infer_n = ".".join(infer_n)
                state = pruned_state_dict[infer_n]
                # if args.weight_scale:
                #     keep_ratio = (state.shape[0] / pruned_state_dict[infer_n].shape[0]) ** 0.5
                #     print(keep_ratio)
                # else:
                #     keep_ratio = 1
                # p.data.copy_(state/keep_ratio)
                p.data.copy_(state)
                print(infer_n,"-->",n)    
        return model
def init_side_network_t5(args, qst_config, model):
    print("before init",model.state_dict()["transformer.encoder.side_block.0.layer.0.SelfAttention.k.weight"][0][0].item())
    # init side-transformer
    if "t5" not in args.model_name_or_path:
        raise NotImplementedError
    if args.init_side_method == "none":
        print("Not init side_network")
    elif args.init_side_method == "zero":
        print("Init side_network by zero")
        for name, param in model.named_parameters():
            if 'side' in name:
                print("fill",name,"by zero")
                param.data.fill_(0)  # 将参数值赋为0
    elif args.init_side_method == "pruning":
        print("Init side_network by pruning")
        pruned_state_dict = get_pruning_dict(args, qst_config,data_type=model.config.torch_dtype)
        # model is QSTForSequenceClassification
        for n, p in model.named_parameters():
            if "side_block" in n:
                # n = "transformer.decoder.side_block.11.layer.1.EncDecAttention.o.weight"
                # infer_n = "decoder.block.11.layer.1.EncDecAttention.o.weight"
                infer_n = n.split(".")
                infer_n[2] = "block"
                infer_n = ".".join(infer_n[1:])
                print(infer_n,"-->",n)
                state = pruned_state_dict[infer_n]
                p.data.copy_(state)
                
            if "side_final_layer_norm" in n:
                # n = "transformer.encoder.side_final_layer_norm.weight"
                # infer_n = 'decoder.final_layer_norm.weight'
                infer_n = n.split(".")
                infer_n[2] = "final_layer_norm"
                infer_n = ".".join(infer_n[1:])
                state = pruned_state_dict[infer_n]
                p.data.copy_(state)
                print(infer_n,"-->",n)
    else:
        raise NotImplementedError
    print("after init",model.state_dict()["transformer.encoder.side_block.0.layer.0.SelfAttention.k.weight"][0][0].item())
    return model

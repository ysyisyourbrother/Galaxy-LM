import sys, os, copy
import numpy as np
import torch
import torch_pruning as tp
import transformers

import functools

def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    n_remain = round_to*max(int(total_parameters - n_to_prune)//round_to, 1)
    return max(total_parameters - n_remain, 0)
def select_weights(weights, idxs):
    num_features = len(weights)
    keep_idxs = list(set(range(num_features)) - set(idxs))

    return weights[keep_idxs]

def pruning_without_residual(model, tokenizer, reduction_factor, importance_measure=None, version=2, num_heads=12, iterations=2):
    inputs_dict = tokenizer(["I like dogs.", "I really like your cats."], padding=True, return_tensors="pt")

    inputs_dict = {"input_ids": inputs_dict["input_ids"], "attention_mask": inputs_dict["attention_mask"]} # transform to dict

    # Build dependency graph
    DG = tp.DependencyGraph()
    
    DG.build_dependency(model, example_inputs = inputs_dict, pruning_dim = -1) # Note to set pruning_dim to -1 to prune BertModel on hidden_states.

    # get a pruning plan by pruning from word embedding

    if version in [2, 5]:
        strategy_for_attn = tp.strategy.L1Strategy()
        strategy_for_others = tp.strategy.L1Strategy()
    elif version == 3:
        strategy_for_attn = tp.strategy.AttnL1Strategy()
        strategy_for_others = tp.strategy.L1Strategy()
        num_heads = num_heads
        strategy_for_attn = functools.partial(strategy_for_attn, num_heads=num_heads)
    elif version == 4:
        strategy_for_attn = tp.strategy.AttnL1Strategy()
        strategy_for_others = tp.strategy.AttnL1Strategy()

        strategy_for_attn = functools.partial(strategy_for_attn, num_heads=num_heads)
        strategy_for_others = functools.partial(strategy_for_attn, num_heads=num_heads)

    if version in [2, 3, 4]:
        prune_vals = [1 - 1 / reduction_factor]
    elif version in [5]:
        prune_vals = np.linspace(0, 1 - 1 / reduction_factor, iterations)
        # print(np.rint(prune_vals * 768))
        keep_vals = 1 - prune_vals
        keep_vals_now = keep_vals[1:]
        keep_vals_pre = keep_vals[:-1] 
        prune_vals = (keep_vals_pre - keep_vals_now) / keep_vals_pre

        prune_vals = prune_vals.tolist()
        # round_pruning_amount(n, n_to_prune, round_to)

    state_dict = model.state_dict()
    
    if importance_measure is None:
        importance_measure = copy.deepcopy(state_dict)

    # construct ordered layers to process
    ordered_target_layers = ["model.shared.weight"]

    sub_model = "encoder"
    ordered_target_layers.append( f"model.{sub_model}.embed_positions.weight" )
    ordered_target_layers.append([f"model.{sub_model}.layernorm_embedding.weight"])
    ordered_target_layers.append([f"model.{sub_model}.layernorm_embedding.bias"])
    for i in range(model.config.encoder_layers):
        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.self_attn.{n}_proj.weight" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.self_attn.{n}_proj.bias" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn.out_proj.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn.out_proj.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn_layer_norm.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn_layer_norm.bias"])

        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc1.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc1.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc2.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc2.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.final_layer_norm.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.final_layer_norm.bias"])

    sub_model = "decoder"
    ordered_target_layers.append(f"model.{sub_model}.embed_positions.weight")
    ordered_target_layers.append([f"model.{sub_model}.layernorm_embedding.weight"])
    ordered_target_layers.append([f"model.{sub_model}.layernorm_embedding.bias"])
    for i in range(model.config.decoder_layers):
        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.self_attn.{n}_proj.weight" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.self_attn.{n}_proj.bias" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn.out_proj.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn.out_proj.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn_layer_norm.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.self_attn_layer_norm.bias"])

        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.encoder_attn.{n}_proj.weight" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append(
            [f"model.{sub_model}.layers.{i}.encoder_attn.{n}_proj.bias" for n in ["q", "k", "v"]]
            )
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.encoder_attn.out_proj.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.encoder_attn.out_proj.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.encoder_attn_layer_norm.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.encoder_attn_layer_norm.bias"])

        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc1.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc1.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc2.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.fc2.bias"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.final_layer_norm.weight"])
        ordered_target_layers.append([f"model.{sub_model}.layers.{i}.final_layer_norm.bias"])

    pruning_idxs_first_layer = None

    for prune_val in prune_vals:
        new_state_dict = {}
        for layer in ordered_target_layers:
            if isinstance(layer, list):
                is_layernorm = all(["layer_norm" in sub_layer for sub_layer in layer]) or all(["layernorm" in sub_layer for sub_layer in layer])
                is_attn = all(["attn" in sub_layer for sub_layer in layer]) and not is_layernorm
                is_bias = all(["bias" in sub_layer for sub_layer in layer])

                if is_bias or is_layernorm:
                    # use previous idx to prune the weights

                    for sub_layer in layer:
                        weights = state_dict[sub_layer]
                        weights = select_weights(weights, pruning_idxs)

                        importance = importance_measure[sub_layer]
                        importance = select_weights(importance, pruning_idxs)
                        
                        new_state_dict[sub_layer] = weights
                        importance_measure[sub_layer] = importance

                else:
                    # the most common case
                    # will prune the weights according to previous prune idx and select next prune idx
                    if is_attn:
                        strategy = strategy_for_attn
                    else:
                        strategy = strategy_for_others
                    
                    weights = [state_dict[sub_layer] for sub_layer in layer]
                    importances = [importance_measure[sub_layer] for sub_layer in layer]

                    # prune according to previous idx
                    weights = [select_weights(w.T, pruning_idxs).T for w in weights]
                    importances = [select_weights(imp.T, pruning_idxs).T for imp in importances]

                    # use the sum of log values instead of the product of values
                    prod_imp = 0
                    for imp in importances:
                        prod_imp += torch.log(imp)

                    pruning_idxs = strategy(weights=prod_imp, amount=prune_val)

                    weights = [select_weights(w, pruning_idxs) for w in weights]
                    importances = [select_weights(imp, pruning_idxs) for imp in importances]

                    for l, w in zip(layer, weights):
                        new_state_dict[l] = w

                    # update importance measure
                    for l, imp in zip(layer, importances):
                        importance_measure[l] = imp

            elif "shared.weight" in layer:
                # the first layer
                # will only select next prune idx
                # and direct copy the same weights
                importance = importance_measure[layer]
                weights = state_dict[layer]
                pruning_idxs = strategy_for_others(weights=importance.T, amount=prune_val)

                weights = select_weights(weights.T, pruning_idxs).T
                importance = select_weights(importance.T, pruning_idxs).T

                new_state_dict[layer] = weights
                importance_measure[layer] = importance

                pruning_idxs_first_layer = pruning_idxs
            
            else:
                # embed_positions, use the the pruning_idxs_first_layer to prune, and set the pruning_idxs with it for the next layer.
                pruning_idxs = pruning_idxs_first_layer
                weights = state_dict[layer]
                weights = select_weights(weights.T, pruning_idxs).T

                importance = importance_measure[layer]
                importance = select_weights(importance.T, pruning_idxs).T
                
                new_state_dict[layer] = weights
                importance_measure[layer] = importance

        state_dict = new_state_dict

    return new_state_dict

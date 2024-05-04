# Flops Profiler

https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler

```python
from galaxy.profiler.utils import FlopsProfiler
profile_step = 5
print_profile= True
prof = FlopsProfiler(model)
for step in range(10):
    # start profiling at training step "profile_step"
    if step == profile_step:
        prof.start_profile()
    # foward
    trains, labels = next(train_iter)
    outputs = model(trains)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    # end profiling and print output
    if step == profile_step: # if using multi nodes, check global_rank == 0 as well
        prof.stop_profile()
        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        if print_profile:
            prof.print_model_profile(profile_step=profile_step)
        prof.end_profile()
```

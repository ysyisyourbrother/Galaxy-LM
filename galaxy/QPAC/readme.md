参数由 json 文件确定

```shell
CUDA_VISIBLE_DEVICES=1 python run_glue.py  configs/t5-base/lst.json
```

**scripts**

```shell
# bash scripts/xx.sh ${gpu id} ${task_name}
bash scripts/t5-base/qst.sh 1 mrpc
```

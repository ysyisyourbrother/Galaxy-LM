# PAC

- Src: https://github.com/ylsung/Ladder-Side-Tuning
- Model: [t5-base](https://huggingface.co/google-t5/t5-base/tree/main)

## env

```shell
conda create -n lst python=3.8
source activate lst
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
cd Ladder-Side-Tuning
pip install -e .
cd Torch-Pruning
pip install -e .

pip install protobuf==3.19.0
pip install -U numpy==1.20.0
pip install pandas==1.3.3
```

## scripts

- 路径: `galaxy/PAC/scripts`

```shell
conda activate lst
cd galaxy/PAC
bash scripts/t5base/baseline.sh 0 rte #  full parameter
bash scripts/t5base/lora.sh 0 rte # lora
bash scripts/t5base/adapter.sh 0 rte # adapter tuning
bash scripts/t5base/side.sh 0 rte # side tuning
```

- 第一个参数: 选择 gpu id
- 第二个参数: 选择 task name [rte cola sst mrpc qqp stsb sst2 qnli mnli]
- `.sh` 文件中需要修改权重路径 ( `.sh`内修改对应 `.json`文件参数)
- 设置 epoch: `galaxy/PAC/scripts/env.sh`
- 结果路径: `galaxy/PAC/all_output_logs`

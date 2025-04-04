apt install tmux

tmux new

conda create --name llava --clone torch

conda activate llava

pip install flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

pip install transformers==4.50.3

pip install decord==0.6.0

pip install av==14.0.1

pip install open_clip_torch

pip install accelerate>=0.26.0



python3 prepare_quota.py --cached-data-root=/data/lokivideo --output-dir=./output/quotaloki

python3 prepare_quota040401.py --cached-data-root=/data/lokivideo --output-dir=./output/quotaloki040401


python3 lokitest2.py --loki_path=./output/quotaloki/loki.json
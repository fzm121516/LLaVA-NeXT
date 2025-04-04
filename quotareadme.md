export CUDA_HOME=/home/ubuntu/anaconda3/envs/vcsft/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTHONPATH="$PYTHONPATH:/home/ubuntu/2025/sft"

conda create -n vcsft python=3.11
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 -c constraints.txt
pip install flash-attn==2.3.6 -c constraints.txt
pip install transformers==4.40.1 -c constraints.txt
pip install timm==0.4.12 -c constraints.txt
pip install av==14.0.1 -c constraints.txt
pip install imageio==2.31.1 -c constraints.txt
pip install decord==0.6.0 -c constraints.txt
pip install opencv-python==4.8.1.78 -c constraints.txt
pip install deepspeed==0.14.2 -c constraints.txt
pip install psutil==5.9.5 -c constraints.txt
pip install accelerate==0.33.0 -c constraints.txt
pip install peft==0.5.0 -c constraints.txt
pip install tensorboardX==2.6 -c constraints.txt
pip install transformers==4.50.3 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com




CUDA_VISIBLE_DEVICES=1 python prepare_v7b_0403loki.py --cached-data-root /home/ubuntu/2025/loki --output-dir /home/ubuntu/2025/v7b_loki


CUDA_VISIBLE_DEVICES=1 python3 quota_pipeline.py






























## QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension

[![Arxiv](https://img.shields.io/badge/Arxiv-2411.13093-red)](https://arxiv.org/abs/2503.08689)
![](https://img.shields.io/badge/Task-VideoQA-blue)

<font size=7><div align='center' >[[📖 arXiv Paper](https://arxiv.org/abs/2503.08689)]  </div></font>

## 😮 Highlights


![introductionv2_01](https://github.com/user-attachments/assets/f568fcc0-bf46-41d0-a0cf-bd30745069ab)


- **We design a versatile plug-and-play pipeline for existing LVLMs:** QuoTA provides a training-free solution applicable to diverse LVLMs, enhancing long video understanding performance by assigning visual tokens based on text instruction (query) relevance. This approach offers a more elegant and direct methodology compared to conventional attention-based analytical techniques.
- **We propose CoT-driven query decouple for query-oriented frame scoring:** QuoTA employs Chain-of-Thoughts to decouple query into a specific-designed question, enabling high-quality scoring of video frames.
- **Our QuoTA setting a new state-of-the-art:** Integration of QuoTA with LLaVA-Video-7B yields a 3.2% average performance improvement across six benchmarks, achieving the best results in five video benchmarks, including Video-MME and MLVU, among 7B LVLMs.

![framework_01](https://github.com/user-attachments/assets/ff04dec6-a4d2-4032-aae5-276e5a681439)

![results](https://github.com/user-attachments/assets/179cbb51-45b9-4947-8187-15a5de00bd4d)



## 🔨 Usage

This repo is built upon LLaVA-NeXT:

- Step 1: Clone and build LLaVA-NeXT conda environment, then install the following packages in llava envs:

```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
# install qwen toolkit
pip install qwen-vl-utils
```

- Step 2: Replace the file under `LLaVA-NeXT/llava/model/llava_arch.py` with `core/llava_arch.py`: 

- Step 3: Copy the file `core/merge.py` under `LLaVA-NeXT/llava/model/`

- Step 4: Move all our code (`tools/` and `quota_pipeline.py`) under the root dir (`LLaVA-NeXT`) of LLaVA-NeXT 

- Step 5: You can now run our pipeline build upon LLaVA-Video-7B by:

```
python quota_pipeline.py
```

- Note that you can also use our pipeline for other LVLMs.


## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```
@article{luo2025quota,
  title={QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension},
  author={Luo, Yongdong and Chen, Wang and Zheng, Xiawu and Huang, Weizhong and Yin, Shukang and Lin, Haojia and Fu, Chaoyou and Huang, Jinfa and Ji, Jiayi and Luo, Jiebo and others},
  journal={arXiv preprint arXiv:2503.08689},
  year={2025}
}
```

import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoProcessor
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse
import cv2
import torch
import numpy as np
import random
import torch
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torch
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached-data-root', type=str, default='./data/dvf_recons', help='the data root for frames in the reconstruction structure')
    parser.add_argument('--output-dir', type=str, default='./outputs/mm_representations', help='the output directory')
    parser.add_argument('--output-fn', type=str, default='loki.json', help='the output file name')
    return parser.parse_args()

def get_dataset_meta(dataset_path):
    fns = sorted(os.listdir(dataset_path))
    meta = {}
    for fn in fns:
        data_id = fn.rsplit('_', maxsplit=1)[0]
        if data_id not in meta:
            meta[data_id] = 1
        else:
            meta[data_id] += 1
    return meta

def get_dataset_mp4(dataset_path):
    # 获取目录下所有文件，并过滤出 .mp4 文件
    mp4_files = [fn for fn in os.listdir(dataset_path) if fn.endswith('.mp4')]
    # 按名称排序
    mp4_files_sorted = sorted(mp4_files)
    # 返回绝对路径
    return [os.path.join(dataset_path, fn) for fn in mp4_files_sorted]

def sample_by_interval(frame_count, interval=200):
    sampled_index = []
    count = 1
    while count <= frame_count:
        sampled_index.append(count)
        count += interval
    return sampled_index

import requests
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel
import copy
from decord import VideoReader, cpu
import numpy as np
import json
import os
from tools.qwen2vl_batch import frame_retrieve
import re

def split_and_process(frames, query_relation, batch_size=16):
    results = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:min(i + batch_size, len(frames))]
        batch_result = frame_retrieve(batch, query_relation)
        results.extend(batch_result)
        torch.cuda.empty_cache()
        
    return results


def process_video_prune(video_path, max_frames_num, obj_list, query, additional_frames, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]

    # dynamic
    max_frames_num = max_frames_num + int((video_time / 3600) * additional_frames)

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    similarities_list = []

    if len(obj_list) > 0:
        query_relation = "Does the image contain any object of " + ", ".join(obj_list) + "? A. yes, B. no\nAnswer with the option's letter directly."
    else:
        query_relation = f"""
        {query}\n 
        Choose the correct option: This question is used to help determine whether the image is a natural or synthetic image. 
        Does the image contain any elements that are physically impossible or do not exist in the real world? 
        A. Yes (unrealistic elements) B. No (realistic scene only). 
        Answer with the option letter only (A or B).
        """
    similarities = split_and_process(spare_frames, query_relation, batch_size=24)
    similarities_list.append(similarities)

    print("\n------")
    print("1. spare_frames:")
    print("spare_framesshape:", spare_frames.shape)  # 形状 (N, 336, 336, 3)
    # print("spare_framesdtype:", spare_frames.dtype)  # 数据类型（通常是 uint8 或 float32）
    
    # print("\n2. frame_time ):")
    # print("frame_time", frame_time)  # 如 "0.00s,1.00s,2.00s,..."
    
    # print("\n3. video_time :")
    # print("video_time", video_time)  # 标量值（如 30.5）
    
    print("\n4. similarities_list :")
    # print("similaritieslength:", len(similarities_list))  # 列表长度（固定为1）
    print("similaritieslength:", len(similarities_list[0])) 
    print("similarities_list:", similarities_list[0])

    return spare_frames, frame_time, video_time, similarities_list


# --------------- Setting ---------------
max_frames_num = 24  # base video frame
additional_frames = 16  # maximum additional video frames
enhance_tokens = 196  # 27 * 27 = 576 -> pooling -> 14 * 14 = 196
enhance_total = 64  # total tokens = enhance_tokens * enhance_total
enhance_version = "v5"  # bilinear
weight_scale = [100, 2]

device = "cuda"
overwrite_config = {}
overwrite_config['mm_vision_tower'] = "/data/siglip-so400m-patch14-384" 
overwrite_config['prune'] = True
overwrite_config["enhance_total"] = enhance_total
overwrite_config["enhance_tokens"] = enhance_tokens
overwrite_config["enhance_version"] = enhance_version
overwrite_config['low_cpu_mem_usage'] = False
tokenizer, model, image_processor, max_length = load_pretrained_model(
    "/data/LLaVA-Video-7B-Qwen2", 
    None, 
    "llava_qwen", 
    torch_dtype="bfloat16", 
    device_map="auto", 
    overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
model.eval()
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models


print(f"---------------Frames: {max_frames_num}-----------------")
print("-----total: " + str(overwrite_config["enhance_total"]) + "----tokens: " + str(overwrite_config["enhance_tokens"]) + "----version: " + overwrite_config["enhance_version"] + "-----")

# data_path = "path/to/your/video"



@torch.inference_mode()
def infer(video_path):

    print("Video path:", video_path)  # 打印视频路径
    query = ""
    final_object=[]
    frames, frame_time, video_time, score_list = process_video_prune(video_path, max_frames_num, final_object, query, additional_frames, force_sample=True)
    
    
    q_num = 0  # 添加这行
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    weights = [(w * weight_scale[0]) ** weight_scale[1] for w in score_list[q_num]]
    video = [[video], weights]
    # qs = "You have been shown a video, which may be real or generated by an AI model. Choose one of the following options: A. The video contains anomalies suggesting it might be AI-generated. B. The video appears to be from the real world. Respond with only one letter: A or B.\n."
    # res = llava_inference(qs, video)

    qs = """
    You have been shown one video, which might be taken from real world or generated by an advanced AI model. \n
    Is this video taken in the real world? (Answer yes if you think it is taken in the real world, and answer no otherwise.)\n.
    """

    if video is not None:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    if video is not None:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # else:
        #     cont = model.generate(
        #         input_ids,
        #         images=video,
        #         modalities= ["video"],
        #         do_sample=False,
        #         temperature=0,
        #         max_new_tokens=4096,
        #     )
        
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        # return text_outputs


    print("Answer:",text_outputs)


    # return selected_layer_final,pooled_mm_feature
    return text_outputs


if __name__ == '__main__':
    config = parse_args().__dict__
    output_dir = config['output_dir']
    output_fn = config['output_fn']  # This should be your JSON output filename
    input_data_root = config['cached_data_root']
    cls_folder = sorted(os.listdir(input_data_root))
    cls_folder = list(filter(lambda x: os.path.isdir(os.path.join(input_data_root, x)), cls_folder))
    print(f'Find {len(cls_folder)} classes: {cls_folder}')
    
    # Initialize a list to store all JSON entries
    json_output = []
    
    with torch.inference_mode():
        for cls_idx, sub_cls in enumerate(cls_folder, 1):
            # Directly process files in the class directory, skipping the label level
            os.makedirs(os.path.join(output_dir, sub_cls), exist_ok=True)
            
            dataset_mp4 = get_dataset_mp4(os.path.join(input_data_root, sub_cls))
            
            for data_id in tqdm(dataset_mp4):
                try:
                    file_name = os.path.basename(data_id)
                    file, _ = os.path.splitext(file_name)
                    
                    # Perform inference
                    output1 = infer(data_id)

                    # Create JSON entry similar to your example
                    entry = {
                        "id": file,  # Using filename as ID
                        # "conversations": [
                        #     {
                        #         "from": "human",
                        #         "value": "<image>\n" + question2  # Your question text
                        #     },
                        #     {
                        #         "from": "gpt",
                        #         "value": output1   # Your answer text
                        #     }
                        # ],
                        "data_source": sub_cls,  # Now just using sub_cls without label
                        "video_path": data_id,  # Full video path or relative path
                        "answer": output1
                    }
                    
                    # Add to JSON output list
                    json_output.append(entry)
                    
                    # Save tensor data if needed
                    output_path = os.path.join(output_dir, sub_cls, f"{file}.pth")
                    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
                        result_dict = {
                            "text1": output1,
                            # "bool_value": bool_value,
                        }
                        torch.save(result_dict, output_path)
                
                except Exception as e:
                    print(f"Error processing video {data_id}: {e}")
        
            print(f'Finished {cls_idx}/{len(cls_folder)}')
    
        # Save the JSON output
        with open(os.path.join(output_dir, output_fn), 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output saved to {os.path.join(output_dir, output_fn)}")
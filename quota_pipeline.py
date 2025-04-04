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

def split_and_process(frames, query_relation, batch_size=32):
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
        query_relation = f"{query}\nDoes the image contain sufficient information to answer the given question? A. yes, B. no\nAnswer with the option's letter directly."
    similarities = split_and_process(spare_frames, query_relation, batch_size=64)
    similarities_list.append(similarities)
    
    return spare_frames, frame_time, video_time, similarities_list


# --------------- Setting ---------------
max_frames_num = 96  # base video frame
additional_frames = 64  # maximum additional video frames
enhance_tokens = 196  # 27 * 27 = 576 -> pooling -> 14 * 14 = 196
enhance_total = 64  # total tokens = enhance_tokens * enhance_total
enhance_version = "v1"  # bilinear
weight_scale = [100, 2]

device = "cuda"
overwrite_config = {}
overwrite_config['mm_vision_tower'] = "/home/ubuntu/202502/siglip-so400m-patch14-384" 
overwrite_config['prune'] = True
overwrite_config["enhance_total"] = enhance_total
overwrite_config["enhance_tokens"] = enhance_tokens
overwrite_config["enhance_version"] = enhance_version
overwrite_config['low_cpu_mem_usage'] = False
tokenizer, model, image_processor, max_length = load_pretrained_model(
    "/home/ubuntu/202502/LLaVA-Video-7B-Qwen2", 
    None, 
    "llava_qwen", 
    torch_dtype="bfloat16", 
    device_map="auto", 
    overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
model.eval()
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def llava_inference(qs, video):
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
            num_beams=1
        )
    else:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs

print(f"---------------Frames: {max_frames_num}-----------------")
print("-----total: " + str(overwrite_config["enhance_total"]) + "----tokens: " + str(overwrite_config["enhance_tokens"]) + "----version: " + overwrite_config["enhance_version"] + "-----")
video_path ="/home/ubuntu/2025/DVF_tiny/DVF_tiny/DVF_tiny/pika/1_fake/Cinematic_medium_shot_of_a_tiger_walking_in_the_jungle__soft_lighting__4k__sharp__Canon_C300__depth__seed7129231733499100_upscaled.mp4"
# data_path = "path/to/your/video"
query = "What's the man doing on the bed? A. jumping. B. sleeping. C. eating."

retrieve_pmt_0 = query 
retrieve_pmt_0 += '''\nTo answer this question, please follow the following step to give an object list for object detection.
    Step 1: Do you think it is necessary to detect specific physical entities in the video? Please only answer "yes" or "no" in this step.
    Step 2: If 'yes' in previous step, then what object in the video do you think needs to be detected to answer this question? Please only output the name of all the objects need to detect in the form of a Python list without other information.
    Step 3: Filter the object list provided in step 2, make sure they are physical entities, not abstract concepts. Please only output the name of the filtered object in the form of a Python list without other information. If all the words being filtered, please only output null.
    # Example 1:
    Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
    Your output should be:
    Step 1: yes
    Step 2: ["blue ballons", "long table", "video", "blue"]
    Step 3: ["blue ballons", "long table"]
    # Example 2:
    Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
    Your output should be:
    Step 1: yes
    Step 2: ["the man in black", "woman", "lower left corner"]
    Step 3: ["the man in black", "woman"]
    # Example 3:
    Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
    Your output should be:
    Step 1: no'''
            
out = llava_inference(retrieve_pmt_0, None)
step1_match = re.search(r'step 1: (yes|no)', out.lower())
if step1_match and step1_match.group(1) == 'yes':
    step3_match = re.search(r'step 3: (\[[^\]]*\])', out.lower())
    if step3_match:
        final_object = eval(step3_match.group(1))

frames, frame_time, video_time, score_list = process_video_prune(video_path, max_frames_num, final_object, query, additional_frames, force_sample=True)
q_num = 0  # 添加这行
video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
weights = [(w * weight_scale[0]) ** weight_scale[1] for w in score_list[q_num]]
video = [[video], weights]
qs = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: " + query + '\nThe best answer is:'
res = llava_inference(qs, video)

print(res)

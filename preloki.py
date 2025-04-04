import json
import os
import shutil
from collections import defaultdict
import argparse


def standardize_answer(answer):
    """标准化答案格式"""
    if answer is None:
        return "No"  # 假设None表示No
    answer = str(answer).strip()
    if answer.lower().startswith("yes"):
        return "Yes"
    elif answer.lower().startswith("no"):
        return "No"
    return answer  # 如果不符合上述情况，保持原样


def process_true_or_false_data(data):
    """处理true_or_false数据：去重并根据question_type强制设置answer"""
    # 1. 去重 - 保留每个video_path第一次出现的条目
    seen_paths = set()
    unique_data = []
    
    for item in data:
        if "video_path" not in item:
            continue
        path = item["video_path"]
        if path not in seen_paths:
            seen_paths.add(path)
            unique_data.append(item)
    
    # 2. 根据question_type强制设置answer
    processed_data = []
    for item in unique_data:
        if "question_type" not in item:
            processed_data.append(item)
            continue
        
        question_type = item["question_type"]
        parts = question_type.split("_")
        if len(parts) >= 6:  # 确保有足够的部分
            real_fake = parts[5]  # 第三个_后的词
            new_item = item.copy()
            if real_fake.lower() == "real":
                new_item["answer"] = "Yes"
            elif real_fake.lower() == "fake":
                new_item["answer"] = "No"
            processed_data.append(new_item)
        else:
            processed_data.append(item)
    
    return processed_data


def copy_videos_based_on_answer(data, video_source_dir, real_dest_dir, fake_dest_dir):
    """根据答案将视频文件分类复制到相应目录"""
    # 确保目标目录存在
    os.makedirs(real_dest_dir, exist_ok=True)
    os.makedirs(fake_dest_dir, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for item in data:
        if "video_path" not in item or "answer" not in item:
            skipped_count += 1
            print(f"跳过: 缺少video_path或answer字段 - {item}")
            continue
            
        video_path = item["video_path"]
        answer = standardize_answer(item["answer"])
        
        # 获取视频文件名
        video_filename = os.path.basename(video_path)
        source_path = os.path.join(video_source_dir, video_filename)
        
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            skipped_count += 1
            print(f"警告: 源视频文件不存在 - {source_path}")
            continue
        
        # 根据答案选择目标目录
        if answer == "Yes":
            dest_path = os.path.join(real_dest_dir, video_filename)
            dest_type = "真实(0_real)"
        elif answer == "No":
            dest_path = os.path.join(fake_dest_dir, video_filename)
            dest_type = "伪造(1_fake)"
        else:
            skipped_count += 1
            print(f"跳过: 未知的answer值 '{answer}' - {source_path}")
            continue
        
        # 复制文件并打印详细信息
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"复制: {source_path} -> {dest_type}: {dest_path}")
        except Exception as e:
            print(f"错误: 复制失败 {source_path} -> {dest_path}: {str(e)}")
            skipped_count += 1
    
    print(f"\n操作完成: 成功复制 {copied_count} 个文件，跳过 {skipped_count} 个文件")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="处理true_or_false数据并分类视频文件")
    parser.add_argument(
        "--true_or_false_path", 
        default="true_or_false.json",  # 默认值
        help="true_or_false JSON文件路径"
    )
    parser.add_argument(
        "--video_source_dir",
        default="/data/lokivideo/video",
        help="视频源文件目录"
    )
    parser.add_argument(
        "--real_dest_dir",
        default="/data/loki/0_real",
        help="真实视频目标目录"
    )
    parser.add_argument(
        "--fake_dest_dir",
        default="/data/loki/1_fake",
        help="伪造视频目标目录"
    )
    args = parser.parse_args()

    # 读取true_or_false.json
    try:
        with open(args.true_or_false_path, "r") as f:
            true_or_false_data = json.load(f)
        print(f"成功读取true_or_false.json，共 {len(true_or_false_data)} 条数据")
    except FileNotFoundError:
        print("错误: 找不到true_or_false.json文件")
        return
    except json.JSONDecodeError:
        print("错误: true_or_false.json文件格式不正确")
        return
    except Exception as e:
        print(f"读取true_or_false.json时发生未知错误: {str(e)}")
        return
    
    # 处理true_or_false数据
    processed_true_or_false_data = process_true_or_false_data(true_or_false_data)
    print(f"处理后true_or_false.json，共 {len(processed_true_or_false_data)} 条唯一数据\n")
    
    # 分类复制视频文件
    copy_videos_based_on_answer(
        processed_true_or_false_data,
        args.video_source_dir,
        args.real_dest_dir,
        args.fake_dest_dir
    )


if __name__ == "__main__":
    main()
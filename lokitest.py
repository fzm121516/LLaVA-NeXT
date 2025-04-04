import json
import os
import re
from collections import defaultdict

def extract_filename(video_path):
    """提取文件名和.mp4部分（不区分大小写）"""
    # 获取文件名（去掉路径）
    filename = os.path.basename(video_path)
    # 确保扩展名是.mp4（不区分大小写）
    filename = re.sub(r'\.(mp4|MP4)$', '.mp4', filename, flags=re.IGNORECASE)
    return filename

def standardize_answer(answer):
    """标准化答案格式"""
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
            real_fake = parts[3]  # 第三个_后的词
            new_item = item.copy()
            if real_fake.lower() == "real":
                new_item["answer"] = "Yes"
            elif real_fake.lower() == "fake":
                new_item["answer"] = "No"
            processed_data.append(new_item)
        else:
            processed_data.append(item)
    
    return processed_data

def compare_answers(loki_data, true_or_false_data):
    """比较两个数据集中的答案，并记录所有比较结果"""
    # 创建filename到true_or_false答案的映射
    truth_map = {}
    subset_map = {}  # 存储每个文件的子集信息
    
    for item in true_or_false_data:
        if "video_path" not in item or "answer" not in item:
            continue
        filename = extract_filename(item["video_path"])
        answer = standardize_answer(item["answer"])
        truth_map[filename] = answer
        
        # 提取子集信息
        if "question_type" in item:
            parts = item["question_type"].split("_")
            if parts:  # 确保有内容
                subset = parts[-1]  # 最后一个部分作为子集
                subset_map[filename] = subset
    
    total = 0
    correct = 0
    real_total = 0
    real_correct = 0
    fake_total = 0
    fake_correct = 0
    results = []  # 存储所有比较结果
    
    # 用于子集统计
    subset_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'real_total': 0, 'real_correct': 0, 'fake_total': 0, 'fake_correct': 0})
    
    for item in loki_data:
        if "video_path" not in item or "answer" not in item:
            continue
            
        filename = extract_filename(item["video_path"])
        loki_answer = standardize_answer(item["answer"])
        
        if filename in truth_map:
            truth_answer = truth_map[filename]
            total += 1
            is_correct = loki_answer == truth_answer
            if is_correct:
                correct += 1
            
            # 统计真实视频和合成视频的准确率
            if truth_answer == "Yes":
                real_total += 1
                if is_correct:
                    real_correct += 1
            elif truth_answer == "No":
                fake_total += 1
                if is_correct:
                    fake_correct += 1
            
            # 记录结果
            result = {
                "filename": filename,
                "loki_answer": loki_answer,
                "truth_answer": truth_answer,
                "is_correct": is_correct
            }
            
            # 如果有子集信息，添加到结果中
            if filename in subset_map:
                subset = subset_map[filename]
                result['subset'] = subset
                # 更新子集统计
                subset_stats[subset]['total'] += 1
                if is_correct:
                    subset_stats[subset]['correct'] += 1
                
                # 更新子集的真实/合成视频统计
                if truth_answer == "Yes":
                    subset_stats[subset]['real_total'] += 1
                    if is_correct:
                        subset_stats[subset]['real_correct'] += 1
                elif truth_answer == "No":
                    subset_stats[subset]['fake_total'] += 1
                    if is_correct:
                        subset_stats[subset]['fake_correct'] += 1
            
            results.append(result)
        else:
            print(f"警告: {filename} 在true_or_false.json中未找到")
    
    if total > 0:
        accuracy = correct / total * 100
        real_accuracy = real_correct / real_total * 100 if real_total > 0 else 0
        fake_accuracy = fake_correct / fake_total * 100 if fake_total > 0 else 0
        
        print(f"\n总计比较: {total}")
        print(f"正确数量: {correct}")
        print(f"总体准确率: {accuracy:.2f}%")
        print(f"\n真实视频(Yes)统计: 数量={real_total}, 正确={real_correct}, 准确率={real_accuracy:.2f}%")
        print(f"合成视频(No)统计: 数量={fake_total}, 正确={fake_correct}, 准确率={fake_accuracy:.2f}%")
        
        # 打印子集统计信息
        if subset_stats:
            print("\n子集统计:")
            for subset, stats in subset_stats.items():
                subset_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                subset_real_accuracy = (stats['real_correct'] / stats['real_total'] * 100) if stats['real_total'] > 0 else 0
                subset_fake_accuracy = (stats['fake_correct'] / stats['fake_total'] * 100) if stats['fake_total'] > 0 else 0
                
                print(f"\n子集 '{subset}':")
                print(f"  总计: 数量={stats['total']}, 正确={stats['correct']}, 准确率={subset_accuracy:.2f}%")
                print(f"  真实视频: 数量={stats['real_total']}, 正确={stats['real_correct']}, 准确率={subset_real_accuracy:.2f}%")
                print(f"  合成视频: 数量={stats['fake_total']}, 正确={stats['fake_correct']}, 准确率={subset_fake_accuracy:.2f}%")
        
        # 打印所有比较结果，按正确性分组
        print("\n所有比较结果:")
        print("\n正确的条目:")
        for result in [r for r in results if r["is_correct"]]:
            subset_info = f", 子集: {result['subset']}" if 'subset' in result else ""
            print(f"文件名: {result['filename']}{subset_info} - Loki: {result['loki_answer']}, 真实: {result['truth_answer']}")
        
        print("\n错误的条目:")
        for result in [r for r in results if not r["is_correct"]]:
            subset_info = f", 子集: {result['subset']}" if 'subset' in result else ""
            print(f"文件名: {result['filename']}{subset_info} - Loki: {result['loki_answer']}, 真实: {result['truth_answer']}")

    else:
        print("没有可比较的数据")

import argparse
import json

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="比较两个JSON文件的数据")
    parser.add_argument(
        "--loki_path", 
        default="./lokioutput/loki.json",  # 默认值
        help="loki JSON文件路径"
    )
    parser.add_argument(
        "--true_or_false_path", 
        default="true_or_false.json",  # 默认值
        help="true_or_false JSON文件路径"
    )
    args = parser.parse_args()
    # 读取loki.json
    try:
        with open(args.loki_path, "r") as f:
            loki_data = json.load(f)
        print(f"成功读取{args.loki_path}，共 {len(loki_data)} 条数据")
    except FileNotFoundError:
        print(f"错误: 找不到{args.loki_path}文件")
        return
    except json.JSONDecodeError:
        print("错误: loki.json文件格式不正确")
        return
    except Exception as e:
        print(f"读取loki.json时发生未知错误: {str(e)}")
        return
    
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
    print(f"处理后true_or_false.json，共 {len(processed_true_or_false_data)} 条唯一数据")
    
    # 比较答案
    compare_answers(loki_data, processed_true_or_false_data)

if __name__ == "__main__":
    main()
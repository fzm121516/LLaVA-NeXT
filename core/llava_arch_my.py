#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

class LlavaMetaModel:
    """LLaVA 多模态模型的元模型类，负责视觉和语言特征的融合处理"""

    def __init__(self, config):
        """初始化方法
        Args:
            config: 模型配置对象，包含视觉和语言模型的各种参数
        """
        super(LlavaMetaModel, self).__init__(config)  # 调用父类初始化

        # 如果配置中有视觉塔(vision tower)的设置，则初始化视觉相关组件
        if hasattr(config, "mm_vision_tower"):
            # 获取是否延迟加载视觉塔的配置
            delay_load = getattr(config, "delay_load", False)
            # 构建视觉特征提取塔
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            # 构建视觉特征重采样器
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            # 构建视觉特征投影器
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            # 如果配置中指定了"unpad"类型的patch合并方式，则初始化图像换行参数
            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        """获取视觉塔实例
        Returns:
            视觉塔模型实例，如果是列表则返回第一个元素
        """
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """初始化视觉模块
        Args:
            model_args: 模型参数对象，包含视觉相关的各种配置
            fsdp: 是否使用完全分片数据并行(Full Sharded Data Parallel)
        """
        # 从模型参数中提取视觉相关配置
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        # 更新模型配置
        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        # 如果视觉塔尚未初始化，则构建新的视觉组件
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            
            # 将重采样器的配置更新到模型配置中
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            # 根据是否使用FSDP决定如何存储视觉组件
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            # 如果视觉塔已存在，则加载预训练模型
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()  # 加载预训练权重

            # 确保重采样器的参数是可训练的(防止被LoRA冻结)
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        # 更新模型配置中的视觉相关参数
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # 如果配置中没有'add_faster_video'属性且模型参数要求添加快速视频处理
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                # 初始化快速视频处理的token参数
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        # 如果投影器尚未初始化，则构建新的投影器
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            # 如果使用"unpad"合并方式，初始化图像换行参数
            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # 确保投影器的参数是可训练的(防止被LoRA冻结)
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # 如果提供了预训练的多模态MLP适配器路径，则加载预训练权重
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            # 辅助函数：从权重字典中提取特定模块的权重
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            # 加载投影器权重
            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
            # 加载视觉重采样器权重(非严格模式，允许部分不匹配)
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    对经过填充和调整大小的PyTorch张量图像进行去填充操作

    Args:
    tensor (torch.Tensor): 图像张量，假设为CxHxW格式（通道x高度x宽度）
    original_size (tuple): 图像的原始尺寸（宽度, 高度）

    Returns:
    torch.Tensor: 去填充后的图像张量
    """
    # 获取原始图像的宽度和高度
    original_width, original_height = original_size
    # 获取当前张量的高度和宽度（跳过通道维度）
    current_height, current_width = tensor.shape[1:]

    # 计算原始图像的宽高比
    original_aspect_ratio = original_width / original_height
    # 计算当前张量的宽高比
    current_aspect_ratio = current_width / current_height

    # 根据宽高比判断填充方向并计算去填充区域
    if original_aspect_ratio > current_aspect_ratio:
        # 如果原始宽高比大于当前宽高比，说明填充是加在高度上的
        # 计算宽度缩放比例
        scale_factor = current_width / original_width
        # 计算原始高度按比例缩放后的新高度
        new_height = int(original_height * scale_factor)
        # 计算上下两侧的填充量（取整）
        padding = (current_height - new_height) // 2
        # 在高度维度上裁剪掉填充部分（保留中间有效区域）
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # 否则说明填充是加在宽度上的
        # 计算高度缩放比例
        scale_factor = current_height / original_height
        # 计算原始宽度按比例缩放后的新宽度
        new_width = int(original_width * scale_factor)
        # 计算左右两侧的填充量（取整）
        padding = (current_width - new_width) // 2
        # 在宽度维度上裁剪掉填充部分（保留中间有效区域）
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    # 返回去填充后的张量
    return unpadded_tensor



class LlavaMetaForCausalLM(ABC):
    """用于多模态因果语言模型的抽象基类，处理视觉和语言特征的融合"""

    @abstractmethod
    def get_model(self):
        """抽象方法：必须由子类实现，返回主模型实例"""
        pass

    def get_vision_tower(self):
        """获取视觉编码器（视觉塔）"""
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        """
        对图像特征进行2D池化/下采样
        Args:
            image_feature: 输入特征 [num_frames, num_tokens, num_dim]
            stride: 池化步长（下采样因子）
        Returns:
            处理后的特征 [num_frames, new_num_tokens, num_dim]
        """
        # 获取原始特征图的高度和宽度（假设是正方形）
        height = width = self.get_vision_tower().num_patches_per_side
        
        # 解构输入特征的形状
        num_frames, num_tokens, num_dim = image_feature.shape
        
        # 将特征重塑为4D张量 [帧数, 高, 宽, 通道]
        image_feature = image_feature.view(num_frames, height, width, -1)
        
        # 调整维度顺序为PyTorch卷积格式 [帧数, 通道, 高, 宽]
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

        # 根据配置选择不同的空间池化方式
        if self.config.mm_spatial_pool_mode == "average":
            # 平均池化
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            # 最大池化
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            # 双线性插值
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"不支持的mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")

        # 恢复原始维度顺序 [帧数, 高, 宽, 通道]
        image_feature = image_feature.permute(0, 2, 3, 1)
        
        # 展平空间维度 [帧数, 新token数, 特征维度]
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        """编码单张图像特征"""
        # 通过视觉塔获取原始特征
        image_features = self.get_model().get_vision_tower()(images)
        
        # 通过投影器调整特征维度
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        """
        编码多模态输入（视频/图像）
        Args:
            videos_or_images: 输入的多媒体数据
            video_idx_in_batch: 批次中视频样本的索引列表
            split_sizes: 分割大小（用于处理不同长度的视频）
        Returns:
            tuple: (处理后的特征列表, 快速视频特征列表)
        """
        # 获取原始视觉特征
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        
        # 按分割大小拆分特征（处理不同长度的视频）
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)
        
        all_videos_or_images_features = []  # 存储所有处理后的特征
        all_faster_video_features = []      # 存储快速视频特征（如果启用）
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride  # 当前下采样步长

        for idx, feat in enumerate(per_videos_or_images_features):
            # 通过投影器调整特征维度
            feat = self.get_model().mm_projector(feat)
            
            faster_video_feature = 0  # 初始化快速视频特征
            slower_img_feat = 0       # 初始化慢速/正常特征
            
            # 如果是视频且需要下采样
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                # 生成正常速度的特征（第一次下采样）
                slower_img_feat = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
                
                # 如果启用快速视频特征
                if self.config.add_faster_video:
                    # 增大下采样步长
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    # 生成快速视频特征（更大步长的下采样）
                    faster_video_feature = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
            
            # 保存处理后的特征
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            
            all_faster_video_features.append(faster_video_feature)
        
        return all_videos_or_images_features, all_faster_video_features

    def add_token_per_grid(self, image_feature):
        """
        按网格添加特殊token（如图像换行符）
        Args:
            image_feature: 输入特征 [num_frames, num_tokens, feature_dim]
        Returns:
            处理后的特征 [新num_tokens, num_frames, feature_dim+1]
        """
        # 计算特征图的高度（假设是正方形）
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        # 重塑特征为5D张量 [帧数, 1, 高, 宽, 特征]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        
        # 调整维度顺序 [特征, 帧数, 高, 1, 宽]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        
        # 展平部分维度 [特征, 帧数*高, 1*宽]
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        
        # 添加换行token [特征, 帧数*高*宽, 1]
        image_feature = torch.cat(
            (image_feature, 
             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), 
            dim=-1
        )

        # 如果启用快速视频处理
        if getattr(self.config, "add_faster_video", False):
            # 重塑为4D [特征, 帧数, 高, 宽+1]
            image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
            
            # 调整维度顺序 [帧数, 高, 宽+1, 特征]
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            
            # 展平空间维度 [帧数, 高*(宽+1), 特征]
            image_feature = image_feature.flatten(1, 2)
            return image_feature
        
        # 默认处理：展平并转置 [帧数*高*宽, 特征]
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        """
        按帧添加特殊token
        Args:
            image_feature: 输入特征 [num_frames, num_tokens, feature_dim]
        Returns:
            处理后的特征 [num_frames, num_tokens, feature_dim+1]
        """
        # 调整维度顺序 [feature_dim, num_frames, num_tokens]
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        
        # 添加换行token
        image_feature = torch.cat(
            (image_feature, 
             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), 
            dim=-1
        )
        
        # 恢复原始维度顺序
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        # 获取视觉编码器（vision tower）
        vision_tower = self.get_vision_tower()
        # 如果没有视觉编码器或没有图像输入或输入序列长度为1，则直接返回原始输入
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 如果modalities是字符串，转换为列表
        if isinstance(modalities, str):
            modalities = ["image"]

        # 处理图像/视频输入（可能是列表或5维张量）
        if type(images) is list or images.ndim == 5:
            # 如果是列表，确保每个元素是4维张量（批量维度）
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            # 记录视频模态在批次中的索引
            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            # 将所有图像拼接成一个张量
            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            
            # 编码图像特征
            encoded_image_features = self.encode_images(concat_images)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            
            # 对视频模态应用2D池化
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            # 获取配置参数
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            # 根据不同的patch合并类型处理图像特征
            if mm_patch_merge_type == "flat":
                # 展平处理
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # 视频模态处理
                    if image_idx in video_idx_in_batch:
                        if mm_newline_position == "grid":
                            # 网格级别添加token
                            image_feature = self.add_token_per_grid(image_feature)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # 帧级别添加token
                            image_feature = self.add_token_per_frame(image_feature)
                            new_image_features.append(image_feature.flatten(0, 1))
                        elif mm_newline_position == "one_token":
                            # 单token处理
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    # 多图像处理
                    elif image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        # 处理不同宽高比的图像
                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            vision_tower_image_size = self.get_vision_tower().image_size
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        # 不同的特征合并策略
                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            # 处理未填充的图像
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        
                        # 是否包含基础特征
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    # 单图像处理
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # 单图像编码
            image_features = self.encode_images(images)

        # 处理标签、位置ID和注意力掩码
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 根据注意力掩码去除填充部分
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 构建新的输入嵌入和标签
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        # 遍历每个样本
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # 没有图像token的情况
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 分割图像token和非图像部分
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            
            # 嵌入非图像部分的token
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 构建最终的嵌入和标签序列
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 截断序列到最大长度
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # 填充序列到统一长度
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 根据填充方向（左或右）进行填充
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 处理最终输出
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
            
        # 位置跳跃训练（如果启用）
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        初始化视觉相关的特殊token和嵌入层
        
        参数:
            model_args: 包含模型配置参数的对象
            tokenizer: 文本tokenizer对象
        """
        
        # 1. 处理图像patch token
        if model_args.mm_use_im_patch_token:
            # 添加默认的图像patch token到tokenizer中(作为特殊token)
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            # 调整token嵌入层的大小以匹配新的tokenizer词汇表大小
            self.resize_token_embeddings(len(tokenizer))

        # 2. 处理图像开始/结束token
        if model_args.mm_use_im_start_end:
            # 添加图像开始和结束token到tokenizer中
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], 
                special_tokens=True
            )
            # 再次调整嵌入层大小
            self.resize_token_embeddings(len(tokenizer))

            # 如果有新添加的token
            if num_new_tokens > 0:
                # 获取输入和输出嵌入层的权重数据
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                # 计算现有token嵌入的平均值(用于初始化新token)
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                # 用平均值初始化新token的嵌入
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 3. 微调MLP适配器相关设置
            if model_args.tune_mm_mlp_adapter:
                # 设置输入嵌入层参数需要梯度(可训练)
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # 设置输出嵌入层参数不需要梯度(冻结)
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # 4. 加载预训练的MLP适配器权重
            if model_args.pretrain_mm_mlp_adapter:
                # 从文件加载预训练权重
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                
                # 确保新添加的token数量是2(开始和结束token)
                assert num_new_tokens == 2
                
                # 处理不同形状的预训练权重
                if input_embeddings.shape == embed_tokens_weight.shape:
                    # 直接复制最后两个token的权重
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    # 预训练权重只有新token的嵌入
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    # 形状不匹配时抛出错误
                    raise ValueError(
                        f"预训练权重形状不匹配。预训练: {embed_tokens_weight.shape}。"
                        f"当前: {input_embeddings.shape}。新token数量: {num_new_tokens}。"
                    )
        
        # 5. 仅使用图像patch token时的处理
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                # 冻结输入和输出嵌入层的参数
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False



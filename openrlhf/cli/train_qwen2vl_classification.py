#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import time
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from transformers import get_scheduler, set_seed
from sklearn.metrics import classification_report, confusion_matrix

from openrlhf.datasets import MultimodalClassificationDataset
from openrlhf.models import load_qwen2vl_for_classification
from openrlhf.trainer import MultimodalClassificationTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

logger = logging.getLogger(__name__)

def setup_logging(args):
    """设置日志记录"""
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    
    # 创建文件处理器
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

def validate_args(args):
    """验证命令行参数"""
    if not args.pretrain:
        raise ValueError("必须提供预训练模型路径 (--pretrain)")
    
    if not args.train_data:
        raise ValueError("必须提供训练数据路径 (--train_data)")
    
    if args.fp16 and args.bf16:
        raise ValueError("不能同时启用 fp16 和 bf16")
    
    if args.lora_rank > 0 and args.target_modules == "all-linear" and args.vision_tower_lora:
        logger.warning("同时对所有线性层和视觉塔应用LoRA可能导致显存不足")
    
    if args.train_batch_size < args.micro_train_batch_size:
        logger.warning(f"train_batch_size ({args.train_batch_size}) 小于 micro_train_batch_size ({args.micro_train_batch_size})")
        args.train_batch_size = args.micro_train_batch_size
        logger.warning(f"已将 train_batch_size 设置为 {args.train_batch_size}")

def train(args):
    """训练Qwen2VL分类模型"""
    # 设置日志
    setup_logging(args)
    
    # 验证参数
    validate_args(args)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"已设置随机种子: {args.seed}")
    
    # 配置分布式训练策略
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    logger.info(f"使用设备: {strategy.device}")
    logger.info(f"分布式训练: {dist.is_initialized()}")
    if dist.is_initialized():
        logger.info(f"世界大小: {dist.get_world_size()}, 本地排名: {dist.get_rank()}")
    
    # 配置模型
    start_time = time.time()
    logger.info(f"正在加载模型: {args.pretrain}")
    
    try:
        model, tokenizer = load_qwen2vl_for_classification(
            args.pretrain,
            num_classes=args.num_classes,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            vision_tower_lora=args.vision_tower_lora,
        )
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    
    if args.verbose:
        strategy.print(model)
    
    # 配置数据集
    logger.info("正在准备训练数据集...")
    try:
        train_data = blending_datasets(
            args.train_data,
            args.train_data_probs,
            strategy,
            return_eval=False,
            train_split=args.train_split,
        )
        logger.info(f"原始训练数据集大小: {len(train_data)}")
        
        # 限制样本数量
        max_samples = min(args.max_samples, len(train_data))
        train_data = train_data.select(range(max_samples))
        logger.info(f"使用的训练样本数: {len(train_data)}")
        
        # 创建训练数据集
        train_dataset = MultimodalClassificationDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            image_key=args.image_key,
            text_key=args.text_key,
            label_key=args.label_key,
            image_folder=args.image_folder,
            image_size=args.image_size,
            use_augmentation=args.use_augmentation,
        )
        logger.info("训练数据集准备完成")
    except Exception as e:
        logger.error(f"训练数据集准备失败: {e}")
        raise
    
    # 准备评估数据集
    if args.eval_data:
        logger.info("正在准备评估数据集...")
        try:
            eval_data = blending_datasets(
                args.eval_data,
                args.eval_data_probs,
                strategy,
                return_eval=True,
                eval_split=args.eval_split,
            )
            logger.info(f"原始评估数据集大小: {len(eval_data)}")
            
            # 限制样本数量
            max_eval_samples = min(args.max_eval_samples, len(eval_data))
            eval_data = eval_data.select(range(max_eval_samples))
            logger.info(f"使用的评估样本数: {len(eval_data)}")
            
            # 创建评估数据集
            eval_dataset = MultimodalClassificationDataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                image_key=args.image_key,
                text_key=args.text_key,
                label_key=args.label_key,
                image_folder=args.image_folder,
                image_size=args.image_size,
                use_augmentation=False,  # 评估时不使用数据增强
            )
            logger.info("评估数据集准备完成")
        except Exception as e:
            logger.error(f"评估数据集准备失败: {e}")
            raise
    else:
        eval_dataset = None
        logger.info("未提供评估数据集")
    
    # 配置数据加载器
    logger.info("正在准备数据加载器...")
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    logger.info(f"训练数据加载器批次数: {len(train_dataloader)}")
    
    if eval_dataset:
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_eval_batch_size or args.micro_train_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )
        logger.info(f"评估数据加载器批次数: {len(eval_dataloader)}")
    else:
        eval_dataloader = None
    
    # 计算训练步数
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    logger.info(f"每轮更新步数: {num_update_steps_per_epoch}, 总步数: {max_steps}")
    
    # 配置优化器
    logger.info("正在配置优化器...")
    try:
        optim = strategy.create_optimizer(
            model,
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
            fused=args.fused,
        )
        
        # 配置学习率调度器
        warmup_steps = math.ceil(max_steps * args.lr_warmup_ratio)
        logger.info(f"学习率预热步数: {warmup_steps}")
        
        scheduler = get_scheduler(
            args.lr_scheduler,
            optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        logger.info(f"使用学习率调度器: {args.lr_scheduler}")
    except Exception as e:
        logger.error(f"优化器配置失败: {e}")
        raise
    
    # 准备模型/优化器
    model, optim, scheduler = strategy.prepare((model, optim, scheduler))
    
    # 配置训练器
    logger.info("正在初始化训练器...")
    try:
        trainer = MultimodalClassificationTrainer(
            model=model,
            strategy=strategy,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=args.max_norm,
            max_steps=max_steps,
            num_epochs=args.max_epochs,
            tokenizer=tokenizer,
            ckpt_path=args.ckpt_path,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            aux_loss_coef=args.aux_loss_coef,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            label_names=args.label_names,
            num_classes=args.num_classes,
        )
    except Exception as e:
        logger.error(f"训练器初始化失败: {e}")
        raise
    
    # 开始训练
    logger.info("开始训练...")
    consumed_samples = 0
    
    try:
        trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
        logger.info("训练完成")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise
    
    # 保存模型
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"正在保存模型到 {args.save_path}")
        try:
            os.makedirs(args.save_path, exist_ok=True)
            strategy.save_model(model, tokenizer, args.save_path)
            logger.info("模型保存完成")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    # 最终评估
    if eval_dataloader and (not dist.is_initialized() or dist.get_rank() == 0):
        logger.info("进行最终评估...")
        try:
            eval_results = trainer.evaluate(detailed=True)
            logger.info(f"最终评估结果: {eval_results}")
            
            # 保存评估报告
            if args.save_eval_results:
                report_path = os.path.join(args.save_path, "evaluation_report.txt")
                with open(report_path, "w") as f:
                    f.write(f"分类报告:\n{eval_results['classification_report']}\n\n")
                    f.write(f"混淆矩阵:\n{eval_results['confusion_matrix']}\n")
                logger.info(f"评估报告已保存到 {report_path}")
        except Exception as e:
            logger.error(f"最终评估失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练Qwen2VL多模态分类模型")
    
    # 检查点相关
    parser.add_argument("--save_path", type=str, default="./ckpt", help="模型保存路径")
    parser.add_argument("--save_steps", type=int, default=-1, help="每多少步保存一次检查点，-1表示不保存")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False, help="是否保存HuggingFace格式的检查点")
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False, help="是否禁用DeepSpeed检查点")
    parser.add_argument("--logging_steps", type=int, default=1, help="每多少步记录一次日志")
    parser.add_argument("--eval_steps", type=int, default=-1, help="每多少步评估一次，-1表示每轮结束评估")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_qwen2vl", help="检查点保存路径")
    parser.add_argument("--save_eval_results", action="store_true", default=False, help="是否保存详细评估结果")

    # DeepSpeed相关
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO优化阶段")
    parser.add_argument("--bf16", action="store_true", default=False, help="是否使用bfloat16精度")
    parser.add_argument("--fp16", action="store_true", default=False, help="是否使用float16精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="是否使用梯度检查点")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False, help="是否使用可重入梯度检查点")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False, help="是否禁用快速分词器")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="是否启用FlashAttention2")

    # LoRA相关
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="是否使用4位量化加载模型")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA秩，0表示不使用LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout概率")
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear", help="LoRA目标模块")
    parser.add_argument("--vision_tower_lora", action="store_true", default=False, help="是否对视觉塔应用LoRA")

    # Qwen2VL分类训练
    parser.add_argument("--pretrain", type=str, required=True, help="预训练模型路径或名称")
    parser.add_argument("--num_classes", type=int, default=2, help="分类类别数")
    parser.add_argument("--label_names", type=str, nargs="*", default=None, help="标签名称列表")
    parser.add_argument("--max_epochs", type=int, default=2, help="最大训练轮数")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="辅助损失系数，用于MoE平衡损失")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03, help="学习率预热比例")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr", help="学习率调度器类型")
    parser.add_argument("--l2", type=float, default=0, help="权重衰减系数")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Adam优化器beta参数")
    parser.add_argument("--max_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--fused", action="store_true", default=False, help="是否使用融合优化器")
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="每个GPU的训练批次大小")
    parser.add_argument("--micro_eval_batch_size", type=int, default=None, help="每个GPU的评估批次大小，默认与训练相同")
    parser.add_argument("--train_batch_size", type=int, default=128, help="全局训练批次大小")

    # 数据集相关
    parser.add_argument("--train_data", type=str, required=True, help="训练数据集名称或路径")
    parser.add_argument("--train_data_probs", type=str, default="1.0", help="多数据集采样概率")
    parser.add_argument("--eval_data", type=str, default=None, help="评估数据集名称或路径")
    parser.add_argument("--eval_data_probs", type=str, default="1.0", help="多评估数据集采样概率")
    parser.add_argument("--train_split", type=str, default="train", help="训练数据集分割")
    parser.add_argument("--eval_split", type=str, default="test", help="评估数据集分割")
    parser.add_argument("--image_key", type=str, default="image", help="图像路径的JSON键")
    parser.add_argument("--text_key", type=str, default="text", help="文本的JSON键")
    parser.add_argument("--label_key", type=str, default="label", help="标签的JSON键")
    parser.add_argument("--image_folder", type=str, default=None, help="图像文件夹路径")
    parser.add_argument("--image_size", type=int, default=448, help="图像大小")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="最大训练样本数")
    parser.add_argument("--max_eval_samples", type=int, default=int(1e8), help="最大评估样本数")
    parser.add_argument("--max_len", type=int, default=512, help="最大标记长度")
    parser.add_argument("--use_augmentation", action="store_true", default=False, help="是否使用数据增强")

    # wandb参数
    parser.add_argument("--use_wandb", type=str, default=None, help="是否使用wandb")
    parser.add_argument("--wandb_org", type=str, default=None, help="wandb组织")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb组")
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_qwen2vl", help="wandb项目名称")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="qwen2vl_%s" % datetime.now().strftime("%m%dT%H:%M"),
        help="wandb运行名称",
    )

    # TensorBoard参数
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard日志路径")

    # 其他参数
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", default=False, help="是否输出详细日志")
    parser.add_argument("--log_file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--use_ms", action="store_true", default=False, help="是否使用ModelScope")

    args = parser.parse_args()

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # 修补hub以从modelscope下载模型以加速
        patch_hub()

    train(args)
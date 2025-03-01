#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import logging
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from openrlhf.models import load_qwen2vl_for_classification

logger = logging.getLogger(__name__)

def setup_logging(args):
    """Set up logging"""
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

def load_data(args):
    """Load data"""
    data = []
    
    if args.input_file.endswith(".json"):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.input_file.endswith(".jsonl"):
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append({args.text_key: line.strip()})
    
    logger.info(f"Loaded {len(data)} entries")
    
    if args.max_samples < len(data):
        data = data[:args.max_samples]
        logger.info(f"Limited to {len(data)} entries")
    
    return data

def process_batch(model, processor, tokenizer, batch, args):
    """Process a batch of data"""
    texts = [item[args.text_key] for item in batch]
    image_paths = [os.path.join(args.image_folder, item[args.image_key]) if args.image_folder else item[args.image_key] for item in batch]
    
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except Exception as e:
            logger.error(f"Cannot load image {image_path}: {e}")
            images.append(Image.new("RGB", (args.image_size, args.image_size), color=(128, 128, 128)))
    
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    encoding = tokenizer(
        texts,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding.input_ids.to(args.device)
    attention_mask = encoding.attention_mask.to(args.device)
    pixel_values = pixel_values.to(args.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    
    preds = preds.cpu().numpy().tolist()
    probs = probs.cpu().numpy().tolist()
    
    return preds, probs

def inference(args):
    """Perform inference"""
    setup_logging(args)
    
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    logger.info(f"Loading model: {args.model_path}")
    try:
        model, tokenizer = load_qwen2vl_for_classification(
            args.model_path,
            num_classes=args.num_classes,
            load_in_4bit=args.load_in_4bit,
        )
        model = model.to(args.device)
        model.eval()
        
        processor = model.processor
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise
    
    data = load_data(args)
    
    results = []
    
    for i in tqdm(range(0, len(data), args.batch_size), desc="Inference"):
        batch = data[i:i+args.batch_size]
        
        try:
            preds, probs = process_batch(model, processor, tokenizer, batch, args)
            
            for j, (pred, prob) in enumerate(zip(preds, probs)):
                item = batch[j].copy()
                item["predicted_label"] = pred
                
                if args.label_names:
                    item["predicted_label_name"] = args.label_names[pred]
                
                item["probabilities"] = prob
                item["confidence"] = prob[pred]
                
                results.append(item)
        except Exception as e:
            logger.error(f"Batch {i} processing failed: {e}")
            for j in range(len(batch)):
                item = batch[j].copy()
                item["error"] = str(e)
                results.append(item)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    if args.output_file.endswith(".json"):
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif args.output_file.endswith(".jsonl"):
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        output_file = args.output_file + ".jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Results saved to {args.output_file}")
    
    if args.label_key in data[0]:
        correct = 0
        total = 0
        
        for item in results:
            if "error" not in item and args.label_key in item:
                total += 1
                if item["predicted_label"] == item[args.label_key]:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        if args.label_names:
            class_stats = {label: {"correct": 0, "total": 0} for label in range(args.num_classes)}
            
            for item in results:
                if "error" not in item and args.label_key in item:
                    true_label = item[args.label_key]
                    pred_label = item["predicted_label"]
                    
                    class_stats[true_label]["total"] += 1
                    if pred_label == true_label:
                        class_stats[true_label]["correct"] += 1
            
            logger.info("Accuracy for each class:")
            for label, stats in class_stats.items():
                label_name = args.label_names[label] if label < len(args.label_names) else str(label)
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                logger.info(f"  {label_name}: {acc:.4f} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Qwen2VL classification model for inference")
    
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--label_names", type=str, nargs="*", default=None, help="List of label names")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load model in 4-bit quantization")
    
    parser.add_argument("--input_file", type=str, required=True, help="Input file path (JSON or JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--image_folder", type=str, default=None, help="Image folder path")
    parser.add_argument("--image_key", type=str, default="image", help="JSON key for image path")
    parser.add_argument("--text_key", type=str, default="text", help="JSON key for text")
    parser.add_argument("--label_key", type=str, default="label", help="JSON key for label")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="Maximum number of samples")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum token length")
    parser.add_argument("--image_size", type=int, default=448, help="Image size")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', or 'auto')")
    
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    
    args = parser.parse_args()
    
    inference(args) 
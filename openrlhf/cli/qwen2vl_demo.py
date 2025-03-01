#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import gradio as gr
import torch
from PIL import Image

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
    
    # Create file handler
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

def load_model(args):
    """Load the model"""
    logger.info(f"Loading model: {args.model_path}")
    try:
        model, tokenizer = load_qwen2vl_for_classification(
            args.model_path,
            num_classes=args.num_classes,
            load_in_4bit=args.load_in_4bit,
        )
        model = model.to(args.device)
        model.eval()
        
        # Get the processor
        processor = model.processor
        
        logger.info("Model loaded successfully")
        return model, tokenizer, processor
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

def predict(model, tokenizer, processor, image, text, device, label_names):
    """Make predictions"""
    # Process image
    if image is None:
        return "Please upload an image", None, None
    
    # Process text
    if not text:
        return "Please enter text", None, None
    
    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Process text
    encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Move to device
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
    
    # Get predictions and probabilities
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    # Get label name
    if label_names and pred < len(label_names):
        pred_label = label_names[pred]
    else:
        pred_label = str(pred)
    
    # Format probabilities
    formatted_probs = {}
    for i, p in enumerate(probs):
        label = label_names[i] if label_names and i < len(label_names) else str(i)
        formatted_probs[label] = float(p)
    
    # Create result string
    result = f"Predicted class: {pred_label} (Confidence: {probs[pred]:.4f})"
    
    return result, pred_label, formatted_probs

def create_demo(args):
    """Create Gradio demo"""
    # Load model
    model, tokenizer, processor = load_model(args)
    
    # Create Gradio interface
    with gr.Blocks(title="Qwen2VL Classification Demo") as demo:
        gr.Markdown("# Qwen2VL Multimodal Classification Demo")
        gr.Markdown(f"Model: {args.model_path}")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                text_input = gr.Textbox(lines=3, label="Input Text")
                submit_button = gr.Button("Submit")
            
            with gr.Column():
                result_output = gr.Textbox(label="Prediction Result")
                label_output = gr.Textbox(label="Predicted Label")
                prob_output = gr.JSON(label="Class Probabilities")
        
        # Set submit button event
        submit_button.click(
            fn=lambda img, txt: predict(model, tokenizer, processor, img, txt, args.device, args.label_names),
            inputs=[image_input, text_input],
            outputs=[result_output, label_output, prob_output],
        )
        
        # Add examples
        if args.examples_folder:
            examples = []
            try:
                import glob
                import random
                import json
                
                # Find example images
                image_files = glob.glob(os.path.join(args.examples_folder, "*.jpg")) + \
                              glob.glob(os.path.join(args.examples_folder, "*.jpeg")) + \
                              glob.glob(os.path.join(args.examples_folder, "*.png"))
                
                # Find example texts
                text_file = os.path.join(args.examples_folder, "examples.json")
                if os.path.exists(text_file):
                    with open(text_file, "r", encoding="utf-8") as f:
                        texts = json.load(f)
                else:
                    texts = ["What is this?", "Describe this image", "What does this image show?"]
                
                # Create examples
                for image_file in image_files[:5]:  # Up to 5 examples
                    text = random.choice(texts)
                    examples.append([image_file, text])
                
                gr.Examples(
                    examples=examples,
                    inputs=[image_input, text_input],
                )
            except Exception as e:
                logger.error(f"Loading examples failed: {e}")
    
    return demo

def run_demo(args):
    """Run Gradio demo"""
    # Set up logging
    setup_logging(args)
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    # Create demo
    demo = create_demo(args)
    
    # Launch demo
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2VL Classification Model Demo")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--label_names", type=str, nargs="*", default=None, help="List of label names")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load model in 4-bit quantization")
    
    # Gradio parameters
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", default=False, help="Create a public link")
    parser.add_argument("--examples_folder", type=str, default=None, help="Path to examples folder")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', or 'auto')")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    
    args = parser.parse_args()
    
    run_demo(args) 
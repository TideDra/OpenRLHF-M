import argparse
import torch
from torch import distributed as dist
from tqdm import tqdm

from openllama2.models import Actor
from openllama2.utils import get_strategy, get_tokenizer

# from openllama2.models.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()


def generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    from_config = bool(args.load_model)
    model = Actor(args.pretrain, from_config, use_flash_attention_2=args.flash_attn)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    # load Pytorch model
    if args.load_model:
        strategy.print("Load model: ", args.load_model)
        strategy.load_model(model, args.load_model)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    if args.ta_prompt:
        with open(args.ta_prompt, "r") as f:
            user_prompt = f.read()
    else:
        user_prompt = ""

    while True:
        inputs = input("Please enter a prompt (or type 'exit' to quit): ")
        if inputs.strip().lower() == "exit":
            print("Exiting program...")
            break
        if inputs.strip().lower() == "clear":
            user_prompt = ""
            continue

        # get input prompt
        user_prompt = user_prompt + "\nHuman: " + inputs + "\nAssistant: "
        if args.dt_prompt:
            user_prompt += args.dt_prompt

        user_prompt_len = len(user_prompt)

        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(torch.cuda.current_device())
        outputs = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_length=args.max_len,
            do_sample=True,
            top_p=0.9,
            early_stopping=True,
            num_beams=1,
            temperature=0.5,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        user_prompt = output[0]
        output = output[0][user_prompt_len:].replace(r"\n", "\n")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument(
        "--dt_prompt", type=str, default=None, help="decision transformer prompt, such as <rm_score>: 5.00 "
    )
    parser.add_argument("--flash_attn", action="store_true", default=False)
    args = parser.parse_args()
    generate(args)

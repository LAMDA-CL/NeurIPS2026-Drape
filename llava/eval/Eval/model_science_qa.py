import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
from llava.mm_utils import get_model_name_from_path

from PIL import Image
import math


TASK_ORDER = [
    "ScienceQA", "TextVQA", "ImageNet", "GQA",
    "VizWiz", "Grounding", "VQAv2", "OCRVQA"
]


def resolve_task_name(task_id, task_name):
    if task_name:
        return task_name
    if task_id is None:
        return None
    if 1 <= task_id <= len(TASK_ORDER):
        return TASK_ORDER[task_id - 1]
    return None


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
def load_anchors_to_model(model, anchor_path):
    if not os.path.exists(anchor_path):
        print(f"⚠️ Warning: {anchor_path} not found!")
        return
    
    state_dict = torch.load(anchor_path, map_location='cpu')
    print(f"✅ Loading anchors from {anchor_path}")
    
    # 键名前缀是 "base_model.model."
    prefix = "base_model.model."
    
    # 定义要加载的参数组
    param_groups = ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']
    
    for group in param_groups:
        if hasattr(model, group):
            param_list = getattr(model, group)
            if isinstance(param_list, torch.nn.ParameterList):
                loaded_count = 0
                for i in range(len(param_list)):
                    key = f"{prefix}{group}.{i}"
                    if key in state_dict:
                        param_list[i].data.copy_(state_dict[key].to(param_list[i].device))
                        loaded_count += 1
                    else:
                        print(f"  ⚠️ Key not found: {key}")
                print(f"  -> Loaded {loaded_count} parameters into {group}")
            else:
                print(f"  -> {group} is not a ParameterList, skipping")
        else:
            print(f"  -> Model has no attribute {group}")


import torch.distributed as dist
def sync_anchors_across_gpus(model):
    """在多 GPU 环境下同步 anchors 到所有进程"""
    if not (dist.is_available() and dist.is_initialized()):
        return
    
    # 同步 image_anchors
    if hasattr(model, 'image_anchors') and isinstance(model.image_anchors, torch.nn.ParameterList):
        for param in model.image_anchors:
            dist.broadcast(param.data, src=0)
    
    # 同步 text_anchors
    if hasattr(model, 'text_anchors') and isinstance(model.text_anchors, torch.nn.ParameterList):
        for param in model.text_anchors:
            dist.broadcast(param.data, src=0)
    
    # 同步 boundaries
    if hasattr(model, 'image_boundary') and isinstance(model.image_boundary, torch.nn.ParameterList):
        for param in model.image_boundary:
            dist.broadcast(param.data, src=0)
            
    if hasattr(model, 'text_boundary') and isinstance(model.text_boundary, torch.nn.ParameterList):
        for param in model.text_boundary:
            dist.broadcast(param.data, src=0)
    
    print("✅ Anchors synchronized across all GPUs")

from torch import nn
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # Only task_id triggers forced routing (oracle mode).
    forced_task_name = resolve_task_name(args.task_id, None)
    if forced_task_name and hasattr(model, "set_forced_task_name"):
        model.set_forced_task_name(forced_task_name)

    use_router = (args.use_router or forced_task_name is not None) and hasattr(model, "generate_with_routing")
    if (args.use_router or forced_task_name is not None) and not use_router:
        print("[Warning] --use-router set but model has no generate_with_routing; fallback to generate().")


    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    print(f"Evaluating {len(questions)} questions, saving to {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        question = line['text']
        qs = question.replace('<image>', '').strip()
        route_prompt = qs
        cur_prompt = qs

        if 'image' in line.keys():
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            gen_kwargs = dict(
                input_ids=input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
            if use_router:
                output_ids = model.generate_with_routing(texts=[route_prompt], **gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                gen_kwargs = dict(
                    input_ids=input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                if use_router:
                    output_ids = model.generate_with_routing(texts=[route_prompt], **gen_kwargs)
                else:
                    output_ids = model.generate(**gen_kwargs)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()


    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-text-select-layer", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--use-router", action="store_true", help="Use CLIP router for DRAPE")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Oracle task id (1-based): 1=ScienceQA ... 8=OCRVQA")
    args = parser.parse_args()

    eval_model(args)

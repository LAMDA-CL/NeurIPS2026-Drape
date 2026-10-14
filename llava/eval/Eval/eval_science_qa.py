import argparse
import json
import os
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', nargs='+', default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems_list = json.load(open(os.path.join(base_dir, "test.json")))
    problems = {prob["question_id"]: prob for prob in problems_list}

    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {
        'acc': None,
        'correct': None,
        'count': None,
        'results': {},
        'outputs': {}
    }

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred = {'text': 'FAILED', 'prompt': 'Unknown'}
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text']

        # 解析预测答案
        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        else:
            # pattern = re.compile(r'The answer is ([A-Z]).')
            # res = pattern.findall(pred_text)
            # if len(res) == 1:
            #     answer = res[0]
            # else:
            #     answer = "FAILED"
            # pattern = re.compile(r'\b(\w)\b')
            # res = pattern.findall(pred_text)
            # if len(res) > 0:
            #     answer = res[0].upper()
            pattern = re.compile(r"\b([A-E])\b")
            res = pattern.findall(pred_text.upper())
            if len(res) > 0:
                answer = res[0]
            else:
                answer = "FAILED"
        # 获取索引
        pred_idx = args.options.index(answer) if answer in args.options else -1
        gt_idx = args.options.index(prob['answer'])  # prob['answer'] is "A", "B", etc.

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': prob['answer'],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        # ✅ 关键修正：直接存储 pred_idx，不要调用 get_pred_idx
        sqa_results['results'][prob_id] = pred_idx
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == gt_idx:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(split_problems)

    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)

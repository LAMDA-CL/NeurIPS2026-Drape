import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str,
                        default='data/instructions/Instructions_Original/OCRVQA/test.json')
    parser.add_argument('--result-file', type=str,
                        default='./results/CoIN/OCRVQA/task8/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/CoIN/OCRVQA/task8')
    return parser.parse_args()


def normalize_ocrvqa_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().casefold()
    if text.startswith("answer:"):
        text = text[len("answer:"):].strip()
    text = text.rstrip(" .,!?:;\"'`")
    return text


def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {data['question_id']: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        ground_truth = annotation['answer']
        # Original logic (kept for reference):
        # if 'Unanswerable' in result['text']:
        #     continue
        # if result['text'].lower() == ground_truth.lower():
        #     right += 1
        if 'unanswerable' in (result['text'] or '').casefold():
            continue
        if normalize_ocrvqa_answer(result['text']) == normalize_ocrvqa_answer(ground_truth):
            right += 1

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

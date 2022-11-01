import os
from modules.annotator import Annotator
from modules.tokenizer import Tokenizer
import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")

def annotate(line,args):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if args.segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if args.segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not args.no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str


def main2(args,file):
    tokenizer = Tokenizer(args.granularity, args.device, args.segmented)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)
    lines = []
    for item in file:
        lines.append(item.strip())

    count = 0
    sentence_set = set()
    sentence_to_tokenized = {}
    opt_result = []
    for line in lines:
        sent_list = line.split("\t")[1:]
        for idx, sent in enumerate(sent_list):
            if args.segmented:
                # print(sent)
                sent = sent.strip()
            else:
                sent = "".join(sent.split()).strip()
            if idx >= 1:
                if not args.no_simplified:
                    sentence_set.add(cc.convert(sent))
                else:
                    sentence_set.add(sent)
            else:
                sentence_set.add(sent)
    batch = []
    for sent in tqdm(sentence_set):
        count += 1
        if sent:
            batch.append(sent)
        if count % args.batch_size == 0:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r  # Get tokenization map.

    # 单进程模式
    for line in tqdm(lines):
        ret = annotate(line,args)
        opt_result.append(ret)

    return opt_result


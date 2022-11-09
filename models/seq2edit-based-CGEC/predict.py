# -*- coding: utf-8
import os
from transformers import BertModel
import torch
import tokenization
import argparse
from gector.gec_model import GecBERTModel
import re
from opencc import OpenCC

cc = OpenCC("t2s")

class seq2Edit:
    def __init__(self):
        self.args = self.load_args()
        self.model = GecBERTModel(vocab_path=self.args.vocab_path,
                         model_paths=self.args.model_path.split(','),
                         weights_names=self.args.weights_name.split(','),
                         max_len=self.args.max_len,
                         min_len=self.args.min_len,
                         iterations=self.args.iteration_count,
                         min_error_probability=self.args.min_error_probability,
                         min_probability=self.args.min_error_probability,
                         log=False,
                         confidence=self.args.additional_confidence,
                         is_ensemble=self.args.is_ensemble,
                         weigths=self.args.weights,
                         cuda_device=self.args.cuda_device
                         )
        self.tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=False)
    def load_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path',
                            help='Path to the model file',
                            default='./exps/seq2edit_lang8/Best_Model_Stage_2.th')  # GECToR模型文件，多个模型集成的话，可以用逗号隔开
        parser.add_argument('--weights_name',
                            help='Path to the pre-trained language model',
                            default='./plm/chinese-struct-bert-large', )  # 预训练语言模型文件，多个模型集成的话，每个模型对应一个PLM，可以用逗号隔开
        parser.add_argument('--vocab_path',
                            help='Path to the vocab file',
                            default='./data/output_vocabulary_chinese_char_hsk+lang8_5',
                            )  # 词表文件
        parser.add_argument('--max_len',
                            type=int,
                            help='The max sentence length'
                                 '(all longer will be truncated)',
                            default=200)  # 最大输入长度（token数目），大于该长度的输入将被截断
        parser.add_argument('--min_len',
                            type=int,
                            help='The minimum sentence length'
                                 '(all longer will be returned w/o changes)',
                            default=0)  # 最小修改长度（token数目），小于该长度的输入将不被修改
        parser.add_argument('--batch_size',
                            type=int,
                            help='The number of sentences in a batch when predicting',
                            default=128)  # 预测时的batch大小（句子数目）
        parser.add_argument('--iteration_count',
                            type=int,
                            help='The number of iterations of the model',
                            default=5)  # 迭代修改轮数
        parser.add_argument('--additional_confidence',
                            type=float,
                            help='How many probability to add to $KEEP token',
                            default=0.0)  # Keep标签额外置信度
        parser.add_argument('--min_probability',
                            type=float,
                            default=0)  # token级别最小修改阈值
        parser.add_argument('--min_error_probability',
                            type=float,
                            default=0.0)  # 句子级别最小修改阈值
        parser.add_argument('--is_ensemble',
                            type=int,
                            help='Whether to do ensembling.',
                            default=0)  # 是否进行模型融合
        parser.add_argument('--weights',
                            help='Used to calculate weighted average', nargs='+',
                            default=None)  # 不同模型的权重（加权集成）
        parser.add_argument('--cuda_device',
                            help='The number of GPU',
                            default=-1)  # 使用GPU编号
        parser.add_argument('--log',
                            action='store_true')  # 是否输出完整信息
        parser.add_argument('--seg',
                            action='store_true',
                            default=True)  # 是否切分长句预测后再合并
        # parser.add_argument('--input_file',
        #                     help='Path to the input file',
        #                     default=input)
        # 输入文件，要求：预先分好词/字
        args = parser.parse_args()
        return args



    def deal_input(self,input_list):
        input = []
        for ret in input_list:
            line = tokenization.convert_to_unicode(ret)
            tokens = self.tokenizer.tokenize(line)
            input.append(" ".join(tokens))
        return input

    def predict_for_http(self,input_list, log=True, seg=True):

        input_list = self.deal_input(input_list)

        sents = [s.strip() for s in input_list]

        predictions = []
        cnt_corrections = 0
        batch = []
        for sent in sents:
            batch.append(sent.split())
            if len(batch) == self.args.batch_size:  # 如果数据够了一个batch的话，
                preds, cnt = self.model.handle_batch(batch)
                assert len(preds) == len(batch)
                predictions.extend(preds)
                cnt_corrections += cnt
                batch = []

        if batch:
            preds, cnt = self.model.handle_batch(batch)
            assert len(preds) == len(batch)
            predictions.extend(preds)
            cnt_corrections += cnt

        assert len(sents) == len(predictions)

        results = []
        for i, ret in enumerate(predictions):
            ret_new = [tok.lstrip("##") for tok in ret]
            ret = cc.convert("".join(ret_new))
            results.append(cc.convert(ret))

        return results






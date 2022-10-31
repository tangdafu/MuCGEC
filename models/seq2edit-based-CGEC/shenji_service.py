import json

import requests
import tokenization
import argparse
import predict
import parallel_to_m2
from opencc import OpenCC
import compare_m2_for_evaluation
import pandas as pd


cc = OpenCC("t2s")
class Corrector_Service(object):
    def __init__(self):
        self.operation_dict = {
            'M': '缺失',
            'S': '替换',
            'R': '冗余',
            'W': '乱序',
            'noop': '无错误'
        }
        self.url = 'http://jkgpu.cshuaju.top:8082/corrector'
        self.word_dict = self.load_dict()


    def load_dict(self):
        df = pd.read_excel('word_dict.xlsx')
        key = list(df['错误'])
        val = list(df['正确'])
        word_dict = {}
        for k, v in zip(key, val):
            word_dict[k] = v
        return  word_dict

    def predict_dict(self, line) -> str:
        idx = line.find('从严治党')
        while idx != -1:
            idx_end = idx + 4
            if idx - 2 < 0:
                line = line[:idx] + '全面从严治党' + line[idx_end:]
            else:
                if line[idx - 2:idx] != '全面':
                    line = line[:idx] + '全面从严治党' + line[idx_end:]
            idx = line.find('从严治党', idx_end)
        for key in self.word_dict.keys():
            if key == '从严治党':
                continue
            line = line.replace(key, self.word_dict[key])
        return line



    def predict(self, input_texts):
        dict_inputs = [self.predict_dict(line) for line in input_texts]        

        input = []
        tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=False)
        for ret in dict_inputs:
            line = tokenization.convert_to_unicode(ret)
            tokens = tokenizer.tokenize(line)
            input.append(" ".join(tokens))
	
        # read parameters
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
                            default=0)  # 使用GPU编号
        parser.add_argument('--log',
                            action='store_true')  # 是否输出完整信息
        parser.add_argument('--seg',
                            action='store_true',
                            default=True)  # 是否切分长句预测后再合并
        parser.add_argument('--input_file',
                            help='Path to the input file',
                            default=input)
                            # 输入文件，要求：预先分好词/字
        args = parser.parse_args()
        results = predict.main(args)
        # print(results)
        data = {
            "token": '31b2acb73d49fe3656ef07b1e977a344',
            "inputStr": results
        }
        data = json.dumps(data)
        resp = requests.post(self.url, data)
        resp = json.loads(resp.text.encode('utf-8'))
        resp = resp['result']
        results = []
        for item in resp:
            results.append(item['cor_text'])
        # print(results)
        op_results = []
        for i in range(0, len(results)):
            op_results.append(str(i) + '\t' + input_texts[i] + '\t' + results[i])

        parser2 = argparse.ArgumentParser(description="Choose input file to annotate")
        parser2.add_argument("-f", "--file", default=op_results,
                             help="Input parallel file")
        parser2.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
        parser2.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
        parser2.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
        parser2.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation",
                             default="char")
        parser2.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion",
                             action="store_true")
        parser2.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
        parser2.add_argument("--segmented", help="Whether tokens have been segmented",
                             action="store_true")  # 支持提前token化，用空格隔开
        parser2.add_argument("--no_simplified", help="Whether simplifying chinese",
                             action="store_true")  # 将所有corrections转换为简体中文
        args2 = parser2.parse_args()

        opt = parallel_to_m2.main2(args2)

        hpy_edit = compare_m2_for_evaluation.main(opt)

        output = []
        for original, correct, tags in zip(input_texts, results, hpy_edit):
            tag_json = []
            for hpy_item in tags:
                b_idx, e_idx, operation, cor_str, num = hpy_item
                operation = self.operation_dict[operation]
                tag_json.append({
                    'begin_idx': b_idx,
                    'end_idx': e_idx,
                    'operation': operation,
                    'cor_str': cor_str
                })
            output.append({
                'original_text': original,
                'correct_text': correct,
                'tags': tag_json
            })


        return output

if __name__ == '__main__':
    service = Corrector_Service()
    result = service.predict(['各班子成员和各科室同志，国家审计署要加快各项目的审计进度，统筹做好当前项目审计、贫攻坚、年度总结计划',
                                '“莫道桑愉晚，为霞尚满天。”老党员是新中国审计事业的创始者、见证者和实践者，是党和国家的宝贵财富。',])
    print(result)
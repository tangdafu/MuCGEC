import datetime
import json
import os

import requests
import tokenization
import argparse
import predict
import parallel_to_m2
from opencc import OpenCC
import compare_m2_for_evaluation
import pandas as pd
import re
from opencc import OpenCC

cc = OpenCC("t2s")

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
        # 是否分段
        self.seg = True
        self.url = 'http://jkgpu.cshuaju.top:8082/corrector'
        self.token = '31b2acb73d49fe3656ef07b1e977a344'
        self.word_dict = self.load_dict()
        self.seq2Edit_Model = predict.seq2Edit()
        self.args2 = self.load_args2()
        self.pwd_path = os.path.abspath(os.path.dirname(__file__))

    def _write_result_line(self, data, result_path):
        jsonData = json.dumps(data, ensure_ascii=False)
        fileObject = open(result_path, 'a', encoding='UTF-8')
        fileObject.write(jsonData)
        fileObject.write('\n')
        fileObject.close()

    def _get_result_save_path(self):
        """
        获取存放结果的文件路径
        :return: 存放结果的文件路径
        """
        year = str(datetime.datetime.now().year)
        month = str(datetime.datetime.now().month)
        day = str(datetime.datetime.now().day)
        result_path_dir = os.path.join(self.pwd_path, 'result', year, month, day)
        result_path = os.path.join(result_path_dir, 'result.json')
        if not os.path.exists(result_path_dir):
            os.makedirs(result_path_dir)
        return result_path

    def load_args2(self):
        parser2 = argparse.ArgumentParser(description="Choose input file to annotate")
        parser2.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
        parser2.add_argument("-d", "--device", type=int, help="The ID of GPU", default=-1)
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
        return args2

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

    def request_csc_web(self,original_data):
        data = {
            "token": self.token,
            "inputStr": original_data
        }
        data = json.dumps(data)
        resp = requests.post(self.url, data)
        resp = json.loads(resp.text.encode('utf-8'))
        resp = resp['result']
        return resp

    def get_para_data(self,input_texts,results):
        assert len(input_texts) == len(results), print(len(input_texts), len(results))
        op_results = []
        for i in range(0, len(results)):
            op_results.append(str(i) + '\t' + input_texts[i] + '\t' + results[i])
        return op_results

    def deal_Json_results(self,input_texts,results,hpy_edit):
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

    def split_sentence(self,document: str, flag: str = "all", limit: int = 510):
        """
        Args:
            document:
            flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
            limit: 默认单句最大长度为510个字符
        Returns: Type:list
        """
        sent_list = []
        try:
            if flag == "zh":
                document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
            elif flag == "en":
                document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 英文单字符断句符
                document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
            else:
                document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                                  document)  # 特殊引号

            sent_list_ori = document.splitlines()
            for sent in sent_list_ori:
                sent = sent.strip()
                if not sent:
                    continue
                else:
                    while len(sent) > limit:
                        temp = sent[0:limit]
                        sent_list.append(temp)
                        sent = sent[limit:]
                    sent_list.append(sent)
        except:
            sent_list.clear()
            sent_list.append(document)
        return sent_list
    def predict(self, input_texts):
        result_path = self._get_result_save_path()

        sents = [s.strip() for s in input_texts]

        input_texts = sents


        # 截断
        subsents = []
        s_map = []
        len_map = []
        for i, sent in enumerate(input_texts):  # 将篇章划分为子句，分句预测再合并
            if self.seg:
                subsent_list = self.split_sentence(sent, flag="zh")
            else:
                subsent_list = [sent]
            s_map.extend([i for _ in range(len(subsent_list))])
            total = 0;
            for j in range(len(subsent_list)):
                if j == 0:
                    len_map.append(0)
                else:
                    len_map.append(total)
                total += len(subsent_list[j])
            subsents.extend(subsent_list)
        assert len(subsents) == len(s_map)

        dict_inputs = [self.predict_dict(line) for line in subsents]

        predictions = self.seq2Edit_Model.predict_for_http(dict_inputs)

        try:
            resp = self.request_csc_web(predictions)
            predictions = []
            for item in resp:
                predictions.append(item['cor_text'])
        except Exception as e:
            pass




        # print(results)
        op_results = self.get_para_data(subsents,predictions)

        opt = parallel_to_m2.main2(self.args2,op_results)

        hpy_edit = compare_m2_for_evaluation.main(opt)
        hpy_edit2 = [[] for _ in range(len(input_texts))]
        results = ["" for _ in range(len(input_texts))]
        for i, ret in enumerate(predictions):
            ret_new = [tok.lstrip("##") for tok in ret]
            ret = cc.convert("".join(ret_new))
            results[s_map[i]] += cc.convert(ret)
            temp = []
            for item in hpy_edit[i]:
                if item[2] == 'noop':
                    continue
                item[0] += len_map[i]
                item[1] += len_map[i]
                temp.append(item)
            hpy_edit2[s_map[i]].extend(temp)
        for i in range(len(hpy_edit2)):
            if len(hpy_edit2[i]) == 0:
                hpy_edit2[i].append([-1,-1,'noop','-NONE',0])
        output = self.deal_Json_results(input_texts,results,hpy_edit2)

        self._write_result_line(output, result_path)

        return output

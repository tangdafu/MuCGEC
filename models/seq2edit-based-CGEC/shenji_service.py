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

    def predict(self, input_texts):
        result_path = self._get_result_save_path()

        dict_inputs = [self.predict_dict(line) for line in input_texts]        

        results = self.seq2Edit_Model.predict_for_http(dict_inputs)

        try:
            resp = self.request_csc_web(results)
            results = []
            for item in resp:
                results.append(item['cor_text'])
        except Exception as e:
            pass

        # print(results)
        op_results = self.get_para_data(input_texts,results)

        opt = parallel_to_m2.main2(self.args2,op_results)

        hpy_edit = compare_m2_for_evaluation.main(opt)

        output = self.deal_Json_results(input_texts,results,hpy_edit)

        self._write_result_line(output, result_path)

        return output

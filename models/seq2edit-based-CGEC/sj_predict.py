


class SjPredictor():
    def __init__(self):
        self.csc_predictor = Predictor()

    def predict_gec(self, text_list: list) -> list:


        result = []
        return result

    def predict_csc_and_dict(self, text_list: list) -> list:
        response = self.csc_predictor.predict('token', text_list)
        cor_text = []
        for item in response:
            line = item['cor_text']
            line = self.predict_dict(line)
            cor_text.append(line)
        return cor_text

    def predict_dict(self, line) -> str:
        line = line.replace('国家审计署', '审计署').replace('新冠疫情', '新冠肺炎疫情')
        line = line.replace('人代会', '人民代表大会').replace('用社会主义核心价值推进', '用社会主义核心价值观推进')
        line = line.replace('改革领导小组', '改革委员会')
        idx = line.find('从严治党')
        while idx != -1:
            idx_end = idx + 4
            if idx - 2 < 0:
                line = line[:idx] + '全面从严治党' + line[idx_end:]
            else:
                if line[idx - 2:idx] != '全面':
                    line = line[:idx] + '全面从严治党' + line[idx_end:]
            idx = line.find('从严治党', idx_end)
        return line

    def to_output_formate(self, text_list: list, cor_text: list) -> list:
        result = []
        for line, cor_line in zip(text_list, cor_text):
            result.append({
                'original_text': line,
                'correct_text': cor_line
            })
        return result

    def predict(self, text_list: list):
        gec_result = self.predict_gec(text_list)
        cor_text = self.predict_csc_and_dict(gec_result)
        output = self.to_output_formate(text_list, cor_text)
        return output




if __name__ == '__main__':
    pass
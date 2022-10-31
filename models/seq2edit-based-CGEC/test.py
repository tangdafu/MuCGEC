# -*- coding: utf-8 -*-

import argparse
import predict
import parallel_to_m2
from opencc import OpenCC
cc = OpenCC("t2s")
# read parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    help='Path to the model file',
                    default='./exps/seq2edit_lang8/Best_Model_Stage_2.th')  # GECToR模型文件，多个模型集成的话，可以用逗号隔开
parser.add_argument('--weights_name',
                    help='Path to the pre-trained language model',
                    default='./plm/chinese-struct-bert-large',)  # 预训练语言模型文件，多个模型集成的话，每个模型对应一个PLM，可以用逗号隔开
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
parser.add_argument('--input_file',
                        help='Path to the input file',
                        default=['1冬阴功是泰国最著名的菜之一，它虽然不是很豪华，但它的味确实让人上瘾，做法也不难、不复杂。', '2首先，我们得准备：大虾六到九只、盐一茶匙、已搾好的柠檬汁三汤匙、泰国柠檬叶三叶、柠檬香草一根、鱼酱两汤匙、辣椒6粒，纯净水4量杯、香菜半量杯和草菇10个。', '3这样，你就会尝到泰国人爱的味道。', '4另外，冬阴功对外国人的喜爱不断地增加。', '5这部电影不仅是国内，在国外也很有名。', '6不管是真正的冬阴功还是电影的“冬阴功”，都在人们的心里刻骨铭心。', '7随着中国经济突飞猛进，建设工业与日俱增。', '8虽然工业的发展和城市规模的扩大对经济发展有积极作用，但是同时也对环境问题日益严重造成了空气污染问题。', '9那些空气污染也没有助于人民的身体建康。', '10由此可见，首先我们要了解一些关于空气污染对我们人生有什么危害的话题知道了这些常识以后对我们人类会有积极作用。以及要学会怎样应对和治理空气污染的问题。'])  # 输入文件，要求：预先分好词/字
args = parser.parse_args()
results = predict.main(args)
print(results)

op_results = []

for i in range(0,len(results)):
    op_results.append(str(i)+'\t'+results[i]+'\t'+results[i])

parser2 = argparse.ArgumentParser(description="Choose input file to annotate")
parser2.add_argument("-f", "--file", default=['1\t冬阴功是泰国最著名的菜之一，它虽然不是很豪华，但它的味确实让人上瘾，做法也不难、不复杂。\t冬阴功是泰国最著名的菜之一，它虽然不是很豪华，但它的味道确实让人上瘾，做法也不难、不复杂。', '2\t首先，我们得准备：大虾六到九只、盐一茶匙、已搾好的柠檬汁三汤匙、泰国柠檬叶三叶、柠檬香草一根、鱼酱两汤匙、辣椒6粒，纯净水4量杯、香菜半量杯和草菇10个。\t首先，我们得准备：大虾六到九只、盐一茶匙、已搾好的柠檬汁三汤匙、泰国柠檬叶三叶、柠檬香草一根、鱼酱两汤匙、辣椒6粒，纯净水4量杯、香菜半量杯和草菇10个。', '3\t这样，你就会尝到泰国人死爱的味道。\t这样，你就会尝到泰国人死爱的味道。', '4\t另外，冬阴功对外国人的喜爱不断地增加。\t另外，冬阴功对外国人的喜爱也不断地增加。', '5\t这部电影不仅是国内，在国外也很有名。\t这部电影不仅是在国内，在国外也很有名。', '6\t不管是真正的冬阴功还是电影的“冬阴功”，都在人们的心里刻骨铭心。\t不管是真正的冬阴功还是电影里的“冬阴功”，都在人们的心里刻骨铭刻。', '7\t随着中国经济突飞猛近，建造工业与日俱增。\t随着中国经济突飞猛快，建造工业与日俱增。', '8\t虽然工业的发展和城市规模的扩大对经济发展有积极作用，但是同时也对环境问题日益严重造成了空气污染问题。\t虽然工业的发展和城市规模的扩大对经济发展有积极作用，但是同时也使环境问题日益严重造成了空气污染问题。', '9\t那些空气污染也没有助于人生的身体建康。\t那些空气污染也没有助于人们的身体健康。', '10\t由此可见，首先我们要了解一些关于空气污染对我们人生有什么危害的话题知道了这些常识以后对我们人类会有积极作用。以及要学会怎样应对和治理空气污染的问题。\t由此可见，首先我们要了解一些关于空气污染对我们人生有什么危害的话题，知道了这些常识以后对我们人类会有积极作用。并且要学会怎何应对和治理空气污染的问题。', '11\t任何事情都是各有利弊，众所周知越建立工业越对经济方面有所发展。\t任何事情都是各有利弊，众所周知越建立工业对经济方面越有发展。', '12\t对我看来，曼古空气污染的问题与日俱增。\t在我看来，曼古空气污染的问题与日俱增。', '13\t每天会有不少的毒气体泄漏从工厂里出来。\t每天会有不少的有毒气体从工厂里出来。', '14\t在工厂里的工作人员为了工作，而每天吸了不少的毒气体，经过了一年多的时间，连工作人员也得了严重的病。更不用说住在这家工厂近的家庭。\t在工厂里的工作人员为了工作，而每天吸不少的毒气体，经过了一年多的时间，连工作人员也得了严重的病。更不用说住在这家工厂附近的家庭。', '15\t沙尘暴也是一类空气污染之一。\t沙尘暴也是一类空气污染之一。', '16\t不官是从口、眼、鼻子进去这样会伤害身体的建康。\t不管是从口、眼、鼻子进去，这样都会伤害身体的健康。', '17\t这样做会避免受到沙尘暴。\t这样做会避免受到沙尘暴。', '18\t最后，要关主一些关于天气预报的新闻。\t最后，要关注一些关于天气预报的新闻。', '19\t中国，悠久的历史，灿烂的文化，真是在历史上最难忘的国家。\t中国，悠久的历史，灿烂的文化，真是历史上最难忘的国家。', '20\t对一个生名来说空气污染是很危害的问题，对身体不好。\t对一个生人来说空气污染是很严重的问题，对身体不好。'], help="Input parallel file")
parser2.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
parser2.add_argument("-d", "--device", type=int, help="The ID of GPU", default=-1)
parser2.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
parser2.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation", default="char")
parser2.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion", action="store_true")
parser2.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
parser2.add_argument("--segmented", help="Whether tokens have been segmented", action="store_true")  # 支持提前token化，用空格隔开
parser2.add_argument("--no_simplified", help="Whether simplifying chinese", action="store_true")  # 将所有corrections转换为简体中文
args2 = parser2.parse_args()

opt = parallel_to_m2.main2(args2)

print(opt)

import compare_m2_for_evaluation

hpy_edit = compare_m2_for_evaluation.main(opt)



print(hpy_edit)


import tornado.ioloop
import tornado.web
import random
import numpy as np
import torch
from shenji_handler import *
import os

#cv2.setNumThread(0)

def seed_everything(seed=1234):
    '''固定随机种子'''
    random.seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_app():
    return tornado.web.Application([  # tornado配置，静态文件等
        (r"/corrector", Corrector_Handler)  # 配置路由
    ])


def main():
    seed_everything()
    app = make_app()
    app.listen(8085)
    tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    main()

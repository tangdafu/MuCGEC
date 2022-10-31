import tornado.web
import json
from shenji_service import Corrector_Service



class Corrector_Handler(tornado.web.RequestHandler):
    service = Corrector_Service()

    def post(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        # self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',  # '*')
                        'authorization, Authorization, Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
        post_data = self.request.body_arguments
        post_data = {x: post_data.get(x)[0].decode("utf-8") for x in post_data.keys()}
        if not post_data:
            post_data = self.request.body.decode('utf-8')
            post_data = json.loads(post_data)
        inputStr = post_data['inputStr']
        if not isinstance(inputStr, list):
            inputStr = [inputStr]
        input_list = []
        for inp in inputStr:
            inp = inp.replace('&nbsp;', '')  # 替换空格
            inp = inp.replace(r'\xa0', '')  # 替换空格
            inp = inp.replace(chr(0xa0), '')
            # inp = " ".join(inp.split())
            input_list.append(inp)
        resp = self.service.predict(input_list)
        resp = json.dumps(resp, ensure_ascii=False)
        self.write(resp)



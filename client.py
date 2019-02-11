import argparse
import json
import os
import requests
import string
from termcolor import colored

REQUEST = '{"inputs": ["%s"]}'
REQUEST_SCORE = '{"inputs": {"seq_input_src": ["%s"], "seq_input_tgt": ["%s"]}}'
EOS = "</s>"

class Client:
    def __init__(self, host):
        self.HOST = host
        self.URL = "http://{}/v1/models/seq2seq:predict".format(self.HOST)
    
    def _decode_single(self, response):
        try:
            server_ret = json.loads(response.text)
            query = " ".join(server_ret["outputs"][:-1])
        except:
            print("ZY: json decode error %s" % response.text)
            return "" 
        return query

    def _decode_multiple(self, response):
        ret = []
        try:
            server_ret = json.loads(response.text)
            batch_response = server_ret["outputs"]
            max_seq_len = len(batch_response)
            beam_size = len(batch_response[0])
            for j in range(beam_size):
                single_query = []
                for i in range(max_seq_len):
                    if batch_response[i][j] == EOS:
                        break
                    single_query.append(batch_response[i][j])
                ret.append(" ".join(single_query))
        except:
            print("ZY: json decode error %s" % response.text)
        return os.linesep.join(ret)

    def decode(self, response):
        server_ret = json.loads(response.text)
        if server_ret["outputs"]:
            if isinstance(server_ret["outputs"][0], (list,)):
                server_output = self._decode_multiple(response)
            else:
                server_output = self._decode_single(response)
        return server_output

    def _remove_punctuations(self, str_to_strip):
        str_to_strip = str_to_strip.replace('-', ' ')
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        cleantext = str_to_strip.translate(translator) # remove punc
        cleantext = ' '.join(cleantext.strip().split())
        return cleantext

    def _submit_generative(self, src_str):
        payload = REQUEST % (src_str)
        # print("request:")
        # print(colored(src_str, 'red'))
        response = requests.post(self.URL, data=payload.encode('utf-8'))
        server_output = self.decode(response)
        return server_output

    def _submit_discriminative(self, src_str, tgt_str):
        payload = REQUEST_SCORE % (src_str, tgt_str)
        payload_dummy = REQUEST_SCORE % ("", tgt_str)
        response = requests.post(self.URL, data=payload.encode('utf-8'))
        response_dummy = requests.post(self.URL, data=payload_dummy.encode('utf-8'))
        try:
            # return behavior defined by the tf-serving code
            server_ret = json.loads(response.text)
            server_ret_dummy = json.loads(response_dummy.text)
            score = sum(map(float, server_ret["outputs"]["seq_output"]))
            score_dummy = sum(map(float, server_ret_dummy["outputs"]["seq_output"]))
        except:
            print("ZY: json decode error %s" % response.text)
            return 0.0
        # score is negative log likelihood
        return score_dummy - score

    def submit(self, input_str, gen_or_dis, output_str=None):
        str_to_submit = self._remove_punctuations(input_str)
        if gen_or_dis:
            query = self._submit_generative(str_to_submit)
            print("reply:")
            print(colored(query, 'green'))
        else:
            tgt_to_submit = self._remove_punctuations(output_str)
            query = self._submit_discriminative(str_to_submit, tgt_to_submit)
            print("score={}".format(query))

def submit_generative(client):
    while True:
        input_str = input("document: ")
        client.submit(input_str.lower(), True)

def submit_discriminative(client):
    while True:
        input_str = input("document: ")
        output_str = input("query: ")
        client.submit(input_str.lower(), False, output_str.lower())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="archimedes.elca.mw.int:8501")
    args = parser.parse_args()


    client = Client(args.host)

    if args.model == "generative":
        submit_generative(client)
    elif args.model == "discriminative":
        submit_discriminative(client)
    else:
        raise NotImplementedError('Only supports "generative" or "discriminative" models')

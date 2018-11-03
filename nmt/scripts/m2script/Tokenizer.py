#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: Tokenizer.py
#
# A Penn Treebank tokenizer reimplemented based on the MOSES implementation.
#
# usage : %prog < input > output


import re
import sys


class DummyTokenizer(object):

    def tokenize(self, text):
        return text.split()



class PTBTokenizer(object):

    def __init__(self, language="en"):
        self.language = language
        self.nonbreaking_prefixes = {}
        self.nonbreaking_prefixes_numeric = {}
        self.nonbreaking_prefixes["en"] = ''' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 
            Adj Adm Adv Asst Bart Bldg Brig Bros Capt Cmdr Col Comdr Con Corp Cpl DR Dr Drs Ens 
            Gen Gov Hon Hr Hosp Insp Lt MM MR MRS MS Maj Messrs Mlle Mme Mr Mrs Ms Msgr Op Ord
            Pfc Ph Prof Pvt Rep Reps Res Rev Rt Sen Sens Sfc Sgt Sr St Supt Surg
            v vs i.e rev e.g Nos Nr'''.split()        
        self.nonbreaking_prefixes_numeric["en"] = '''No Art pp'''.split()
        self.special_chars = re.compile(r"([^\w\s\.\'\`\,\-\"\|\/])", flags=re.UNICODE)
                
    def tokenize(self, text, ptb=False):
        text = text.strip()
        text = " " + text + " "
        
        # Separate all "other" punctuation 

        text = re.sub(self.special_chars, r' \1 ', text)    
        text = re.sub(r";", r' ; ', text)    
        text = re.sub(r":", r' : ', text)
                            
        # replace the pipe character
        text = re.sub(r"\|", r' -PIPE- ', text)

        # split internal slash, keep others 
        text = re.sub(r"(\S)/(\S)", r'\1 / \2', text) 

        # PTB tokenization
        if ptb:
            text = re.sub(r"\(", r' -LRB- ', text)    
            text = re.sub(r"\)", r' -RRB- ', text)
            text = re.sub(r"\[", r' -LSB- ', text)    
            text = re.sub(r"\]", r' -RSB- ', text)
            text = re.sub(r"\{", r' -LCB- ', text)    
            text = re.sub(r"\}", r' -RCB- ', text)
        
            text = re.sub(r"\"\s*$", r" '' ", text)
            text = re.sub(r"^\s*\"", r' `` ', text)
            text = re.sub(r"(\S)\"\s", r"\1 '' ", text)
            text = re.sub(r"\s\"(\S)", r" `` \1", text)
            text = re.sub(r"(\S)\"", r"\1 '' ", text)
            text = re.sub(r"\"(\S)", r" `` \1", text)
            text = re.sub(r"'\s*$", r" ' ", text)
            text = re.sub(r"^\s*'", r" ` ", text)
            text = re.sub(r"(\S)'\s", r"\1 ' ", text)
            text = re.sub(r"\s'(\S)", r" ` \1", text) 
        
            text = re.sub(r"'ll", r" -CONTRACT-ll", text) 
            text = re.sub(r"'re", r" -CONTRACT-re", text) 
            text = re.sub(r"'ve", r" -CONTRACT-ve", text)
            text = re.sub(r"n't", r" n-CONTRACT-t", text)
            text = re.sub(r"'LL", r" -CONTRACT-LL", text) 
            text = re.sub(r"'RE", r" -CONTRACT-RE", text) 
            text = re.sub(r"'VE", r" -CONTRACT-VE", text)
            text = re.sub(r"N'T", r" N-CONTRACT-T", text)
            text = re.sub(r"cannot", r"can not", text)
            text = re.sub(r"Cannot", r"Can not", text)
        
        # multidots stay together
        text = re.sub(r"\.([\.]+)", r" DOTMULTI\1", text)
        while re.search("DOTMULTI\.", text):
            text = re.sub(r"DOTMULTI\.([^\.])", r"DOTDOTMULTI \1", text)
            text = re.sub(r"DOTMULTI\.", r"DOTDOTMULTI", text)
        
        # multidashes stay together
        text = re.sub(r"\-([\-]+)", r" DASHMULTI\1", text)
        while re.search("DASHMULTI\-", text):
            text = re.sub(r"DASHMULTI\-([^\-])", r"DASHDASHMULTI \1", text)
            text = re.sub(r"DASHMULTI\-", r"DASHDASHMULTI", text)

        # Separate ',' except if within number. 
        text = re.sub(r"(\D),(\D)", r'\1 , \2', text) 
        # Separate ',' pre and post number. 
        text = re.sub(r"(\d),(\D)", r'\1 , \2', text) 
        text = re.sub(r"(\D),(\d)", r'\1 , \2', text) 
            
        if self.language == "en":
            text = re.sub(r"([^a-zA-Z])'([^a-zA-Z])", r"\1 ' \2", text) 
            text = re.sub(r"(\W)'([a-zA-Z])", r"\1 ' \2", text)
            text = re.sub(r"([a-zA-Z])'([^a-zA-Z])", r"\1 ' \2", text)
            text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1 '\2", text)
            text = re.sub(r"(\d)'(s)", r"\1 '\2", text)
            text = re.sub(r" '\s+s ", r" 's ", text)
            text = re.sub(r" '\s+s ", r" 's ", text)
        elif self.language == "fr":
            text = re.sub(r"([^a-zA-Z])'([^a-zA-Z])", r"\1 ' \2", text) 
            text = re.sub(r"([^a-zA-Z])'([a-zA-Z])", r"\1 ' \2", text)
            text = re.sub(r"([a-zA-Z])'([^a-zA-Z])", r"\1 ' \2", text)
            text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1' \2", text)
        else:
            text = re.sub(r"'", r" ' ")
            
        # re-combine single quotes    
        text = re.sub(r"' '", r"''", text)    

        words = text.split()
        text = ''
        for i, word in enumerate(words):
            m = re.match("^(\S+)\.$", word)
            if m:
                pre = m.group(1) 
                if ((re.search("\.", pre) and re.search("[a-zA-Z]", pre)) or \
                    (pre in self.nonbreaking_prefixes[self.language]) or \
                    ((i < len(words)-1) and re.match("^\d+", words[i+1]))):
                    pass  # do nothing
                elif ((pre in self.nonbreaking_prefixes_numeric[self.language] ) and \
                      (i < len(words)-1) and re.match("\d+", words[i+1])):
                    pass  # do nothing
                else:
                    word = pre + " ."
                    
            text += word + " "
        text = re.sub(r"'\s+'", r"''", text)            
       
        # restore multidots
        while re.search("DOTDOTMULTI", text):
            text = re.sub(r"DOTDOTMULTI", r"DOTMULTI.", text)
        text = re.sub(r"DOTMULTI", r".", text)

        # restore multidashes
        while re.search("DASHDASHMULTI", text):
            text = re.sub(r"DASHDASHMULTI", r"DASHMULTI-", text)
        text = re.sub(r"DASHMULTI", r"-", text)    
        text = re.sub(r"-CONTRACT-", r"'", text)
   
        return text.split() 

    
    def tokenize_all(self,sentences, ptb=False):
        return [self.tokenize(t, ptb) for t in sentences]
            
# starting point
if __name__ == "__main__":
    tokenizer = PTBTokenizer()
    for line in sys.stdin:
        line = line.decode("utf8")
        tokens = tokenizer.tokenize(line.strip())
        out = ' '.join(tokens)
        print out.encode("utf8")

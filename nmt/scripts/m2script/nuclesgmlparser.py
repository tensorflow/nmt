# nuclesgmlparser.py
#
# Author:	Yuanbin Wu
#           National University of Singapore (NUS)
# Date:		12 Mar 2013
# Version:      1.0
# 
# Contact:  wuyb@comp.nus.edu.sg
#
# This script is distributed to support the CoNLL-2013 Shared Task.
# It is free for research and educational purposes.

from sgmllib import SGMLParser
from nucle_doc import nucle_doc


class nuclesgmlparser(SGMLParser):
    def __init__(self):
        SGMLParser.__init__(self)
        self.docs = []

    def reset(self):
        self.docs = []
        self.data = []
        SGMLParser.reset(self)

    def unknow_starttag(self, tag, attrs):
        pass

    def unknow_endtag(self):
        pass

    def start_doc(self, attrs):
        self.docs.append(nucle_doc())
        self.docs[-1].docattrs = attrs

    def end_doc(self):
        pass

    def start_matric(self, attrs):
        pass

    def end_matric(self):
        self.docs[-1].matric = ''.join(self.data)
        self.data = []
        pass

    def start_email(self, attrs):
        pass

    def end_email(self):
        self.docs[-1].email = ''.join(self.data)
        self.data = []
        pass

    def start_nationality(self, attrs):
        pass

    def end_nationality(self):
        self.docs[-1].nationality = ''.join(self.data)
        self.data = []
        pass

    def start_first_language(self, attrs):
        pass

    def end_first_language(self):
        self.docs[-1].firstLanguage = ''.join(self.data)
        self.data = []
        pass

    def start_school_language(self, attrs):
        pass

    def end_school_language(self):
        self.docs[-1].schoolLanguage = ''.join(self.data)
        self.data = []
        pass

    def start_english_tests(self, attrs):
        pass

    def end_english_tests(self):
        self.docs[-1].englishTests = ''.join(self.data)
        self.data = []
        pass


    def start_text(self, attrs):
        pass
    
    def end_text(self):
        pass

    def start_title(self, attrs):
        pass

    def end_title(self):
        self.docs[-1].paragraphs.append(''.join(self.data))
        self.data = []
        pass


    def start_p(self, attrs):
        pass

    def end_p(self):
        self.docs[-1].paragraphs.append(''.join(self.data))
        self.data = []
        pass


    def start_annotation(self, attrs):
        self.docs[-1].annotation.append(attrs)

    def end_annotation(self):
        pass

    def start_mistake(self, attrs):
        d = {}
        for t in attrs:
            d[t[0]] = int(t[1])
        self.docs[-1].mistakes.append(d)
        pass 

    def end_mistake(self):
        pass 

    def start_type(self, attrs):
        pass

    def end_type(self):
        self.docs[-1].mistakes[-1]['type'] = ''.join(self.data)
        self.data = []

    def start_correction(self, attrs):
        pass

    def end_correction(self):
        self.docs[-1].mistakes[-1]['correction'] = ''.join(self.data)
        self.data = []

    def start_comment(self, attrs):
        pass

    def end_comment(self):
        self.docs[-1].mistakes[-1]['comment'] = ''.join( self.data)
        self.data = []


    def handle_charref(self, ref):
        self.data.append('&' + ref)

    def handle_entityref(self, ref):
        self.data.append('&' + ref)

    def handle_data(self, text):
        if  text.strip() == '':
            self.data.append('')
            return
        else:
            if text.startswith('\n'):
                text = text[1:]
            if text.endswith('\n'):
                text = text[:-1]
            self.data.append(text)



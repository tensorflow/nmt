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

# file: util.py
#

import operator
import random
import math
import re

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.                                
        # Let's try using the fastest.                                                          
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode)


def randint(b, a=0):
    return random.randint(a,b)

def uniq(seq, idfun=None):
    # order preserving                                                                          
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:                                                               
        # if seen.has_key(marker)                                                               
        # but in new ones:                                                                      
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def sort_dict(myDict, byValue=False, reverse=False):
    if byValue:
        items = myDict.items()
        items.sort(key = operator.itemgetter(1), reverse=reverse)
    else:
        items = sorted(myDict.items())
    return items

def max_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return max(myDict.items(), key=skey)


def min_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return min(myDict.items(), key=skey)

def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def isASCII(word):
    try:
        word = word.decode("ascii")
        return True
    except UnicodeEncodeError :
        return False
    except UnicodeDecodeError:
        return False


def intersect(x, y):
    return [z for z in x if z in y]



# Mapping Windows CP1252 Gremlins to Unicode
# from http://effbot.org/zone/unicode-gremlins.htm
cp1252 = {
    # from http://www.microsoft.com/typography/unicode/1252.htm
    u"\x80": u"\u20AC", # EURO SIGN
    u"\x82": u"\u201A", # SINGLE LOW-9 QUOTATION MARK
    u"\x83": u"\u0192", # LATIN SMALL LETTER F WITH HOOK
    u"\x84": u"\u201E", # DOUBLE LOW-9 QUOTATION MARK
    u"\x85": u"\u2026", # HORIZONTAL ELLIPSIS
    u"\x86": u"\u2020", # DAGGER
    u"\x87": u"\u2021", # DOUBLE DAGGER
    u"\x88": u"\u02C6", # MODIFIER LETTER CIRCUMFLEX ACCENT
    u"\x89": u"\u2030", # PER MILLE SIGN
    u"\x8A": u"\u0160", # LATIN CAPITAL LETTER S WITH CARON
    u"\x8B": u"\u2039", # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u"\x8C": u"\u0152", # LATIN CAPITAL LIGATURE OE
    u"\x8E": u"\u017D", # LATIN CAPITAL LETTER Z WITH CARON
    u"\x91": u"\u2018", # LEFT SINGLE QUOTATION MARK
    u"\x92": u"\u2019", # RIGHT SINGLE QUOTATION MARK
    u"\x93": u"\u201C", # LEFT DOUBLE QUOTATION MARK
    u"\x94": u"\u201D", # RIGHT DOUBLE QUOTATION MARK
    u"\x95": u"\u2022", # BULLET
    u"\x96": u"\u2013", # EN DASH
    u"\x97": u"\u2014", # EM DASH
    u"\x98": u"\u02DC", # SMALL TILDE
    u"\x99": u"\u2122", # TRADE MARK SIGN
    u"\x9A": u"\u0161", # LATIN SMALL LETTER S WITH CARON
    u"\x9B": u"\u203A", # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u"\x9C": u"\u0153", # LATIN SMALL LIGATURE OE
    u"\x9E": u"\u017E", # LATIN SMALL LETTER Z WITH CARON
    u"\x9F": u"\u0178", # LATIN CAPITAL LETTER Y WITH DIAERESIS
}

def fix_cp1252codes(text):
    # map cp1252 gremlins to real unicode characters
    if re.search(u"[\x80-\x9f]", text):
        def fixup(m):
            s = m.group(0)
            return cp1252.get(s, s)
        if isinstance(text, type("")):
            # make sure we have a unicode string
            text = unicode(text, "iso-8859-1")
        text = re.sub(u"[\x80-\x9f]", fixup, text)
    return text

def clean_utf8(text):
    return filter(lambda x : x > '\x1f' and x < '\x7f', text)

def pairs(iterable, overlapping=False):
    iterator = iterable.__iter__()
    token = iterator.next()
    i = 0
    for lookahead in iterator:
        if overlapping or i % 2 == 0: 
            yield (token, lookahead)
        token = lookahead
        i += 1
    if i % 2 == 0:
        yield (token, None)

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L

def softmax(values):
    a = max(values)
    Z = 0.0
    for v in values:
        Z += math.exp(v - a)
    sm = [math.exp(v-a) / Z for v in values]
    return sm

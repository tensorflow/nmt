#!/usr/bin/python

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

# file: levenshtein.py

from optparse import OptionParser
from itertools import izip
from util import uniq
import re
import sys
from copy import deepcopy

# batch evaluation of a list of sentences
def batch_precision(candidates, sources, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=False, verbose=False):
    return batch_pre_rec_f1(candidates, sources, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose)[0]

def batch_recall(candidates, sources, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=False, verbose=False):
    return batch_pre_rec_f1(candidates, sources, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose)[1]

def batch_f1(candidates, sources, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=False, verbose=False):
    return batch_pre_rec_f1(candidates, sources, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose)[2]

def comp_p(a, b):
    try:
        p  = a / b
    except ZeroDivisionError:
        p = 1.0
    return p

def comp_r(c, g):
    try:
        r  = c / g
    except ZeroDivisionError:
        r = 1.0
    return r

def comp_f1(c, e, g, b):
    try:
        f = (1+b*b) * c / (b*b*g+e)
        #f = 2 * c / (g+e)
    except ZeroDivisionError:
        if c == 0.0:
            f = 1.0
        else:
            f = 0.0
    return f

def f1_suffstats(candidate, source, gold_edits, max_unchanged_words=2, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    stat_correct = 0.0
    stat_proposed = 0.0
    stat_gold = 0.0

    candidate_tok = candidate.split()
    source_tok = source.split()
    lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
    V, E, dist, edits = edit_graph(lmatrix, backpointers)
    if very_verbose:
        print "edit matrix:", lmatrix
        print "backpointers:", backpointers
        print "edits (w/o transitive arcs):", edits
    V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
    dist = set_weights(E, dist, edits, gold_edits, very_verbose)
    editSeq = best_edit_seq_bf(V, E, dist, edits, very_verbose)
    if very_verbose:
        print "Graph(V,E) = "
        print "V =", V
        print "E =", E
        print "edits (with transitive arcs):", edits
        print "dist() =", dist
        print "viterbi path =", editSeq
    if ignore_whitespace_casing:
        editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)
    correct = matchSeq(editSeq, gold_edits, ignore_whitespace_casing)
    stat_correct = len(correct)
    stat_proposed = len(editSeq)
    stat_gold = len(gold_edits)
    if verbose:
        print "SOURCE        :", source.encode("utf8")
        print "HYPOTHESIS    :", candidate.encode("utf8")
        print "EDIT SEQ      :", list(reversed(editSeq))
        print "GOLD EDITS    :", gold_edits
        print "CORRECT EDITS :", correct
        print "# correct     :", int(stat_correct)
        print "# proposed    :", int(stat_proposed)
        print "# gold        :", int(stat_gold)
        print "-------------------------------------------"
    return (stat_correct, stat_proposed, stat_gold)

def batch_multi_pre_rec_f1(candidates, sources, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    assert len(candidates) == len(sources) == len(gold_edits)
    stat_correct = 0.0
    stat_proposed = 0.0
    stat_gold = 0.0
    i = 0
    for candidate, source, golds_set in zip(candidates, sources, gold_edits):
        i = i + 1
        # Candidate system edit extraction
        candidate_tok = candidate.split()
        source_tok = source.split()
        #lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
        lmatrix1, backpointers1 = levenshtein_matrix(source_tok, candidate_tok, 1, 1, 1)
        lmatrix2, backpointers2 = levenshtein_matrix(source_tok, candidate_tok, 1, 1, 2)

        #V, E, dist, edits = edit_graph(lmatrix, backpointers)
        V1, E1, dist1, edits1 = edit_graph(lmatrix1, backpointers1)
        V2, E2, dist2, edits2 = edit_graph(lmatrix2, backpointers2)

        V, E, dist, edits = merge_graph(V1, V2, E1, E2, dist1, dist2, edits1, edits2)
        if very_verbose:
            print "edit matrix 1:", lmatrix1
            print "edit matrix 2:", lmatrix2
            print "backpointers 1:", backpointers1
            print "backpointers 2:", backpointers2
            print "edits (w/o transitive arcs):", edits
        V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
        
        # Find measures maximizing current cumulative F1; local: curent annotator only
        sqbeta = beta * beta
        chosen_ann = -1
        f1_max = -1.0

        argmax_correct = 0.0
        argmax_proposed = 0.0
        argmax_gold = 0.0
        max_stat_correct = -1.0
        min_stat_proposed = float("inf")
        min_stat_gold = float("inf")
        for annotator, gold in golds_set.iteritems():
            localdist = set_weights(E, dist, edits, gold, verbose, very_verbose)
            editSeq = best_edit_seq_bf(V, E, localdist, edits, very_verbose)
            if verbose:
                print ">> Annotator:", annotator
            if very_verbose:
                print "Graph(V,E) = "
                print "V =", V
                print "E =", E
                print "edits (with transitive arcs):", edits
                print "dist() =", localdist
                print "viterbi path =", editSeq
            if ignore_whitespace_casing:
                editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)
            correct = matchSeq(editSeq, gold, ignore_whitespace_casing, verbose)
            
            # local cumulative counts, P, R and F1
            stat_correct_local = stat_correct + len(correct)
            stat_proposed_local = stat_proposed + len(editSeq)
            stat_gold_local = stat_gold + len(gold)
            p_local = comp_p(stat_correct_local, stat_proposed_local)
            r_local = comp_r(stat_correct_local, stat_gold_local)
            f1_local = comp_f1(stat_correct_local, stat_proposed_local, stat_gold_local, beta)

            if f1_max < f1_local or \
              (f1_max == f1_local and max_stat_correct < stat_correct_local) or \
              (f1_max == f1_local and max_stat_correct == stat_correct_local and min_stat_proposed + sqbeta * min_stat_gold > stat_proposed_local + sqbeta * stat_gold_local):
                chosen_ann = annotator
                f1_max = f1_local
                max_stat_correct = stat_correct_local
                min_stat_proposed = stat_proposed_local
                min_stat_gold = stat_gold_local
                argmax_correct = len(correct)
                argmax_proposed = len(editSeq)
                argmax_gold = len(gold)

            if verbose:
                print "SOURCE        :", source.encode("utf8")
                print "HYPOTHESIS    :", candidate.encode("utf8")
                print "EDIT SEQ      :", [shrinkEdit(ed) for ed in list(reversed(editSeq))]
                print "GOLD EDITS    :", gold
                print "CORRECT EDITS :", correct
                print "# correct     :", int(stat_correct_local)
                print "# proposed    :", int(stat_proposed_local)
                print "# gold        :", int(stat_gold_local)
                print "precision     :", p_local
                print "recall        :", r_local
                print "f_%.1f         :" % beta, f1_local
                print "-------------------------------------------"
        if verbose:
            print ">> Chosen Annotator for line", i, ":", chosen_ann
            print ""
        stat_correct += argmax_correct
        stat_proposed += argmax_proposed
        stat_gold += argmax_gold

    try:
        p  = stat_correct / stat_proposed
    except ZeroDivisionError:
        p = 1.0

    try:
        r  = stat_correct / stat_gold
    except ZeroDivisionError:
        r = 1.0
    try:
        f1 = (1.0+beta*beta) * p * r / (beta*beta*p+r)
    except ZeroDivisionError:
        f1 = 0.0
    if verbose:
        print "CORRECT EDITS  :", int(stat_correct)
        print "PROPOSED EDITS :", int(stat_proposed)
        print "GOLD EDITS     :", int(stat_gold)
        print "P =", p
        print "R =", r
        print "F_%.1f =" % beta, f1
    return (p, r, f1)
    

def batch_pre_rec_f1(candidates, sources, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    assert len(candidates) == len(sources) == len(gold_edits)
    stat_correct = 0.0
    stat_proposed = 0.0
    stat_gold = 0.0
    for candidate, source, gold in zip(candidates, sources, gold_edits):
        candidate_tok = candidate.split()
        source_tok = source.split()
        lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
        V, E, dist, edits = edit_graph(lmatrix, backpointers)
        if very_verbose:
            print "edit matrix:", lmatrix
            print "backpointers:", backpointers
            print "edits (w/o transitive arcs):", edits
        V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
        dist = set_weights(E, dist, edits, gold, verbose, very_verbose)
        editSeq = best_edit_seq_bf(V, E, dist, edits, very_verbose)
        if very_verbose:
            print "Graph(V,E) = "
            print "V =", V
            print "E =", E
            print "edits (with transitive arcs):", edits
            print "dist() =", dist
            print "viterbi path =", editSeq
        if ignore_whitespace_casing:
            editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)
        correct = matchSeq(editSeq, gold, ignore_whitespace_casing)
        stat_correct += len(correct)
        stat_proposed += len(editSeq)
        stat_gold += len(gold)
        if verbose:
            print "SOURCE        :", source.encode("utf8")
            print "HYPOTHESIS    :", candidate.encode("utf8")
            print "EDIT SEQ      :", list(reversed(editSeq))
            print "GOLD EDITS    :", gold
            print "CORRECT EDITS :", correct
            print "# correct     :", stat_correct
            print "# proposed    :", stat_proposed
            print "# gold        :", stat_gold
            print "precision     :", comp_p(stat_correct, stat_proposed)
            print "recall        :", comp_r(stat_correct, stat_gold)
            print "f_%.1f          :" % beta, comp_f1(stat_correct, stat_proposed, stat_gold, beta)
            print "-------------------------------------------"

    try:
        p  = stat_correct / stat_proposed
    except ZeroDivisionError:
        p = 1.0

    try:
        r  = stat_correct / stat_gold
    except ZeroDivisionError:
        r = 1.0
    try:
        f1 = (1.0+beta*beta) * p * r / (beta*beta*p+r)
        #f1  = 2.0 * p * r / (p+r)
    except ZeroDivisionError:
        f1 = 0.0
    if verbose:
        print "CORRECT EDITS  :", stat_correct
        print "PROPOSED EDITS :", stat_proposed
        print "GOLD EDITS     :", stat_gold
        print "P =", p
        print "R =", r
        print "F_%.1f =" % beta, f1
    return (p, r, f1)

# precision, recall, F1
def precision(candidate, source, gold_edits, max_unchanged_words=2, beta=0.5, verbose=False):
    return pre_rec_f1(candidate, source, gold_edits, max_unchanged_words, beta, verbose)[0]

def recall(candidate, source, gold_edits, max_unchanged_words=2, beta=0.5, verbose=False):
    return pre_rec_f1(candidate, source, gold_edits, max_unchanged_words, beta, verbose)[1]

def f1(candidate, source, gold_edits, max_unchanged_words=2, beta=0.5, verbose=False):
    return pre_rec_f1(candidate, source, gold_edits, max_unchanged_words, beta, verbose)[2]

def shrinkEdit(edit):
    shrunkEdit = deepcopy(edit)
    origtok = edit[2].split()
    corrtok = edit[3].split()
    i = 0
    cstart = 0
    cend = len(corrtok)
    found = False
    while i < min(len(origtok), len(corrtok)) and not found:
        if origtok[i] != corrtok[i]:
            found = True
        else:
            cstart += 1
            i += 1
    j = 1
    found = False
    while j <= min(len(origtok), len(corrtok)) - cstart and not found:
        if origtok[len(origtok) - j] != corrtok[len(corrtok) - j]:
            found = True
        else:
            cend -= 1
            j += 1
    shrunkEdit = (edit[0] + i, edit[1] - (j-1), ' '.join(origtok[i : len(origtok)-(j-1)]), ' '.join(corrtok[i : len(corrtok)-(j-1)]))
    return shrunkEdit

def matchSeq(editSeq, gold_edits, ignore_whitespace_casing= False, verbose=False):
    m = []
    goldSeq = deepcopy(gold_edits)
    last_index = 0
    CInsCDel = False
    CInsWDel = False
    CDelWIns = False
    for e in reversed(editSeq):
        for i in range(last_index, len(goldSeq)):
            g = goldSeq[i]
            if matchEdit(e,g, ignore_whitespace_casing):
                m.append(e)
                last_index = i+1
                if verbose:
                    nextEditList = [shrinkEdit(edit) for edit in editSeq if e[1] == edit[0]]
                    prevEditList = [shrinkEdit(edit) for edit in editSeq if e[0] == edit[1]]

                    if e[0] != e[1]:
                        nextEditList = [edit for edit in nextEditList if edit[0] == edit[1]]
                        prevEditList = [edit for edit in prevEditList if edit[0] == edit[1]]
                    else:
                        nextEditList = [edit for edit in nextEditList if edit[0] < edit[1] and edit[3] == '']
                        prevEditList = [edit for edit in prevEditList if edit[0] < edit[1] and edit[3] == '']

                    matchAdj = any(any(matchEdit(edit, gold, ignore_whitespace_casing) for gold in goldSeq) for edit in nextEditList) or \
                        any(any(matchEdit(edit, gold, ignore_whitespace_casing) for gold in goldSeq) for edit in prevEditList)
                    if e[0] < e[1] and len(e[3].strip()) == 0 and \
                        (len(nextEditList) > 0 or len(prevEditList) > 0):
                        if matchAdj:
                            print "!", e
                        else:
                            print "&", e
                    elif e[0] == e[1] and \
                        (len(nextEditList) > 0 or len(prevEditList) > 0):
                        if matchAdj:
                            print "!", e
                        else:
                            print "*", e
    return m
        
def matchEdit(e, g, ignore_whitespace_casing= False):
    # start offset
    if e[0] != g[0]:
        return False
    # end offset
    if e[1] != g[1]:
        return False
    # original string
    if e[2] != g[2]:
        return False
    # correction string
    if not e[3] in g[3]:
        return False
    # all matches
    return True

def equals_ignore_whitespace_casing(a,b):
    return a.replace(" ", "").lower() == b.replace(" ", "").lower()


def get_edits(candidate, source, gold_edits, max_unchanged_words=2, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    candidate_tok = candidate.split()
    source_tok = source.split()
    lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
    V, E, dist, edits = edit_graph(lmatrix, backpointers)
    V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
    dist = set_weights(E, dist, edits, gold_edits, verbose, very_verbose)
    editSeq = best_edit_seq_bf(V, E, dist, edits)
    if ignore_whitespace_casing:
        editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)
    correct = matchSeq(editSeq, gold_edits)
    return (correct, editSeq, gold_edits)

def pre_rec_f1(candidate, source, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    candidate_tok = candidate.split()
    source_tok = source.split()
    lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
    V, E, dist, edits = edit_graph(lmatrix, backpointers)
    V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
    dist = set_weights(E, dist, edits, gold_edits, verbose, very_verbose)
    editSeq = best_edit_seq_bf(V, E, dist, edits)
    if ignore_whitespace_casing:
        editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)
    correct = matchSeq(editSeq, gold_edits)
    try:
        p  = float(len(correct)) / len(editSeq)
    except ZeroDivisionError:
        p = 1.0
    try:
        r  = float(len(correct)) / len(gold_edits)
    except ZeroDivisionError:
        r = 1.0
    try:
        f1 = (1.0+beta*beta) * p * r / (beta*beta*p+r)
        #f1  = 2.0 * p * r / (p+r)
    except ZeroDivisionError:
        f1 = 0.0
    if verbose:
        print "Source:", source.encode("utf8")
        print "Hypothesis:", candidate.encode("utf8")
        print "edit seq", editSeq
        print "gold edits", gold_edits
        print "correct edits", correct
        print "p =", p
        print "r =", r
        print "f_%.1f =" % beta, f1
    return (p, r, f1)

# distance function
def get_distance(dist, v1, v2):
    try:
        return dist[(v1, v2)]
    except KeyError:
        return float('inf')


# find maximally matching edit squence through the graph using bellman-ford
def best_edit_seq_bf(V, E, dist, edits, verby_verbose=False):
    thisdist = {}
    path = {}
    for v in V:
        thisdist[v] = float('inf')
    thisdist[(0,0)] = 0
    for i in range(len(V)-1):
        for edge in E:
            v = edge[0]
            w = edge[1]
            if thisdist[v] + dist[edge] < thisdist[w]:
                thisdist[w] = thisdist[v] + dist[edge]
                path[w] = v
    # backtrack
    v = sorted(V)[-1]
    editSeq = []
    while True:
        try:
            w = path[v]
        except KeyError:
            break
        edit = edits[(w,v)]
        if edit[0] != 'noop':
            editSeq.append((edit[1], edit[2], edit[3], edit[4]))
        v = w
    return editSeq


# # find maximally matching edit squence through the graph
# def best_edit_seq(V, E, dist, edits, verby_verbose=False):
#     thisdist = {}
#     path = {}
#     for v in V:
#         thisdist[v] = float('inf')
#     thisdist[(0,0)] = 0
#     queue = [(0,0)]
#     while len(queue) > 0:
#         v = queue[0]
#         queue = queue[1:]
#         for edge in E:
#             if edge[0] != v:
#                 continue
#             w = edge[1]
#             if thisdist[v] + dist[edge] < thisdist[w]:
#                 thisdist[w] = thisdist[v] + dist[edge]
#                 path[w] = v
#             if not w in queue:
#                 queue.append(w)
#     # backtrack
#     v = sorted(V)[-1]
#     editSeq = []
#     while True:
#         try:
#             w = path[v]
#         except KeyError:
#             break
#         edit = edits[(w,v)]
#         if edit[0] != 'noop':
#             editSeq.append((edit[1], edit[2], edit[3], edit[4]))
#         v = w
#     return editSeq

def prev_identical_edge(cur, E, edits):
    for e in E:
        if e[1] == cur[0] and edits[e] == edits[cur]:
            return e
    return None

def next_identical_edge(cur, E, edits):
    for e in E:
        if e[0] == cur[1] and edits[e] == edits[cur]:
            return e
    return None

def get_prev_edges(cur, E):
    prev = []
    for e in E:
        if e[0] == cur[1]: 
            prev.append(e)
    return prev

def get_next_edges(cur, E):
    next = []
    for e in E:
        if e[0] == cur[1]: 
            next.append(e)
    return next


# set weights on the graph, gold edits edges get negative weight
# other edges get an epsilon weight added
# gold_edits = (start, end, original, correction)
def set_weights(E, dist, edits, gold_edits, verbose=False, very_verbose=False):
    EPSILON = 0.001
    if very_verbose:
        print "set weights of edges()", 
        print "gold edits :", gold_edits

    gold_set = deepcopy(gold_edits)
    retdist = deepcopy(dist)

    M = {}
    G = {}
    for edge in E:
        tE = edits[edge]
        s, e = tE[1], tE[2]
        if (s, e) not in M:
            M[(s,e)] = []
        M[(s,e)].append(edge)
        if (s, e) not in G:
            G[(s,e)] = []

    for gold in gold_set:
        s, e = gold[0], gold[1]
        if (s, e) not in G:
            G[(s,e)] = []
        G[(s,e)].append(gold)
    
    for k in sorted(M.keys()):
        M[k] = sorted(M[k])

        if k[0] == k[1]: # insertion case
            lptr = 0
            rptr = len(M[k])-1
            cur = lptr

            g_lptr = 0
            g_rptr = len(G[k])-1

            while lptr <= rptr:
                hasGoldMatch = False
                edge = M[k][cur]
                thisEdit = edits[edge]
                # only check start offset, end offset, original string, corrections
                if very_verbose:
                    print "set weights of edge", edge 
                    print "edit  =", thisEdit
                
                cur_gold = []
                if cur == lptr:
                    cur_gold = range(g_lptr, g_rptr+1)
                else:
                    cur_gold = reversed(range(g_lptr, g_rptr+1))

                for i in cur_gold:
                    gold = G[k][i]
                    if thisEdit[1] == gold[0] and \
                        thisEdit[2] == gold[1] and \
                        thisEdit[3] == gold[2] and \
                        thisEdit[4] in gold[3]:
                        hasGoldMatch = True
                        retdist[edge] = - len(E)
                        if very_verbose:
                            print "matched gold edit :", gold
                            print "set weight to :", retdist[edge]
                        if cur == lptr:
                            #g_lptr += 1 # why?
                            g_lptr = i + 1
                        else:
                            #g_rptr -= 1 # why?
                            g_rptr = i - 1
                        break
                        
                if not hasGoldMatch and thisEdit[0] != 'noop':
                    retdist[edge] += EPSILON
                if hasGoldMatch:
                    if cur == lptr:
                        lptr += 1
                        while lptr < len(M[k]) and M[k][lptr][0] != M[k][cur][1]:
                            if edits[M[k][lptr]] != 'noop':
                                retdist[M[k][lptr]] += EPSILON
                            lptr += 1
                        cur = lptr
                    else:
                        rptr -= 1
                        while rptr >= 0 and M[k][rptr][1] != M[k][cur][0]:
                            if edits[M[k][rptr]] != 'noop':
                                retdist[M[k][rptr]] += EPSILON
                            rptr -= 1
                        cur = rptr
                else:
                    if cur == lptr:
                        lptr += 1
                        cur = rptr
                    else:
                        rptr -= 1
                        cur = lptr
        else: #deletion or substitution, don't care about order, no harm if setting parallel edges weight < 0
            for edge in M[k]:
                hasGoldMatch = False
                thisEdit = edits[edge]
                if very_verbose:
                    print "set weights of edge", edge 
                    print "edit  =", thisEdit
                for gold in G[k]:
                    if thisEdit[1] == gold[0] and \
                        thisEdit[2] == gold[1] and \
                        thisEdit[3] == gold[2] and \
                        thisEdit[4] in gold[3]:
                        hasGoldMatch = True
                        retdist[edge] = - len(E)
                        if very_verbose:
                            print "matched gold edit :", gold
                            print "set weight to :", retdist[edge]
                        break
                if not hasGoldMatch and thisEdit[0] != 'noop':
                    retdist[edge] += EPSILON
    return retdist

# add transitive arcs
def transitive_arcs(V, E, dist, edits, max_unchanged_words=2, very_verbose=False):
    if very_verbose:
        print "-- Add transitive arcs --"
    for k in range(len(V)):
        vk = V[k]
        if very_verbose:
            print "v _k :", vk

        for i in range(len(V)):
            vi = V[i]
            if very_verbose:
                print "v _i :", vi
            try:
                eik = edits[(vi, vk)]
            except KeyError:
                continue
            for j in range(len(V)):
                vj = V[j]
                if very_verbose:
                    print "v _j :", vj
                try:
                    ekj = edits[(vk, vj)]
                except KeyError:
                    continue
                dik = get_distance(dist, vi, vk)
                dkj = get_distance(dist, vk, vj)
                if dik + dkj < get_distance(dist, vi, vj):
                    eij = merge_edits(eik, ekj)
                    if eij[-1] <= max_unchanged_words:
                        if very_verbose:
                            print " add new arcs v_i -> v_j:", eij 
                        E.append((vi, vj))
                        dist[(vi, vj)] = dik + dkj
                        edits[(vi, vj)] = eij
    # remove noop transitive arcs 
    if very_verbose:
        print "-- Remove transitive noop arcs --"
    for edge in E:
        e = edits[edge]
        if e[0] == 'noop' and dist[edge] > 1:
            if very_verbose:
                print " remove noop arc v_i -> vj:", edge
            E.remove(edge)
            dist[edge] = float('inf')
            del edits[edge]
    return(V, E, dist, edits)


# combine two edits into one
# edit = (type, start, end, orig, correction, #unchanged_words)
def merge_edits(e1, e2, joiner = ' '):
    if e1[0] == 'ins':
        if e2[0] == 'ins':
            e = ('ins', e1[1], e2[2], '', e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'del':
            e = ('sub', e1[1], e2[2], e2[3], e1[4], e1[5] + e2[5])
        elif e2[0] == 'sub':
            e = ('sub', e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'noop':
            e = ('sub', e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
    elif e1[0] == 'del':
        if e2[0] == 'ins':
            e = ('sub', e1[1], e2[2], e1[3], e2[4], e1[5] + e2[5])
        elif e2[0] == 'del':
            e = ('del', e1[1], e2[2], e1[3] + joiner + e2[3], '', e1[5] + e2[5])
        elif e2[0] == 'sub':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e2[4], e1[5] + e2[5])
        elif e2[0] == 'noop':
            e = ('sub', e1[1], e2[2], e1[3] + joiner +  e2[3], e2[4], e1[5] + e2[5])
    elif e1[0] == 'sub':
        if e2[0] == 'ins':
            e = ('sub', e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'del':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
        elif e2[0] == 'sub':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'noop':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
    elif e1[0] == 'noop':
        if e2[0] == 'ins':
            e = ('sub', e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'del':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
        elif e2[0] == 'sub':
            e = ('sub', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e2[0] == 'noop':
            e = ('noop', e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
    else:
        assert False
    return e

# build edit graph
def edit_graph(levi_matrix, backpointers):
    V = []
    E = []
    dist = {}
    edits = {}
    # breath-first search through the matrix
    v_start = (len(levi_matrix)-1, len(levi_matrix[0])-1)
    queue = [v_start]
    while len(queue) > 0:
        v = queue[0]
        queue = queue[1:]
        if v in V:
            continue
        V.append(v)
        try:
            for vnext_edits in backpointers[v]:
                vnext = vnext_edits[0]
                edit_next = vnext_edits[1]
                E.append((vnext, v))
                dist[(vnext, v)] = 1
                edits[(vnext, v)] = edit_next
                if not vnext in queue:
                    queue.append(vnext)
        except KeyError:
            pass
    return (V, E, dist, edits)

# merge two lattices, vertices, edges, and distance and edit table
def merge_graph(V1, V2, E1, E2, dist1, dist2, edits1, edits2):
    # vertices
    V = deepcopy(V1)
    for v in V2:
        if v not in V:
            V.append(v)
    V = sorted(V)

    # edges
    E = E1
    for e in E2:
        if e not in V:
            E.append(e)
    E = sorted(E)

    # distances
    dist = deepcopy(dist1)
    for k in dist2.keys():
        if k not in dist.keys():
            dist[k] = dist2[k]
        else:
            if dist[k] != dist2[k]:
                print >> sys.stderr, "WARNING: merge_graph: distance does not match!"
                dist[k] = min(dist[k], dist2[k])

    # edit contents
    edits = deepcopy(edits1)
    for e in edits2.keys():
        if e not in edits.keys():
            edits[e] = edits2[e]
        else:
            if edits[e] != edits2[e]:
                print >> sys.stderr, "WARNING: merge_graph: edit does not match!"
    return (V, E, dist, edits)

# convenience method for levenshtein distance
def levenshtein_distance(first, second):
    lmatrix, backpointers = levenshtein_matrix(first, second)
    return lmatrix[-1][-1]
    

# levenshtein matrix
def levenshtein_matrix(first, second, cost_ins=1, cost_del=1, cost_sub=2):
    #if len(second) == 0 or len(second) == 0:
    #    return len(first) + len(second)
    first_length = len(first) + 1
    second_length = len(second) + 1

    # init
    distance_matrix = [[None] * second_length for x in range(first_length)]
    backpointers = {}
    distance_matrix[0][0] = 0
    for i in range(1, first_length):
        distance_matrix[i][0] = i
        edit = ("del", i-1, i, first[i-1], '', 0)
        backpointers[(i, 0)] = [((i-1,0), edit)]
    for j in range(1, second_length):
        distance_matrix[0][j]=j
        edit = ("ins", j-1, j-1, '', second[j-1], 0)
        backpointers[(0, j)] = [((0,j-1), edit)]

    # fill the matrix
    for i in xrange(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + cost_del
            insertion = distance_matrix[i][j-1] + cost_ins
            if first[i-1] == second[j-1]:
                substitution = distance_matrix[i-1][j-1]
            else:
                substitution = distance_matrix[i-1][j-1] + cost_sub
            if substitution == min(substitution, deletion, insertion):
                distance_matrix[i][j] = substitution
                if first[i-1] != second[j-1]:
                    edit = ("sub", i-1, i, first[i-1], second[j-1], 0)
                else:
                    edit = ("noop", i-1, i, first[i-1], second[j-1], 1)
                try:
                    backpointers[(i, j)].append(((i-1,j-1), edit))
                except KeyError:
                    backpointers[(i, j)] = [((i-1,j-1), edit)]
            if deletion == min(substitution, deletion, insertion):
                distance_matrix[i][j] = deletion
                edit = ("del", i-1, i, first[i-1], '', 0)
                try:
                    backpointers[(i, j)].append(((i-1,j), edit))
                except KeyError:
                    backpointers[(i, j)] = [((i-1,j), edit)]
            if insertion == min(substitution, deletion, insertion):
                distance_matrix[i][j] = insertion
                edit = ("ins", i, i, '', second[j-1], 0)
                try:
                    backpointers[(i, j)].append(((i,j-1), edit))
                except KeyError:
                    backpointers[(i, j)] = [((i,j-1), edit)]
    return (distance_matrix, backpointers)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
from copy import deepcopy
try:
    import Hangulpy as hp
except:
    from . import Hangulpy as hp
from konlpy.tag import Mecab;tagger=Mecab()
flatten = lambda l: [item for sublist in l for item in sublist]

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if w in to_index.keys() else to_index["<unk>"], seq))
    return Variable(torch.LongTensor(idxs))

def getBatch(batch_size,train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
        
        
def pad_to_batch_seq_pairs(batch,x_to_ix,y_to_ix):
    
    sorted_batch =  sorted(batch, key=lambda b:b[0].size(1),reverse=True) # sort by len
    x,y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p,y_p=[],[]
    for i in range(len(batch)):
        if x[i].size(1)<max_x:
            x_p.append(torch.cat([x[i],Variable(torch.LongTensor([x_to_ix['<pad>']]*(max_x-x[i].size(1)))).view(1,-1)],1))
        else:
            x_p.append(x[i])
        if y[i].size(1)<max_y:
            y_p.append(torch.cat([y[i],Variable(torch.LongTensor([y_to_ix['<pad>']]*(max_y-y[i].size(1)))).view(1,-1)],1))
        else:
            y_p.append(y[i])
        
    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in input_var]
    target_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in target_var]
    
    return input_var, target_var, input_len, target_len

def pad_to_batch_so_seq(batch,word2index):
    x,y = zip(*batch)
    max_x = max([s.size(1) for s in x])
    x_p = []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(torch.LongTensor([word2index['<pad>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
    return torch.cat(x_p), torch.cat(y).view(-1)


def build_vocab_seq_pairs(corpus,tied_vocab=False,noise=0,max_length=30):
    """
    corpus : \t로 구분 되는 sequence pair의 list
    """
    X_r,y_r=[],[] # raw

    for parallel in corpus:
        try:
            if '\t' not in parallel: continue
            so,ta = parallel[:-1].split('\t')
            if so.strip()=="" or ta.strip()=="": continue

            normalized_ta = tagger.morphs(ta)#normalize_string(ta).split()
            if noise>1:
                sos = make_noise(so,noise)
                sos.append(so)
            else:
                sos = [so]
            normalized_sos=[]
            for s in sos:
                normalized_so = tagger.morphs(s)
                if seq_check(normalized_so,normalized_ta,max_length):
                    X_r.append(normalized_so)
                    y_r.append(normalized_ta)
        except Exception as e:
            print(str(e))
            break
    print("Num of data : ",len(X_r),len(y_r))

    if tied_vocab:
        vocab = list(set(flatten(X_r+y_r)))
        print("vocab : ",len(vocab))
        word2index = {'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}
        for vo in vocab:
            word2index[vo]=len(word2index)
        #index2word = {v:k for k,v in word2index.items()}
        
        X_p,y_p=[],[]
        for so,ta in zip(X_r,y_r):
            X_p.append(prepare_sequence(['<s>']+so+['</s>'],word2index).view(1,-1))
            y_p.append(prepare_sequence(ta+['</s>'],word2index).view(1,-1))
        train_data = list(zip(X_p,y_p))
        
        return train_data,word2index
        
    else:
        source_vocab = list(set(flatten(X_r)))
        target_vocab = list(set(flatten(y_r)))
        print("vocab : ",len(source_vocab),len(target_vocab))

        source2index = {'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}
        for vo in source_vocab:
            source2index[vo]=len(source2index)
        #index2source = {v:k for k,v in source2index.items()}

        target2index = {'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}
        for vo in target_vocab:
            target2index[vo]=len(target2index)
        #index2target = {v:k for k,v in target2index.items()}
        
        X_p,y_p=[],[]
        for so,ta in zip(X_r,y_r):
            X_p.append(prepare_sequence(['<s>']+so+['</s>'],source2index).view(1,-1))
            y_p.append(prepare_sequence(ta+['</s>'],target2index).view(1,-1))
        train_data = list(zip(X_p,y_p))
        
        return train_data,source2index,target2index

def seq_check(normalized_so,normalized_ta,max_length):
    MIN_LENGTH=1
    MAX_LENGTH=max_length
    if len(normalized_so)>=MIN_LENGTH and len(normalized_so)<=MAX_LENGTH \
    and len(normalized_ta)>=MIN_LENGTH and len(normalized_ta)<=MAX_LENGTH:
        return True
    else:
        return False

def make_noise(text,num=1):
    result=[]
    text = deepcopy(text)
    for n in range(num):
        result.append(korean_noise(text))
            
    return result

confusion_dic = {
    'ㄱ' : ['r','R','ㄲ','ㅅ','ㄷ','ㄹ'],
    'ㄲ' : ['ㄱ','R'],
    'ㅂ' : ['q','Q','ㅃ','ㅈ','ㅁ'],
    'ㅃ' : ['ㅂ','Q'],
    'ㅈ' : ['w','W','ㅉ','ㄷ','ㅂ','ㄴ'],
    'ㅉ' : ['ㅈ','W'],
    'ㄷ' : ['e','E','ㄸ','ㅈ','ㄱ','ㅇ'],
    'ㄸ' : ['ㄷ','E'],
    'ㅅ' : ['t','T','ㅆ','ㄱ','ㅛ','ㅎ'],
    'ㅆ' : ['ㅅ','T'],
    'ㅛ' : ['y','Y','ㅕ','ㅗ','ㅅ'],
    'ㅕ' : ['u','U','ㅛ','ㅓ','ㅑ'],
    'ㅑ' : ['i','I','ㅕ','ㅐ','ㅏ'],
    'ㅐ' : ['o','O','ㅒ','ㅑ','ㅔ','ㅣ'],
    'ㅒ' : ['ㅐ','O'],
    'ㅔ' : ['p','P','ㅖ','[','ㅐ',';'],
    'ㅖ' : ['ㅔ','P'],
    'ㅁ' : ['a','A','ㄴ','ㅂ','ㅋ'],
    'ㄴ' : ['s','S','ㅁ','ㅇ','ㅈ','ㅌ'],
    'ㅇ' : ['d','D','ㄴ','ㄹ','ㄷ','ㅊ'],
    'ㄹ' : ['f','F','ㅇ','ㄱ','ㅎ','ㅍ'],
    'ㅎ' : ['g','G','ㄹ','ㅅ','ㅗ','ㅠ'],
    'ㅗ' : ['h','H','ㅎ','ㅛ','ㅓ','ㅜ'],
    'ㅓ' : ['j','J','ㅗ','ㅏ','ㅕ','ㅡ'],
    'ㅏ' : ['k','K','ㅓ','ㅣ','ㅑ',','],
    'ㅣ' : ['l','L','ㅏ',';','ㅐ','.'],
    'ㅋ' : ['z','Z','ㅁ','ㅌ'],
    'ㅌ' : ['x','X','ㅋ','ㄴ','ㅊ'],
    'ㅊ' : ['c','C','ㅌ','ㅍ','ㅇ'],
    'ㅍ' : ['v','V','ㅊ','ㅠ','ㄹ'],
    'ㅠ' : ['b','B','ㅍ','ㅜ','ㅎ'],
    'ㅜ' : ['n','N','ㅠ','ㅡ','ㅗ'],
    'ㅡ' : ['m','M','ㅜ',',','ㅓ']
}
    
def korean_noise(sent):
    try:
        num_try=0
        while True:
            num_try+=1

            if num_try>10:
                return sent

            c_index = random.choice(range(len(sent)))
            c = sent[c_index] 
            if hp.is_all_hangul(c)==False: continue
            if hp.has_jongsung(c):
                if random.uniform(0,1) > 0.5:
                    try:
                        decomposed = hp.decompose(c)
                        if random.uniform(0,1) > 0.6:
                            cho = decomposed[0]
                            joong = decomposed[1]
                        elif random.uniform(0,1) > 0.3:
                            cho = random.choice(confusion_dic[decomposed[0]])
                            joong = decomposed[1]
                        else:
                            cho = decomposed[0]
                            joong = random.choice(confusion_dic[decomposed[1]])
                        try:
                            composed = hp.compose(cho,joong)
                        except:
                            composed = "".join([cho,joong])

                    except:
                        continue

                    if random.uniform(0,1) > 0.5:
                        if random.uniform(0,1) > 0.5:
                            noised = composed + decomposed[-1]
                        else:
                            noised = composed + random.choice(confusion_dic[decomposed[-1]])
                    else:
                        noised = composed
                else:
                    try:
                        decomposed = hp.decompose(c)
                        if random.uniform(0,1) > 0.5:
                            composed = hp.compose(decomposed[2],decomposed[1],decomposed[0]) # 초성,중성 스위치
                        else:
                            composed = hp.compose(decomposed[2],decomposed[1]) 
                    except:
                        continue

                    noised = composed
            else:
                decomposed = hp.decompose(c)
                if random.uniform(0,1) > 0.6:
                    cho = decomposed[0]
                    joong = decomposed[1]
                elif random.uniform(0,1) > 0.3:
                    cho = random.choice(confusion_dic[decomposed[0]])
                    joong = decomposed[1]
                else:
                    cho = decomposed[0]
                    joong = random.choice(confusion_dic[decomposed[1]])
                try:
                    noised = hp.compose(cho,joong)
                except:
                    noised = "".join([cho,joong])
            break

        if c_index<len(sent)-1:
            sent = sent[:c_index] + noised + sent[c_index+1:]
        else:
            sent = sent[:c_index]+noised
    except:
        pass

    return sent
    
    

    
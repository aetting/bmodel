import gzip,re,scipy
import numpy as np
import torch
from collections import Counter
from torch.autograd import Variable
from scipy import spatial

meaning_dict_list = [
{'agent':'dog','patient':'cat','action':'chased'},
{'agent':'woman','patient':'car','action':'drove'}
]

inflections = {
'chase':{'active':'chases','passive':'chased'},
'drive':{'active':'drives','passive':'driven'}
}

lemmas = {}
for cat in inflections.items()[0][1]:
    for k in inflections:
        lemmas[inflections[k][cat]] = k

# lemmas = {'chases':'chase','chased':'chase','drives':'drive','driven':'drive'}

# glove='/Users/allysonettinger/Desktop/meaning_cc/modeling/models/pretrained_embeddings/glove/glove-Wik-Gig/glove.6B.100d.txt'
# glove = 'brouwerGloVe-50.txt'
# glove = 'brouwerGloVe-100.txt'


def comprehension_test(net,allsents,testsents,word2loc,word2dist,context_size,lemmatize_label = False,lemmatize_input = False):
    #now we have a normed matrix
    total = 0.
    numwrong = 0.
    for sent,meaningdict,voice in testsents:
        total += 1
        print sent
#         outputvec = get_meaning_label(meaningdict,word2dist,'active')
        hidden = Variable(torch.Tensor(1,context_size).fill_(.5))
        for word in sent:
            input_word_rep = get_word_rep(word,word2dist,lemmatize = lemmatize_input)
#             input_word_rep = get_word_rep(word,word2loc,lemmatize = lemmatize_input)
            input_word_rep = input_word_rep.unsqueeze(0)
            hidden = repackage_hidden(hidden)
            outputvec,hidden,_ = net(input_word_rep,hidden)

        normed_out = outputvec.data/torch.norm(outputvec.data)
        normed_out = torch.t(normed_out)

        #dot product of input vector with all available true vectors, and get max
        dots = torch.mm(norm_mat,normed_out)
        maxcos,maxind = torch.max(dots,0)
        maxind = maxind[0][0]
        
        top_meaning = meaninglist[maxind]
        if top_meaning['agent'] != meaningdict['agent'] \
        or top_meaning['patient'] != meaningdict['patient'] \
        or top_meaning['action'] != meaningdict['action']: 
            print 'INCORRECT. CHOSE %s'%top_meaning
            numwrong += 1

    percwrong = (numwrong/total)*100
    print 'Got %s percent wrong'%percwrong


def check_meaning(meaning_vec,meaningdict,matid2mng):
    norm_mat,id2meaning = matid2mng
    normed_out = meaning_vec.data/torch.norm(meaning_vec.data)
    normed_out = torch.t(normed_out)
    dots = torch.mm(norm_mat,normed_out)
    maxcos,maxind = torch.max(dots,0)
    maxind = maxind[0]
        
    top_meaning = id2meaning[maxind]
    if top_meaning['agent'] != meaningdict['agent'] \
    or top_meaning['patient'] != meaningdict['patient'] \
    or top_meaning['action'] != meaningdict['action']: 
        ans = 0
    else:
        ans = 1
    return ans,top_meaning
            
def read_meaning_dict(filename,dutch=False):
    meaning_dict_list = []
    triplets = []
    with open(filename,'rU') as file:
        for line in file:
            meaning_dict = {}
            linesplit = line.strip().split(',')
            triplet_ster = tuple(linesplit[0:3])
            triplet_mis = tuple(linesplit[0:2]+[linesplit[3]])
            if dutch:
                meaning_dict['agent'],meaning_dict['patient'],meaning_dict['action'],meaning_dict['action-mis'],meaning_dict['patdet'] = line.strip().split(',')
            else:
                meaning_dict['agent'],meaning_dict['patient'],meaning_dict['action'],meaning_dict['action-mis'] = line.strip().split(',')
            meaning_dict_list.append(meaning_dict)
            triplets.append(triplet_ster)
            triplets.append(triplet_mis)
    return meaning_dict_list,triplets

def generate_brouwer_train_sentences(meaning_dict_list,triplets,holdout=False,dutch=False):
    #***TODO figure out whether embeddings should be lemma***
    if dutch:
        activestr = '%s heeft %s %s'
        passivestr = '%s werd door %s %s' 
    else:
        activestr = 'the %s has the %s %s'
        passivestr = 'the %s was by the %s %s'      
    inputpairs = []
    if dutch:
        allnouns = ['de %s'%d['agent'] for d in meaning_dict_list] + ['%s %s'%(d['patdet'],d['patient']) for d in meaning_dict_list]
    else:
        allnouns = [d['agent'] for d in meaning_dict_list] + [d['patient'] for d in meaning_dict_list]        
    allverbs = [d['action'] for d in meaning_dict_list]
    #all permutations
    i = 0
    id2meaning = {}
    for action in allverbs:
        for ag in allnouns:
            for pa in allnouns:
                if dutch: trip = (ag.split()[1],pa.split()[1],action)
                else: trip = (ag,pa,action)
                trip_rev = (trip[1],trip[0],trip[2])
                if holdout:
                    if trip in triplets or trip_rev in triplets: 
                        continue
                if dutch:
                    meaning = {'agent':ag.split()[1],'patient':pa.split()[1],'action':action}
                else:    
                    meaning = {'agent':ag,'patient':pa,'action':action}
                active = activestr%(ag,pa,action)
                passive = passivestr%(pa,ag,action)
                active = active.split()
                passive = passive.split()
                inputpairs.append((active,meaning,i))
                inputpairs.append((passive,meaning,i))
                id2meaning[i] = meaning 
                i += 1
    if holdout:
        num_each_ster = (len(inputpairs))/len(meaning_dict_list)
    else:
        num_each_ster = (len(inputpairs)/2)/len(meaning_dict_list) 
    print 'all-comb len: %s'%len(inputpairs)
    print 'num each ster: %s'%num_each_ster   
    sters = []
    for meaning in meaning_dict_list:
        if dutch:
            active = activestr%('de %s'%meaning['agent'],'%s %s'%(meaning['patdet'],meaning['patient']),meaning['action'])
            passive = passivestr%('%s %s'%(meaning['patdet'],meaning['patient']),'de %s'%meaning['agent'],meaning['action'])
        else:
            active = activestr%(meaning['agent'],meaning['patient'],meaning['action'])
            passive = passivestr%(meaning['patient'],meaning['agent'],meaning['action'])
        active = active.split()
        passive = passive.split()
        sters.append((active,meaning,i))
        if not holdout: 
            sters.append((passive,meaning,i))
        id2meaning[i] = meaning
        i += 1
    sters = sters * num_each_ster
    inputpairs += sters
    return inputpairs,id2meaning

def generate_hoeks(meaning_dict_list,dutch=False):
    simsents = []
    if dutch:
        activestr = '%s heeft %s %s'
        passivestr = '%s werd door %s %s'    
    else:
        activestr = 'the %s has the %s %s'
        passivestr = 'the %s was by the %s %s'
    for meaning in meaning_dict_list:
        ag = meaning['agent']
        pa = meaning['patient']
        vb = meaning['action']
        vbm = meaning['action-mis']
        if dutch:
            patdet=meaning['patdet']
            agstr = 'de %s'%ag
            pastr = '%s %s'%(patdet,pa)
        else:
            agstr = ag
            pastr = pa
        passive = passivestr%(pastr,agstr,vb)
        reversal = activestr%(pastr,agstr,vb)
        mism_pass = passivestr%(pastr,agstr,vbm)
        mism_act = activestr%(pastr,agstr,vbm)
        reverse_mng = {'agent':pa,'patient':ag,'action':vb}
        mismp_mng = {'agent':ag,'patient':pa,'action':vbm}
        misma_mng = {'agent':pa,'patient':ag,'action':vbm}
        for s in [(passive,meaning,'passive'),(reversal,reverse_mng,'reversal'),(mism_pass,mismp_mng,'mis-pass'),(mism_act,misma_mng,'mis-act')]:
            simsents.append((s[0].split(),s[1],s[2]))
    return simsents

def get_vars(inputpairs,embs,debug=False,binary=True):
    word2id = make_word2id(inputpairs)
    word2loc = make_word2loc(word2id)
#     if debug:
#         word2dist, emb_size = make_toy_vecs(word2id,7)
#     else:
    word2dist,emb_size = load_word2dist(embs,word2id,debug=debug,binary=binary)
    vocab_size = len(word2id)
    
    return word2loc,word2dist,vocab_size,emb_size
    
def get_meaning_matrix(inputpairs,word2vec):
#     meaninglist = []
    vectorlist = []
    matid2mng = {}
    i = 0
    for _,mng,origid in inputpairs:
#     for sent,meaningdict,id in inputpairs: 
        meaning_vec = get_meaning_label(mng,word2vec,None)
#         meaninglist.append(meaningdict)
        normed = meaning_vec.data/torch.norm(meaning_vec.data)
        vectorlist.append(normed)
        matid2mng[i] = mng
        i += 1
    norm_mat = torch.stack(vectorlist)
    
    return norm_mat,matid2mng
    
def load_word2dist(file,worddict,binary=True,debug=False,filename=None,delim=None):
    if file.endswith('.gz'):
        fileObject = gzip.open(file, 'rU')
    else:
        fileObject = open(file, 'rU')
    
    if filename: small = open(filename,'w')
    wordVectors = {}
    line = 1
    linenum = 0
    print('Getting pretrained word vectors for vocabulary words')
    while line:
        line = fileObject.readline()
        linenum += 1
        if delim:
            s = line.lower().strip().split(delim)
        else:
            s = line.lower().strip().split()
        if len(s) < 3: 
            continue
        word = s[0]
        word = re.sub('"','',word)
#         if linenum % 100000 == 0:
#             print('            %s'%linenum)
        if word not in worddict: 
            continue
        if debug:
            vector = Variable(torch.Tensor(map(float,s[1:26])))
        else:
            vector = Variable(torch.Tensor(map(float,s[1:])))
        if binary:
            for ind in range(len(vector)):
                if vector.data[ind] > 0: 
                    vector.data[ind] = 1
                else: vector.data[ind] = 0
        if filename: 
            line = line.lower()
            line = re.sub('"','',line)
            line = re.sub(',',' ',line)
            small.write(line)
#         vector = np.array(map(float,s[1:]))
        wordVectors[word] = vector
    emb_size = len(vector)
    
    if filename: small.close() 
    fileObject.close()
    print('Done getting word vectors')
        
    return wordVectors,emb_size
    
def get_meaning_label(meaningdict,worddict,lemmatize=False):
    #***TODO decide whether action embedding should be lemma
    ag = worddict[meaningdict['agent']]
    pat = worddict[meaningdict['patient']]
    act = worddict[meaningdict['action']]
#     if lemmatize:
#         act = worddict[inflections[meaningdict['action']][voice]]
#     else:    
#         act = worddict[meaningdict['action']]
    
    return torch.cat((ag,act,pat))
    
def get_word_rep(word,input_word_dict,lemmatize=False):
    if lemmatize and (word in lemmas):
            rep = input_word_dict[lemmas[word]]
    else:
        rep = input_word_dict[word]
    
    return rep

#input list of sentences in string or list format
def make_word2id(inputpairs):
    sentencelist = [sent for sent,_,_ in inputpairs] + [mean['action'] for _,mean,_ in inputpairs]
    wordcounter = Counter()
    for sent in sentencelist:
        if type(sent) == str:
            wordlist = sent.split()
        else: wordlist = sent
        wordcounter.update(wordlist)
    return {w:id for id,w in enumerate(wordcounter.keys())}
    
def make_word2loc(word2id):
    word2loc = {}
    for word in word2id:
        word2loc[word] = Variable(torch.zeros(len(word2id)))
        index = word2id[word]
        word2loc[word][index] = 1
    return word2loc
    
def make_toy_vecs(word2id,size):
    word2dist_toy = {}
    for word in word2id:
        word2dist_toy[word] = Variable(torch.randn(size))
    return word2dist_toy, size
    
def freeze_weights(netw,layerlist):
    for par_to_freeze in layerlist:
        for name,param in netw.named_parameters():
            if name == '%s'%par_to_freeze: 
                param.requires_grad = False
    
def insert_saved_weights(netw,saveddict,weights_to_insert):
    if type(saveddict) == str:
        prevmoddict = torch.load(saveddict)
    else: 
        prevmoddict = saveddict
    for name,param in netw.named_parameters():
        if name in weights_to_insert:
            if name in prevmoddict: 
                param.data = prevmoddict[name]
            else:
                print('%s not in saved state dict'%name)
                
def repackage_hidden(h):
    return Variable(h.data)
    
def cos_dist(x,y):
    return (scipy.spatial.distance.cosine(x,y))
    
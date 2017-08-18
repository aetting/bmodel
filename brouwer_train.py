import torch,random,copy,pickle
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util_brouwer import *
from brouwer_model import *


def maxind(varvec):
    v = list(varvec.data)
    return v.index(max(v))
    
def trace(cr):
    try:
        nextcr = cr.previous_functions[0][0]
    except:
        print 'done'
    else:
        print nextcr
        trace(nextcr)


def get_weight_update(gradient,prev_update,lr,alpha):
    n = torch.norm(gradient)
    if n > 1: 
        rho = 1.0/n
    else: 
        rho = 1.0
    t1 = -1.0*lr*rho*gradient
    t2 = alpha*prev_update
    update = t1+t2
    return update
    
def train(net,phase,inputpairs_orig,word2loc,word2dist,matid2mng,labnum,context_size=200,vars_to_freeze=None,lemmatize_label=False,lemmatize_input=False,output='dist',outlog=None):
    num_items = 0
    num_updates = 0
    
    max_updates = 7000 #7000 in Brouwer
    items_per_update = 100 #100 in Brouwer
    print('%s items per update, %s total updates\n'%(items_per_update,max_updates))
    if outlog: outlog.write('%s items per update, %s total updates\n'%(items_per_update,max_updates))
    
    if phase == 'integ':
        input_word_dict = word2dist
    elif phase == 'full':
        input_word_dict = word2loc
    else:
        raise Exception('Invalid phase type for train()!')
        
    if vars_to_freeze:
        to_update = [p for n,p in net.named_parameters() if n not in vars_to_freeze]
    else:
        to_update = list(net.parameters())
        
#     id2word = {maxind(v):word for word,v in word2loc.items()}
#     print id2word
    criterion = nn.MSELoss()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(to_update,lr=.001)
    inputpairs = copy.deepcopy(inputpairs_orig)
    print net
    if outlog: outlog.write('\n' + str(net) + '\n')
#     for p in net.named_parameters(): print p
    prev_updates = []
    for p in to_update:
#         p.data = torch.from_numpy(np.random.uniform(-.25,.25,p.size()))
        prev_updates.append(torch.zeros(p.size()))
    num_sents = 0
    num_words = 0
    acc = (0.0,1.0)
    cumloss = 0
    
    lr = .2
    while num_updates < max_updates: 
        random.shuffle(inputpairs)
        tot = 0
        corr = 0
        for sentence,meaningdict,origid in inputpairs:
            tot += 1
#             print '\nSENT:' + ' '.join(sentence)
            num_sents += 1
            
            #initialize hidden state at .5 at start of every sentence
            hidden = Variable(torch.Tensor(1,context_size).fill_(.5))

#             meaning_label = Variable(torch.LongTensor([lab]))
            if output == 'loc':
                meaning_label = get_meaning_label(meaningdict,word2loc,lemmatize = lemmatize_label)
            elif output == 'dist':
                meaning_label = get_meaning_label(meaningdict,word2dist,lemmatize = lemmatize_label)
            elif output == 'cat':
                meaning_label = torch.zeros(labnum)
                meaning_label[lab] = 1
                meaning_label = Variable(meaning_label)
            else:
                raise Exception('Invalid output type for train()!')
            meaning_label = meaning_label.unsqueeze(0)
            for wInd,word in enumerate(sentence):
                num_words += 1
                input_word_rep = get_word_rep(word,input_word_dict)
                input_word_rep = input_word_rep.unsqueeze(0)
                
                if wInd == (len(sentence) - 1): 
                    label_word_rep = get_word_rep(sentence[0],word2loc)
                else:
                    label_word_rep = get_word_rep(sentence[wInd],word2loc) #TODO changed this without checking it!!!
                label_word_rep = label_word_rep.unsqueeze(0)
                
                hidden = repackage_hidden(hidden)
                out,hidden,_ = net(input_word_rep,hidden)
#                 out = net(input_word_rep)

                loss = criterion(out, meaning_label)
                loss.backward()
                cumloss += loss.data[0]
#                 loss = criterion(out, label_word_rep)
                
#                 label = id2word[maxind(label_word_rep.squeeze(0))]
#                 print 'CORRECT %s'%label

#             print 'CORRECT %s'%label
#             print 'OUT'
#             print out
#             print meaning_label
                    
            if output == 'cat':
                top = maxind(out.squeeze(0))
                topmng = id2meaning[top]
                guess = '%s %s %s'%(topmng['agent'],topmng['action'],topmng['patient'])
#                 print 'GUESS %s'%guess
                if top == lab: 
#                     print 'CORRECT'
                    corr += 1
            elif output in ('dist','loc'):
                ans,topmng = check_meaning(out,meaningdict,matid2mng)
                guess = '%s %s %s'%(topmng['agent'],topmng['action'],topmng['patient'])
#                 print 'GUESS %s'%guess
                if ans:
#                     print 'CORRECT'
                    corr += 1
                

            if num_sents % items_per_update == 0: 
#                     optimizer.step()
#                 print 'PARAMETERS'
                next_prev = []
                for parind,p in enumerate(to_update):
#                     print p
                    update = get_weight_update(p.grad.data,prev_updates[parind],lr,.9)
                    next_prev.append(update)
#                     print 'UPDATE %s'%update
                    p.data.add_(update)
#                     print 'UPDATED DATA %s'%p.data
                prev_updates = next_prev
                net.zero_grad()
                num_updates += 1

                if (num_updates % 100 == 0): print 'loss: %s lr: %s update: %s'%(cumloss,lr,num_updates)
                if outlog and (num_updates % 100 == 0): outlog.write('\nloss: %s  lr: %s'%(cumloss,lr))
                
                cumloss = 0

                if num_updates % 700 == 0:
                    lr *= .95
                if num_updates == max_updates: 
                    print 'Correct: %s out of %s (%s)'%(acc[0],acc[1],float(acc[0])/acc[1])
                    if outlog: outlog.write('\n\nCorrect: %s out of %s (%s)\n\n'%(acc[0],acc[1],float(acc[0])/acc[1]))
                    break

        acc = (corr,tot)
        s = 1
#         print loss.creator
#         trace(loss.creator)

def full_training(inputpairs,word2loc,word2dist,settingvars,matid2mng,labelnum,modelID=''):
    lemmatize_label = False
    lemmatize_input = False
    weights_to_freeze = ['integ.weight','integ_out.weight']
    
    (trainingsuf,dict,binary,context_size,retrieval_size,vocab_size,emb_size) = settingvars
    
    out = open('traininglog%s.txt'%modelID,'w')
    out.write('Training set: trainingpairs-%s\n'%trainingsuf)
    out.write('Embeddings: %s\n'%dict)
    out.write('Binary: %s\n'%binary)
    out.write('Context %s, Retrieval %s\n'%(context_size,retrieval_size))
    out.write('Vocab %s, Embedding %s\n'%(vocab_size,emb_size))

    wgt = open('weightcheck%s.txt'%modelID,'w')
    
    net1 = NetInteg(vocab_size,emb_size,context_size,labelnum,input='dist',output = 'dist')
    
    print '\n\nTRAINING PART ONE\n\n'
    if out: out.write('\nTRAINING PART ONE\n\n')

    wgt.write('\n\npre training 1\n\n')
    for p in net1.named_parameters(): wgt.write(str(p))
    
    train(net1,'integ',inputpairs,word2loc,word2dist,matid2mng,labelnum,context_size=context_size,output='dist',outlog=out)

    wgt.write('\n\npost training 1\n\n')
    for p in net1.named_parameters(): wgt.write(str(p))

    net2 = NetFull(vocab_size,emb_size,context_size,retrieval_size,labelnum,output = 'dist')

    insert_saved_weights(net2,net1.state_dict(),weights_to_freeze)
    freeze_weights(net2,weights_to_freeze)

    wgt.write('\n\npre training 2\n\n')
    for p in net2.named_parameters(): wgt.write(str(p))

    print '\n\n\n\nTRAINING PART TWO\n\n\n\n'
    if out: out.write('TRAINING PART TWO\n\n')
    train(net2,'full',inputpairs,word2loc,word2dist,matid2mng,labelnum,context_size=context_size,vars_to_freeze=weights_to_freeze,output='dist',outlog=out) 

    wgt.write('\n\npost training 2\n\n')
    for p in net2.named_parameters(): wgt.write(str(p))

    wgt.close()
    out.close()
    
    torch.save(net2.state_dict(), 'modelsave%s'%modelID)

    return net2
    
modelID = '1d'
binary = True

context_size = 200
retrieval_size = 80

dict='brouwerCOALS-100.txt'
trainingsuf = 'br-origfulldutch'

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

with open('settings%s'%modelID,'w') as settings: pickle.dump((trainingsuf,dict,binary,context_size,retrieval_size),settings,pickle.HIGHEST_PROTOCOL)

# meaning_dict_list = read_meaning_dict('sim1.csv')
# inputpairs,origid2meaning = generate_brouwer_train_sentences(meaning_dict_list)

# random.shuffle(inputpairs)
# inputpairs = inputpairs[0:12]
# origid2meaning = {id:m for _,m,id in inputpairs}
# for pair in inputpairs: print pair

with open('trainingpairs-%s'%trainingsuf) as inputfile: trainingpairs = pickle.load(inputfile)
word2loc,word2dist,vocab_size,emb_size = get_vars(trainingpairs,dict,debug=False,binary=binary)

settingvars = (trainingsuf,dict,binary,context_size,retrieval_size,vocab_size,emb_size)

matid2mng = get_meaning_matrix(trainingpairs,word2dist)

labelnum = len(set([l for _,_,l in trainingpairs]))

# net = testNet(vocab_size,emb_size,labelnum,context_size,intersize,input='dist',output='dist')
# net = NetInteg(vocab_size,emb_size,context_size,labelnum,input='dist',output = 'dist')

# train(net,inputpairs,word2loc,word2dist,matid2mng,labelnum,context_size=context_size,input='dist',output='dist')

full_training(trainingpairs,word2loc,word2dist,settingvars,matid2mng,labelnum,modelID=modelID)
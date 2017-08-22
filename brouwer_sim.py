import torch,random,copy,pickle
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util_brouwer import *
from brouwer_model import *
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate(inputpairs,trained_model,context_size,retrieval_size,word2loc,word2dist,matid2mng,lemmatize_input=False):
    results = {}
    tot = 0
    corr = 0
    for sentence,meaningdict,cond in inputpairs:
        tot += 1
        if cond not in results: results[cond] = []
        print '\n' + ' '.join(sentence)
        meaning_label = get_meaning_label(meaningdict,word2dist)
        meaning_label = meaning_label.unsqueeze(0)
        
        #initialize hidden state at .5 at start of every sentence
        hidden = Variable(torch.Tensor(1,context_size).fill_(.5))
#         meaning_label = get_meaning_label(meaningdict,word2dist,voice,lemmatize = lemmatize_label)
        for i,word in enumerate(sentence):
            print word
            input_word_rep = get_word_rep(word,word2loc)
            input_word_rep = input_word_rep.unsqueeze(0)
            hidden = repackage_hidden(hidden)
            out,hidden,retr,retrout = trained_model(input_word_rep,hidden)
            if i == (len(sentence) - 2):
#                 print 'is the PREV WORD'
                integ_prev = hidden.clone()
                retr_prev = retr.clone()
            if i == (len(sentence) - 1):
#                 print 'is the TARGET WORD'
                integ_targ = hidden.clone()
                retr_targ = retr.clone()
            ans,topmng = check_meaning(out,meaningdict,matid2mng)
            guess = '%s %s %s'%(topmng['agent'],topmng['action'],topmng['patient'])
#             print 'GUESS %s'%guess
        if ans: 
            corr += 1

        n400 = cos_dist(retr_prev.data[0].numpy(),retr_targ.data[0].numpy())
        p600 = cos_dist(integ_prev.data[0].numpy(),integ_targ.data[0].numpy())
        results[cond].append((n400,p600))
    print 'Correct: %s out of %s (%s)'%(corr,tot,float(corr)/tot)
    n400_means = {}
    p600_means = {}
    n400_se = {}
    p600_se = {}
    print results
    for cond in results:
        n400_means[cond] = np.mean(np.array([n400 for n400,_ in results[cond]]))
        n400_se[cond] = stats.sem(np.array([n400 for n400,_ in results[cond]]))
        p600_means[cond] = np.mean(np.array([p600 for _,p600 in results[cond]]))
        p600_se[cond] = stats.sem(np.array([p600 for _,p600 in results[cond]]))

    return (n400_means,n400_se),(p600_means,p600_se)
    
def plot_means(meandict,sedict,title,filestr,modelid,color='steelblue'):
    labels = ['passive','reversal','mis-pass','mis-act']
    means = [meandict[k] for k in labels]
    se = [sedict[k] for k in labels]
    
    ind = (np.arange(len(means))) # the x locations for the groups 
    width = 0.5 
    
    fig, ax = plt.subplots()
    rects = ax.bar(ind, means, width, color=color,yerr=se,error_kw={'ecolor':'black'})
#     rects = ax.bar(ind, means, width, color='r', yerr=men_std) 
    ax.set_ylabel('Condition')
    ax.set_title(title)
    ax.set_xticks(ind) 
    ax.set_xticklabels(labels)
    plt.xlim(xmin=-.5,xmax=3.5) 
    plt.savefig('plots/%s-%s.png'%(filestr,modelid))
    

modelID = '1r'

print 'Loading variables ...'
with open('settings/settings%s'%modelID) as settings: trainingsuf,embdic,binary,context_size,retrieval_size = pickle.load(settings)
with open('trainingpairs/trainingpairs-%s'%trainingsuf) as inputfile: trainingpairs = pickle.load(inputfile) 

if not embdic.startswith('embs'): embdic = 'embs/%s'%embdic

word2loc,word2dist,vocab_size,emb_size, = get_vars(trainingpairs,embdic,debug=False,binary=binary)
matid2mng = get_meaning_matrix(trainingpairs,word2dist)

labelnum = None

meaning_dict_list = read_meaning_dict('sentsources/sim1-d.csv',dutch=True)
siminput = generate_hoeks(meaning_dict_list,dutch=True)

netTest = NetFull(vocab_size,emb_size,context_size,retrieval_size,labelnum,output = 'dist')
loaded_dict = torch.load('modelsaves/modelsave%s'%modelID)
netTest.load_state_dict(loaded_dict)

(n400_means,n400_se),(p600_means,p600_se) = simulate(siminput,netTest,context_size,retrieval_size,word2loc,word2dist,matid2mng)
print n400_means
print n400_se
print p600_means
print p600_se

# n400means = {'passive': 0.33484166912370306, 'mis-pass': 0.48263581026787927, 'mis-act': 0.45988320220812245, 'reversal': 0.34160386911697888}
plot_means(n400_means,n400_se,'N400 means','N400',modelID)
plot_means(p600_means,p600_se,'P600 means','P600',modelID,color='peru')

# print cos_dist(np.array([1,0,3]),np.array([1,2,3]))
modelID = '4a'

context_size = 200
retrieval_size = 80

# trainingsuf = 'br-origfulldutch'
trainingsuf = 'br-holdout-dutch'
#trainingsuf = 'br-nonsterdutch8k'

traincode='full'

maxup = 7000 #7000 in Brouwer
itperup = 100 #100 in Brouwer

reducelr = False

embdic='embs/brouwerCOALS-100.txt'
binary = True

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

notes='train on holdout data: stereotypical sentences are passive only, and all-combinations has no sentences, active or passive, from the simulation triplets'

modelID = '4c'

context_size = 200
retrieval_size = 80

# trainingsuf = 'br-origfulldutch'
trainingsuf = 'br-holdoutB-dutch'
#trainingsuf = 'br-nonsterdutch8k'

traincode='full'

maxup = 7000 #7000 in Brouwer
itperup = 100 #100 in Brouwer

reducelr = True

embdic='embs/brouwerCOALS-100.txt'
binary = True

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

notes='train on holdout data B (even halves), this time with reducing lr in case it helps the erratic loss'

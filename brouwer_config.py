modelID = '4g'

context_size = 200
retrieval_size = 80

trainingsuf = 'br-orig-ster0.25-dutch'
# trainingsuf = 'br-origfulldutch'
# trainingsuf = 'br-holdoutB-dutch'
#trainingsuf = 'br-nonsterdutch8k'

traincode='full'

maxup = 7000 #7000 in Brouwer
itperup = 100 #100 in Brouwer

embdic='embs/brouwerCOALS-100.txt'
binary = True

reducelr=False
tryloc=False

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

notes='try .25 of ster data -- now with constant lr'

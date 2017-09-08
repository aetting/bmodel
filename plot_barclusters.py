import matplotlib.pyplot as plt
import numpy as np
import brouwer_sim


mods = ['1c','2a','2b','4c']
# mods = ['1c','2a']

# conds = ['a','b','c','d']
meanslist = [{'a':1,'b':2,'c':5,'d':.5},{'a':1,'b':3,'c':4,'d':.6},{'a':1,'b':4,'c':3,'d':.2}]
selist = [{'a':.1,'b':.2,'c':.5,'d':.05},{'a':.1,'b':.3,'c':.4,'d':.06},{'a':.1,'b':.4,'c':.3,'d':.02}]

def plotseveral():
    n4meanslist = []
    n4selist = []
    p6meanslist = []
    p6selist = []
    for mod in mods:
        n400means, n400se, p600means, p600se = brouwer_sim.main(mod,plot=False)
        n4meanslist.append(n400means)
        n4selist.append(n400se)
        p6meanslist.append(p600means)
        p6selist.append(p600se)
    
    clusterbars(n4meanslist,n4selist,'N400',mods)
    clusterbars(p6meanslist,p6selist,'P600',mods)
    
        
        

def clusterbars(meanslist,selist,title,mods):
    print 'PLOTTING'
    print meanslist
    print selist
    pos = list(range(len(mods))) 
    width = 0.2
    colors = ['#EE3224','#F78F1E','#EE3224','#FFC222']

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))
    conds = ['passive','reversal','mis-pass','mis-act']
    
    for i,cond in enumerate(conds):
        plt.bar([p + width*i for p in pos], 
        #using df['pre_score'] data,
        [e[cond] for e in meanslist], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=colors[i], 
        # with label the first value in first_name
        label=['t','t','t'],
        yerr=[e[cond] for e in selist],
        error_kw={'ecolor':'black'})
        
    # Set the y axis label
    ax.set_ylabel('Cosine')

    # Set the chart's title
#     ax.set_title('Simulated amplitudes')

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(mods)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, .6])

    # Adding the legend and showing the plot
    plt.legend(conds, loc='lower right')
    # plt.grid()
    plt.savefig('plots/%s-plotall.png'%title)
        
    
# pos = list(range(3)) 
# width = 0.25
# 
# # Plotting the bars
# fig, ax = plt.subplots(figsize=(10,5))
# 
# # Create a bar with pre_score data,
# # in position pos,
# plt.bar(pos, 
#         #using df['pre_score'] data,
#         [1,2,3], 
#         # of width
#         width, 
#         # with alpha 0.5
#         alpha=0.5, 
#         # with color
#         color='#EE3224', 
#         # with label the first value in first_name
#         label=['t','t','t']) 
# 
# # Create a bar with mid_score data,
# # in position pos + some width buffer,
# plt.bar([p + width for p in pos], 
#         #using df['mid_score'] data,
#         [3.5,3,2],
#         # of width
#         width, 
#         # with alpha 0.5
#         alpha=0.5, 
#         # with color
#         color='#F78F1E', 
#         # with label the second value in first_name
#         label=['t','t','t']) 
# 
# # Create a bar with post_score data,
# # in position pos + some width buffer,
# plt.bar([p + width*2 for p in pos], 
#         #using df['post_score'] data,
#         [1,1,1], 
#         # of width
#         width, 
#         # with alpha 0.5
#         alpha=0.5, 
#         # with color
#         color='#FFC222', 
#         # with label the third value in first_name
#         label=['t','t','t']) 

# # Set the y axis label
# ax.set_ylabel('Score')
# 
# # Set the chart's title
# ax.set_title('Test Subject Scores')
# 
# # Set the position of the x ticks
# ax.set_xticks([p + 1 * width for p in pos])
# 
# # Set the labels for the x ticks
# ax.set_xticklabels(['orig','SterOnly','AllCombOnly'])
# 
# # Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*4)
# plt.ylim([0, max([1,2,3,4])] )
# 
# # Adding the legend and showing the plot
# plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
# # plt.grid()
# plt.savefig('testcluster.png')

if __name__ == "__main__":
    plotseveral()
#     clusterbars(meanslist,selist,'N400')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


##############################################################
## plot functions
##############################################################
def bar_plot(pos, x=None, ylim=(0,1), title=None, c='#30a2da', **kwargs):

    ax = plt.gca()
    if x is None:
        x = np.arange(len(pos))
    ax.bar(x, pos, color=c, **kwargs)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(np.asarray(x), x)
    if title is not None:
        plt.title(title)

    return ax

def plot_belief_vs_prior(belief, prior, **kwargs):
    plt.subplot(121)
    bar_plot(belief, title='belief', **kwargs)
    plt.subplot(122)
    bar_plot(prior, title='prior', **kwargs)

##############################################################
## update functions
##############################################################

def lh_hallway(hall, z, door=0.75, ndoor=0.2): 
    ## to-do
    #p(D|x=door) = 0.75, p(D|x=non-door) = 0.2


    ########################
    #tlfwpfh ansdl dlTdmfEo tpstjepdlxjrk 1dl skdhf ghkrfbfdl 0.75dlsrjdla!

    #a = [0.75, 0.75, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.75, 0.2]
    likelihood = np.ones(len(hall)) # hall = 10 ro # 11111111111 dlfjgrp emfdjrkdltdma
    likelihood[hall==z] *= door
    likelihood[hall!=z] *= ndoor

    return likelihood



def update(likelihood, prior):
    #to-do

    
    return normalize(prior * likelihood)



def normalize(p):
    return p / sum(p) # wjscpgkqdl 1dl ehlehfhr normalize gkrl!

##############################################################
## prediction functions
##############################################################
def perfect_predict(belief, move):
    n = len(belief)
    result = np.zeros(n)
    for i in range(n):
        result[i] = belief[(i-move) % n]
    return result

def predict_move(belief, move, p_under, p_correct, p_over):
    n = len(belief)
    prior = np.zeros(n)
    for i in range(n):
        prior[i] = (
            belief[(i-move) % n]   * p_correct +
            belief[(i-move-1) % n] * p_over +
            belief[(i-move+1) % n] * p_under)      
    return prior

def predict_move_convolution(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range (kN):
            index = (i + (width-k) - offset) % N
            prior[i] += pdf[index] * kernel[k]
    return prior


##############################################################
## main
##############################################################







# prediction example(one step) ########################################
belief = [0, 0, .4, .6, 0, 0, 0, 0, 0, 0]
prior = predict_move(belief, 2, .1, .8, .1)
plot_belief_vs_prior(belief, prior)
plt.show()  # one step


# prediction example(100 step) ########################################
belief = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
predict_beliefs = []
    
for i in range(100):
    belief = predict_move(belief, 1, .1, .8, .1)
    predict_beliefs.append(belief)

def show_prior(step): # make interactive plot
    plt.cla()
    plt.title(f'Step {step}')
    bar_plot(predict_beliefs[step-1])

fig, ax = plt.subplots()
ani = FuncAnimation(fig, show_prior, frames=range(0,100))
plt.show()   # 100 step

# update example ########################################
belief = np.array([0.1] * 10)
plt.subplot(121)
bar_plot(belief, title='Before update', ylim=(0, .4))
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
likelihood = lh_hallway(hallway, z=1)
belief = update(likelihood, belief)  
plt.subplot(122)
bar_plot(belief, title='After update', ylim=(0, .4))
plt.show()

# Bayes' filter ########################################
def discrete_bayes_sim(prior, kernel, measurements, hallway):
    posterior = np.array([.1]*10)
    priors, posteriors = [], []
    for i, z in enumerate(measurements):

        prior = predict_move_convolution(posterior, 1, kernel)
        priors.append(prior)

        likelihood = lh_hallway(hallway, z)
        posterior = update(likelihood, prior)
        posteriors.append(posterior)
        #i=i

    return priors, posteriors

kernel = (.1, .8, .1)
z_prob = 1.0
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
zs = [hallway[i % len(hallway)] for i in range(50)]

priors, posteriors = discrete_bayes_sim(prior, kernel, zs, hallway)

def show_prior(step): # make interactive plot
    plt.cla()
    plt.title(f'Step {step}')
    bar_plot(posteriors[step], title='posteriors', ylim=(0, .4))

fig, ax = plt.subplots()
ani = FuncAnimation(fig, show_prior, frames=range(0,100), interval=1000)
plt.show()

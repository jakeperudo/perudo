from itertools import chain
from itertools import combinations
import pandas as pd
import numpy as np
import math
import random
# import sys
# np.set_printoptions(threshold=np.inf)
# pars = [dice, totaldice, diceValue, diceQuantity, P1state]
# x = [totalDice, playerDiceCount, diceQuantity, playerDiceQty)]

from scipy.stats import binom
from scipy.special import comb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_rows', None)

def rollTableFn(rolls):
    rt = (pd.Series(chain.from_iterable(rolls))).value_counts().sort_index()
    rollTable = rt.to_frame().T
    rollTableEmpty = pd.DataFrame(np.zeros((1, 6)), columns=[1, 2, 3, 4, 5, 6])
    rollTable = pd.concat([rollTableEmpty, rollTable], axis=0).groupby(level=0, axis=1).sum()
    rollTable.loc["Total"] = rollTable.sum()
    rollTable.drop(rollTable.head(2).index, inplace=True)
    return rollTable

def calcProb (x, binSize = 20):
    if(x[2] <= x[3]):
        return(1*binSize)
    else:
        n = x[0]-x[1]
        k = np.arange(min(x[2] - x[3], n),n+1)

        if x[-1] == 1:
            return math.floor((np.array([comb(n, K)*(1/6)**K*(5/6)**(n-K) for K in k]).sum())*binSize)
        else:
            return math.floor((np.array([comb(n, K) * (1/3) **K*(2/3)**(n-K) for K in k]).sum()) * binSize)
        # return math.floor((np.array([comb(n, K) * (1 / 6) ** K * (5 / 6) ** (n - K) for K in k]).sum()) * binSize)

def generateGameStates(nplayers, ndice, binSize = 20):
    # create the basic data frame with total dice and player dice count
    totalDice = nplayers * ndice
    ndicelist = np.arange(ndice+1)
    # Size = nplayers * (ndice+!)
    stateslist = np.tile(ndicelist, (nplayers, 1)).flatten()
    print(stateslist)
    nplayerslist = [str(x) for x in range(1, nplayers + 1)]
    print(nplayerslist)
    combn = list(set(combinations(stateslist, nplayers)))
    print(combn)
    states = pd.DataFrame(combn, columns=nplayerslist)
    states = states.loc[~(states == 0).all(axis=1)]
    states['Total'] = states.sum(axis=1)
    states = states[['Total', '1']].drop_duplicates().sort_values(by=['Total'], ignore_index=True)

    binSizelist  = np.arange(binSize+1)
    binSizelist = np.tile(binSizelist, (states.shape[0], 1)).flatten()

    gameStates = pd.DataFrame(binSizelist, columns=['prob_cat'])
    states = pd.DataFrame(pd.np.repeat(states.values, binSize+1, axis=0), columns=['Total', 'p0'])
    gameStates = pd.concat([states, gameStates], axis=1)
    return gameStates

def generateRewardMatrix (gameStates):
    reward = np.zeros((gameStates.shape[0], gameStates.shape[0]))
    for i in range(reward.shape[0]):
        total = gameStates.loc[i, 'Total']
        p0 = gameStates.loc[i, 'p0']
        reward[i, (p0 - gameStates.loc[:, 'p0'] == 1) & (total - gameStates.loc[:, 'Total'] == 1) & (
                    gameStates.loc[:, 'p0'] != gameStates.loc[:, 'Total'])] = -1
        reward[i, (p0 == gameStates.loc[:, 'p0']) & (total - gameStates.loc[:, 'Total'] == 1) & (
                    gameStates.loc[:, 'p0'] != gameStates.loc[:, 'Total'])] = 1
        if (p0 == 1):
            reward[i, np.where((total - gameStates.loc[:, 'Total'] == 1) & (gameStates.loc[:, 'p0'] == 0))] = -10

        # win states when the player dice count equals the total dice count
        if (total - p0 == 1):
            reward[i, np.where((gameStates.loc[:, 'Total'] == p0) & (gameStates.loc[:, 'p0'] == p0))] = 10
    return reward

def liarsDiceRound(players, control, playerDiceCount, agents, gameStates, reward, Qmat, a = 1, verbose = 0):
    rolls = []
    for i in range (len(playerDiceCount)):
        rolls.append(np.random.choice(np.arange(1,7),playerDiceCount[i]))
    rolls = np.array(rolls,dtype='object')

    totalDice = sum(playerDiceCount.flatten())

    penalty  = np.zeros((players,))
    # SUM OF DICE
    rollTable = rollTableFn(rolls)

    p0State = np.array(gameStates[(gameStates.loc[:, 'Total'] == totalDice) & (gameStates.loc[:, 'p0'] == playerDiceCount[0]) & (gameStates.loc[:, 'prob_cat'] == totalDice)].index.tolist())

    dice = rolls[control]
    totalDice = totalDice
    diceValue = None
    diceQuantity = 0

    p0State = p0State
    pars = np.array([dice,totalDice,diceValue,diceQuantity,p0State],dtype="object")

    reslist = agents[control](pars,Qmat)

    if len(reslist) == 3:
        agentAction = reslist[0]
        diceValue = reslist[1]
        # print('Hereeeee 111111')
        diceQuantity = reslist[2]
        # print(diceQuantity)
    else:
        diceValue = reslist[0]
        diceQuantity = reslist[1]


    playerDiceQty = rollTableFn(rolls[[0]]).loc['Total', diceValue]
    probCat = calcProb(np.array([totalDice, playerDiceCount[0], diceQuantity, playerDiceQty,diceValue]))
    p0State = np.array(gameStates[(gameStates.loc[:,'Total'] == totalDice) & (gameStates.loc[:,'p0'] == playerDiceCount[0]) & (gameStates.loc[:,'prob_cat'] == probCat)].index.tolist())
    p0Action = "Raise"

    yCtrl = np.array([])
    yState = np.array([])
    yAction = np.array([])

    prev = control
    # print('previous {}'.format(control))
    control = control % (players) + 1
    if (control == players):
        control = 0
    # print('next {}'.format(control))
    Called = False

    while (not Called):

        if (playerDiceCount[control]) > 0:
            pars = np.array([rolls[control],totalDice, diceValue, diceQuantity, p0State],dtype="object")
            # agent.action = agentActionReslist
            agentActionReslist = agents[control](pars=pars, Qmat = Qmat)

            if (len(agentActionReslist) == 3):
                action = agentActionReslist[0]
            else:
                action = None

            if (control == 0 and (not(action is None))):
                playerDiceQty = rollTableFn(rolls[[0]]).loc['Total',diceValue]
                # agent.action = agentActionReslist
                p0Action = agentActionReslist[0]
                probCat = calcProb(np.array([totalDice, playerDiceCount[0], diceQuantity, playerDiceQty,diceValue]))
                p0State = np.array(gameStates[(gameStates.loc[:, 'Total'] == totalDice) & (gameStates.loc[:, 'p0'] == playerDiceCount[0]) & (gameStates.loc[:, 'prob_cat'] == probCat)].index.tolist())
            # if (action % in% c("call", "c"))

            if (action == "Call"):
                if ((diceQuantity) > (rollTable.iloc[0, diceValue-1])):
                    penalty[prev] = penalty[prev] - 1
                    # print("player", prev, "lost a die")
                else:
                    penalty[control] = penalty[control] - 1

                    # print("player", control, "lost a die")


                yCtrl = np.append(yCtrl, control)
                yState = np.append(yState, p0State)
                yAction = np.append(yAction, p0Action)

                probCat = calcProb(np.array([totalDice, playerDiceCount[0], diceQuantity, playerDiceQty,diceValue]))
                p0State = np.array(gameStates[(gameStates.loc[:, 'Total'] == totalDice-1) & (gameStates.loc[:, 'p0'] == playerDiceCount[0]+ penalty[0]) & (gameStates.loc[:, 'prob_cat'] == probCat)].index.tolist())

                Called = True

            else:
                diceValue = agentActionReslist[1]
                diceQuantity = agentActionReslist[2]

                probCat = calcProb(np.array([totalDice, playerDiceCount[0], diceQuantity, playerDiceQty,diceValue]))
                p0State = np.array(gameStates[(gameStates.loc[:, 'Total'] == totalDice) & (gameStates.loc[:, 'p0'] == playerDiceCount[0]) & (gameStates.loc[:, 'prob_cat'] == probCat)].index.tolist())

            yCtrl = np.append(yCtrl, control)
            yState = np.append(yState, p0State)
            yAction = np.append(yAction, p0Action)
            prev = control

        control = control % (players) + 1
        if (control == players):
            control = 0

    play = pd.DataFrame(np.hstack((yCtrl[:, None], yState[:, None], yAction[:, None])),columns=['yCtrl', 'yState', 'yAction'])
    # print('total dice is {}'.format(totalDice))
    return [penalty,play]

def buildAgent (bluffProb, method = "random"):
    def buildAgentFn(pars, Qmat):
        bluff = np.random.choice([True,False],1,p = bluffProb )[0]
        # pobability table
        rollTable = rollTableFn([pars[0]])
        ptable = rollTable / int(rollTable.sum(axis=1))

        if (pars[2] == None):
            newDiceValue = ptable.idxmax(axis="columns").values[0]
            newDiceQuantity = max(rollTable.iloc[0, :]) + 1
            return [newDiceValue, newDiceQuantity]

        if (method == "Qdecide"):
            if (abs((max(Qmat.iloc[pars[-1][0], :])) - min(Qmat.iloc[pars[-1][0], :])) < 1e-6):
                # 0 means Flase, 1 means True

                Call = np.random.choice(2, 1)[0]
                if (Call):
                    return ["Call", pars[2], pars[3]]

            else:
                if(np.random.uniform(0, 1, 1)< 0.1):
                    # print('I am in explore  2222')
                    Call = np.random.choice(2, 1)[0]
                    if (Call):
                        return ["Call", pars[2], pars[3]]
                else:
                    Call = (Qmat.iloc[pars[-1][0], :].idxmax(axis=1) == 'Call')
                    if (Call):
                        return ["Call", pars[2], pars[3]]

        elif(method == "random"):
            prob = 0.5
            Call = np.random.choice(2, 1, p=[prob, 1 - prob])[0]
            if (Call):
                return ["Call", pars[2], pars[3]]

        elif (method == "randomV2"):
            # print('range is {}'.format(math.ceil(pars[1] / 4.0)))
            proDiceValue = ptable.loc[:, pars[2]].values[0]
            if (pars[2] == 1):
                limit = math.ceil((pars[1] / 6.0) + np.random.choice(math.ceil(pars[1] / 4.0), 1))
            else:
                limit = math.ceil((pars[1] / 6.0)*2 + np.random.choice(math.ceil(pars[1] / 4.0), 1))
            # print('the limit value is {}'.format(limit))

            if ((pars[3] >= limit) or (proDiceValue < 0.1)):
                # print('I am calling right now')
                return ["Call", pars[2], pars[3]]

        elif (method == "trueProb"):
            k = np.max(np.array([pars[3] - rollTable.iloc[0, pars[2] - 1], 0]))
            if (pars[2] == 1):
                prob = 1 - np.array([binom.pmf(x, pars[1] - len(pars[0]), 1 / 6) for x in np.arange(1,k + 1)]).sum()
            else:
                prob = 1 - np.array([binom.pmf(x, pars[1] - len(pars[0]), 1 / 3) for x in np.arange(1, k + 1)]).sum()

            if abs(prob) < 1e-6:
                prob = 0
            Call = np.random.choice(2, 1, p = [prob, 1 - prob])[0]
            if (Call):
                return ["Call", pars[2], pars[3]]

        if (bluff):
            # print('I am not bluffing')
            # print('total dice is {}'.format(pars[1]))
            # print('The original is {} and {}'.format(pars[2],pars[3]))
            newDiceValue = np.random.choice(np.arange(1,7),1)[0]
            newDiceQuantity = pars[3] + 1
            # print('the new value and quantity of bluff {} and {}'.format(newDiceValue,newDiceQuantity))
            if (newDiceQuantity >  pars[1]):
                newDiceQuantity = pars[3]
                if (pars[2] == 6):
                    # print('return call')
                    return ["Call", pars[2], pars[3]]
                else:
                    newDiceValue = np.random.choice(np.arange(pars[2], 7), 1)[0]
                    # print('try new value which is {}'.format(newDiceValue))


        else:
            # print('I am not bluffing')
            # newDiceValue = ptable.idxmax(axis="columns").values[0]
            prbOrderPtable = ptable.sort_values(by='Total', ascending=False, axis=1)
            indexListProTable = list(prbOrderPtable.columns.values)
            # print(indexListProTable)
            newDiceValue = indexListProTable[0]
            newDiceQuantity = pars[3] + 1
            # print('the new value and quantity of not bluff{} and {}'.format(newDiceValue, newDiceQuantity))
            if (newDiceQuantity >  pars[1]):
                newDiceQuantity = pars[3]
                i = 1

                while (newDiceValue <= pars[2]):
                    newDiceValue = indexListProTable[i]
                    # print('i am stop when i {}'.format(i))
                    # print('i is {}'.format(i))
                    # print(newDiceValue)
                    i += 1
                    if i == 6:
                        return ["Call", pars[2], pars[3]]

        # print('previous is {} and {}'.format(pars[2], pars[3]))
        # print('Now is {} and {}'.format(newDiceValue,newDiceQuantity))
        return ["Raise", newDiceValue, newDiceQuantity]

    return buildAgentFn

def updateQ (play, Qmat, reward, alpha=0.1, discount=0.9):

    for k in range(1, play.shape[0]):
        currState = play.loc[play.index[k],'yState']
        prevState = play.loc[play.index[k-1],'yState']
        action = play.loc[play.index[k],'yAction']
        Qmat.loc[int(float(prevState)), action] = (1 - alpha) * Qmat.loc[int(float(prevState)), action] + alpha * (reward.loc[int(float(prevState)), (action, int(float(currState)))] + discount * max(Qmat.loc[int(float(currState)),]))
        #Qmat.loc[prevState, action] = (1 - alpha) * Qmat.loc[prevState, action] + alpha * (reward.loc[prevState, (action, currState)] + discount * max(Qmat.loc[prevState,:]))

    return Qmat

def playLiarsDice(agents,players = 4, numDice = 6, auto = True, Qmat = np.array([]), train = True, printTrans = False):
    ndice = np.array(np.repeat(numDice, players))

    playersLeft = sum(ndice > 0)
    gameStates = generateGameStates(players, numDice)


    rewardEach = generateRewardMatrix(gameStates)

    raiselist = np.tile(np.array(['Raise']),rewardEach.shape[0])
    calllist = np.tile(np.array(['Call']),rewardEach.shape[0])
    colnamelist = np.hstack((raiselist,calllist))

    numraise = np.arange(rewardEach.shape[0])
    numcall= np.arange(rewardEach.shape[0])
    col2namelist = np.hstack((numraise,numcall))

    colarrays = [colnamelist,col2namelist]
    reward = pd.DataFrame(np.hstack((rewardEach,rewardEach)), columns=colarrays)

    if(Qmat.size == 0):
        Qmat = pd.DataFrame(np.zeros((reward.shape[0],2)),columns=['Raise','Call'])

    ctrl = np.random.choice(np.arange(0, players), 1)[0]
    play = pd.DataFrame(columns=['yCtrl', 'yState', 'yAction'])


    while(playersLeft > 1):
        results = liarsDiceRound(players=players, control=ctrl, playerDiceCount=ndice, gameStates=gameStates,reward=reward, Qmat=Qmat, agents=agents, a=not(auto))

        for k in range(len(ndice)):
            ndice[k] = ndice[k] + results[0][k]
            # if ((ndice[k] == 0) and (results[0][k] == -1)):
            #     print("player {} is out of the game\n".format(k))
            if (results[0][k] == -1) :
                ctrl = k
                while (ndice[ctrl] == 0):
                    ctrl = ctrl % (players) + 1
                    if (ctrl == players):
                        ctrl = 0
                    # if (ctrl == players):
                    #     ctrl = 0
                # print(' {} in control due to lose dice'.format(ctrl))

        playersLeft = sum(ndice > 0)
        # if (playersLeft == 1):
        #     print("player {}  won the game\n".format(np.where(ndice>0)[0]))

        play = pd.concat([play,results[1]],axis=0,ignore_index=True)

    if (printTrans):
        # play.df = play
        print(play)
    if(train):
        Qmat = updateQ(play,Qmat,reward)
        # print(Qmat)


    return [np.where(ndice > 0)[0], Qmat]


agent0 = buildAgent([0, 1], method="Qdecide")

agent10 = buildAgent([1, 0], method="trueProb")
agent11 = buildAgent([1, 0], method="trueProb")
agent12 = buildAgent([1, 0], method="trueProb")
agent13 = buildAgent([1, 0], method="trueProb")


agent1 = buildAgent([1,0], method="random")
agent2 = buildAgent([1,0], method="random")
agent3 = buildAgent([1,0], method="random")
agent4 = buildAgent([1,0], method="random")



agent5 = buildAgent([1, 0], method="randomV2")
agent6 = buildAgent([1, 0], method="randomV2")
agent7 = buildAgent([1, 0], method="randomV2")
agent8 = buildAgent([1, 0], method="randomV2")

agents = [agent0, agent1,agent2]
Qmat = np.array([])
its = 5

winners = np.zeros((its,))
times = 0
proplist = []

a = generateGameStates(2,2,1)

print(a)
# for k in tqdm (range(its)):
#     out = playLiarsDice(agents=agents, players=len(agents), numDice= 28, Qmat = Qmat, printTrans=False)
#     winners[k] = out[0]
#     # 21 29
#     Qmat = out[1]
#     if winners[k] == 0:
#         times += 1
#         proplist.append(times/(k+1))
#     else:
#         proplist.append(times / (k + 1))
#
#
#
# print(winners)
# print(proplist)
#
#
# unique, counts = np.unique(winners, return_counts=True)
# print(np.asarray((unique, counts)).T)
#
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
#
#



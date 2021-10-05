#########################################################################################
# Based on (truncated HTML so that it fits):
# https://towardsdatascience.com/evolution-of-a-salesman-a-complete
# -genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
#
# This is just the implementation of an old school GA. Nothing fancy or new, just your
# old friend.
#
#########################################################################################


import numpy as np
import sys, operator, random, pickle


# imports from parent folder
sys.path.append("..")
from Transformer import Transformer
from evaluate import predict_sequence
from datetime import datetime


class GeneticAlgorithm():

    def __init__(self, model, popSize, numInputs, seqLength, 
            eliteSize, mutationRate, generations):

        # save the parameters
        self.model, self.popSize, self.numInputs = model, popSize, numInputs
        self.seqLength, self.eliteSize = seqLength, eliteSize
        self.mutationRate, self.generations = mutationRate, generations

        # create an empty transformers, I cant just load the model
        self.t = Transformer(4,128,8,1024,25,25,300,300)
        # now load weights
        self.t.model.load_weights(model)
        self.model = self.t.model
        
        # log file where to save stuff
        self.logfile = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".txt"
        with open(self.logfile, 'w') as f:
            print('STARTING GA RUN '+self.logfile, file=f)

        # create initial random population
        self.population = self.initialPopulation()


    def run(self):

        for i in range(0, self.generations):
            pop = self.nextGeneration(i)
            self.population = pop


    def saveGenResults(self, results, genNum):

        data = [self.population, results]

        with open(f'gen{genNum:03}.p', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def nextGeneration(self, genNum):
        ''' Does all the GA operations to create a next generation'''

        performedExps = self.performExperiments(genNum)
        popRanked = self.rankExperiments(performedExps)
        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(selectionResults)
        children = self.breedPopulation(matingpool)
        nextGeneration = self.mutatePopulation(children)

        bestExperimentIndex = popRanked[0][0]
        bestExperiment = self.population[bestExperimentIndex]
        avgf = np.mean(popRanked, axis=0)[1]

        with open(self.logfile, 'a') as f:
            print("Gen "+str(genNum)+ " avg fitness: " + str(avgf), file=f)
            print("Gen "+str(genNum)+ " best fitness: " + str(popRanked[0][1]), file=f)
            print("Gen "+str(genNum)+ " best exp: " + str(bestExperiment), file=f)

        self.saveGenResults(performedExps, genNum)

        return np.array(nextGeneration)
    

    def createExperiment(self):
        # Generate a motor pattern with N numInputs
        
        motors = np.random.rand( self.numInputs )
        return motors


    def initialPopulation(self):
        # Create a first population with popSize elements of numInputs features

        population = []

        for i in range(0, self.popSize):
            population.append( self.createExperiment() )
        
        return population


    def performExperiments(self, genNum):
        '''Here is where we ask the Transformer to perform the experiments'''

        # First transform population into a numpy array of (popsize, seqlength, 25)
        trans_input = np.empty( [0, self.seqLength, 25] )

        for individual in self.population:
            ind_seq = np.repeat( individual.reshape((1,1,25)), self.seqLength, axis=1 )
            trans_input = np.concatenate( [trans_input, ind_seq], axis=0 )

        # Generate the experiments through the Transformer
        results = predict_sequence(self.model, trans_input)

        return results


    def rankExperiments(self, finishedExps):
        '''The experiments finishedExps have already been performed'''

        fitnessResults = {}
        for i in range(len(finishedExps)):
            fitnessResults[i] = self.calculateFitness(finishedExps[i])
        
        ranked = sorted(fitnessResults.items(), key = operator.itemgetter(1), 
                reverse = True)

        return ranked


    def calculateFitness(self, experiment):

        total_sum = np.sum(experiment)
        center_sum = np.sum(experiment[:,12])
        fitness = center_sum * 25 - total_sum 

        return fitness


    def selection(self, popRanked):
        ''' select individuals using elite population and roulette wheel.'''

        popRanked = np.array(popRanked)
        lowest = popRanked[-1,1]

        # if it is negative, we need to add it to keep all of them positive
        if lowest < 0:
            popRanked[:,1] -= lowest

        # [0] are the indexes, [1] their fitness values. here calculate the cum sum
        cumsum = np.cumsum(np.array(popRanked), axis=0).T[1]
        # calculate the percentile for each entry
        cum_perc = (cumsum*100)/cumsum[-1]
        selectionResults = []
        
        # elite goes straight
        for i in range(0, self.eliteSize):
            selectionResults.append(popRanked[i][0])
        # use roulette wheel to choose the rest
        for i in range(0, len(popRanked) - self.eliteSize):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= cum_perc[i]:
                    selectionResults.append(popRanked[i][0])
                    break

        return selectionResults


    def matingPool(self, selectionResults):
        ''' from population, choose the ones marked in the selection'''

        matingpool = []
        for i in range(0, len(selectionResults)):
            index = int(selectionResults[i])
            matingpool.append(self.population[index])
        return matingpool


    def breedPopulation(self, matingpool):
        ''' elite goes straight, the others are added after a crossover operation'''

        children = []
        length = len(matingpool) - self.eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, self.eliteSize):
            children.append(matingpool[i])
        
        for i in range(0, length):
            child = self.breed(pool[i], pool[len(matingpool)-i-1])
            children.append(child)

        return children


    def breed(self, parent1, parent2):
        ''' crossover operation'''

        child = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(len(parent1)):
            if i >= startGene and i <= endGene:
                child.append(parent2[i])
            else:
                child.append(parent1[i])
        
        return child


    def mutatePopulation(self, population):
        mutatedPop = []
        
        for ind in range(0, len(population)):
            mutatedInd = self.mutate(population[ind])
            mutatedPop.append(mutatedInd)
        return mutatedPop


    def mutate(self, individual):
        factor = 1/self.mutationRate

        for i in range(len(individual)):
            individual[i] += (random.random()-0.5) / factor
            individual[i] = max(0, individual[i]) # I could numpy clip...
            individual[i] = min(1, individual[i])

        return individual


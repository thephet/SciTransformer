#########################################################################################
#
# A GA with a fitness function adapted for the XOR.
# It will test every genome 4 times, one for each XOR case (00 11, 01, 10).
# It will consider the center cell as the output of the XOR.
# It will use the transformer (see the import of predict sequence) to simulate that 
# particular motor configuration, and then decide if it was good or not.
#
#########################################################################################

from GA import GeneticAlgorithm
import numpy as np
import sys, operator

sys.path.append("..")
from evaluate import predict_sequence


class GA_XOR(GeneticAlgorithm):

    def __init__(self, model, popSize, numInputs, seqLength,
            eliteSize, mutationRate, generations):

        super().__init__(model, popSize, numInputs, seqLength,
                eliteSize, mutationRate, generations)


    def rankExperiments(self, fes):
        '''The experiments finishedExps have already been performed'''

        fitnessResults = {}
        for i in range(len(fes[0])):
            fe = [ fes[0][i], fes[1][i], fes[2][i], fes[3][i] ]
            fitnessResults[i] = self.calculateFitness( fe )
        
        ranked = sorted(fitnessResults.items(), key = operator.itemgetter(1), 
                reverse = True)

        return ranked


    def calculateFitness(self, experiment):
        results = []

        for i in range(4):
            total_sum = np.sum(experiment[i])
            center_sum = np.sum(experiment[i][:,12])

            if i == 1 or i == 2: # center must be on (10, 01)
                fitness = center_sum * 25 - total_sum
                results.append(fitness)

            if i == 0 or i == 3: # center must be off (00, 11)
                fitness = total_sum - center_sum * 25 
                results.append(fitness)
        
        return min(results)


    def performExperiments(self, genNum):
        ''' XOR requires 4 experiments: 00, 01, 10, 11'''

        resultsXOR = []

        for i in ['00', '01', '10', '11']:

            # Transform population into a numpy array of (popsize, seqlength, 25)
            trans_input = np.empty( [0, self.seqLength, 25] )
            
            for individual in self.population:
                # change the columns as XOR inputs
                indi = self.adaptInput(individual, i)
                ind_seq = np.repeat( indi.reshape((1,1,25)), self.seqLength, axis=1 )
                trans_input = np.concatenate( [trans_input, ind_seq], axis=0 )

            results = predict_sequence(self.model, trans_input)
            resultsXOR.append( results )

        return resultsXOR


    def adaptInput(self, ind, inputs):

        if inputs == "00" or inputs == "01":
            ind = np.concatenate( [ np.zeros(5), ind ] )

        if inputs == "00" or inputs == "10":
            ind = np.concatenate( [ ind, np.zeros(5) ] )

        if inputs == "11" or inputs == "10":
            ind = np.concatenate( [ np.ones(5), ind ] )

        if inputs == "11" or inputs == "01":
            ind = np.concatenate( [ ind, np.ones(5) ] )

        return ind

#!/usr/bin/env python

"""
Provides RickTSP class for genetic Travelling Salesman Problem.
See the Jupyter notebook for a working example.
"""

# Import necessary libs
import numpy as np
import pandas as pd


class RickTSP:
    def __init__(self):
        self.towns = None
        self.population = None
        self.population_size = None

    def breed(self, num_gen=500):
        """Breed for the desired number of generations."""
        fittest_known = pd.DataFrame(
            index=np.arange(1), columns=np.arange(1)
        )  # track the distance of the best

        for i in range(num_gen):
            # Determine fitness of population:
            population_fitness = self.determine_fitness()
            # track fittest_known route
            fittest_known = fittest_known.append(population_fitness.nsmallest(1, 0))
            fittest_known = fittest_known.reset_index(drop=True)

            # Select Parents to breed with
            # The parents are the two fittest individualts in the population
            parents_index = population_fitness.nsmallest(2, 0).index
            self.parents = self.population.iloc[parents_index]
            # reset index of parents df... I still need to build in some logic to avoid
            # having to do this here(or at all...?)
            self.parents = self.parents.reset_index(drop=True)

            # Breed with the parents and mutate their offspring at a desired rate:
            mrate = 1
            self.breed_and_mutate(mrate)
            # in case any of the children are identical to already-existing routes: drop duplicates
            self.population = self.population.drop_duplicates()
            self.population = self.population.reset_index(drop=True)

            # determine population fittenss for population WITH children
            population_fitness = self.determine_fitness()
            # now look at the population(with the children included), and get rid of
            # the two worst options (i.e. where fitness is longest)
            # this is to keep the size of the population constant
            winners_index = population_fitness.nsmallest(self.population_size, 0).index
            self.population = self.population.iloc[winners_index]
            # reset index of parents df
            self.population = self.population.reset_index(drop=True)

        population_fitness = self.determine_fitness()

        return population_fitness

    def create_towns(self, random=1, size=10):
        """
        Description:
        Create a df of towns and their Coordinates by manual input or random generation.

        Parameters:
        if 'random'=1 then random generation is applied and 'size' must be specified (else assumed as =10)
        if random=0 then hardcoded town data is used

        Returns:
        df:towns - a dataframe of towns coordinates with town names as the row index and 'x' and 'y' as column indices.

        Rick Comments:
        Figure out why that NAN row is there are how to create a df in a way that does not require it to be deleted
        """

        if random == 1:
            # setup df and column index
            Towns_To_Visit = pd.DataFrame(index=np.arange(1), columns=["x", "y"])

            # Create the random cities (use name as row index) and their coordinates
            Num_Cities = size
            for i in range(Num_Cities):
                city_name = str("city" + str(float(i)))
                Towns_To_Visit.loc[city_name] = [
                    abs(int(np.random.randn() * 100)),
                    abs(int(np.random.randn() * 100)),
                ]

            # get rid of NAN row - ARG
            Towns_To_Visit.dropna(inplace=True)

        elif random == 0:
            Towns_To_Visit = pd.DataFrame(
                data=np.array([[12, 53], [12, 54], [912, 996], [8, 90], [3, 4]]),
                index=["Jhb", "Pta", "Cpt", "Dbn", "Blm"],
                columns=["x", "y"],
            )

        self.towns = Towns_To_Visit

    def clear_population(self, indicator):
        """
        Description:
        Clear all info regarding towns and their coordinates - this could be used for UI

        Parameters:
        if 'indicator'=1 then clear, if 'indicator'=0 then dont clear, else give error message

        Returns:
        nothing, but it can clear the self.towns df, so watch out!

        Rick Comments:
        figure out what the .iloc[0:0] method does not work.
        Play around with this and user input
        """

        # self.towns.iloc[0:0] #why does this not work?!
        if indicator == 0:
            print("The existing population was not cleared")
            pass
        elif indicator == 1:
            self.towns.drop(self.towns.index, inplace=True)
            print("The population has been cleared")
        else:
            print(
                "The argument you passed to the function `clear_population` was not understood"
            )

    def generate_TFM(self):
        """
        Description:
        generate a df with To-From data for all of the cities in the 'self.towns' df

        Parameters:
        none, but the 'self.towns' df should have some data in it

        Returns:
        df:TFM_return - a matrix containing the distances between the towns in the 'self.towns' df

        Rick Comments:
            -This is quite obvioulsy a bad way of doing things. imagine how big this df will get if you have 100 cities
                on top of ^^^ both sides of the matrix are filled, which means you are doing double the work all for easier indexing.
            -this method was used because I was following the textbook which solves things by hand.
            -redo this as a function which is passed two cities and then returns the distance between them(easy enough)
                use ^^^ function in subseqent code (will require a pretty big rework. Will be time intensive but not difficult)
            -...but is the logic above true? I refer to this matrix SO often! Maybe it does make sense to do all of the calcs
                and then just refer to them, instead of redoing the same calcs many times...? think about this
            -TFM is a pretty terrible name. Think more about names in future
        """

        # Create empty matrix of apropriate size
        TFM = pd.DataFrame(index=self.towns.index, columns=self.towns.index)
        TFM = TFM.fillna(0)

        # determine counter size from size of self.towns matrix
        count_to = self.towns.shape[0]

        # run counter. i is index, j is column - with a bit of effoft this computation
        # can be halved, as the "top" half of the TFM is just a transposed "bottom" half
        for i in range(count_to):
            for j in range(count_to):

                # determine X and Y distances between given points
                xDist = abs(
                    (self.towns.loc[TFM.index[i]]["x"])
                    - (self.towns.loc[TFM.columns[j]]["x"])
                )
                yDist = abs(
                    (self.towns.loc[TFM.index[i]]["y"])
                    - (self.towns.loc[TFM.columns[j]]["y"])
                )

                # determine direct distance between given points(hypoteneus)
                distance = np.sqrt((xDist ** 2) + (yDist ** 2))

                # set given cell of TFM matrix to the calculated distance
                TFM.iloc[i][j] = distance

        self.TFM = TFM

    def run_random(self):
        """
        Description:
        generates a 'tour' of all cities in random order. Visiting each city only once,
        except for the final return to the start city

        Parameters:
        none, but requires the 'self.towns' df to be populated

        Returns:
        df:randomized - a single row df representing a single 'tour'

        Rick Comments:
            -I am pretty sure that my understanding of df's and how to work with them
            is serioulsy limmited. There MUST be a better way to do this - how can I append a series to a df without converting the series to a df?
        """

        randomized = self.towns.sample(self.towns.shape[0])
        # determine the home_town
        home_town = randomized.index[0]

        # make df out of randomly aranged index
        randomized = pd.DataFrame(randomized.index)

        # add the return journey
        add_at = randomized.shape[0]
        randomized.loc[add_at] = home_town
        randomized = randomized.transpose()
        return randomized

    def generate_population(self, size):
        """
        Description:
        Generate starting population by creating a df that contains a number of random 'tours'

        Parameters:
        int:size - how big do you want the population to be?

        Returns:
        df:popu - a df containing a 'size' number of tours

        Rick Comments:
            This is SUCH a great place to incorporate the Nearest Neighbor Heuristic!
            Don't just take a random starting pop, rather start with the results of the NNH!
        """

        self.population_size = size

        count_cities = self.towns.shape[0] + 1
        popu = pd.DataFrame(index=np.arange(1), columns=np.arange(count_cities))

        size_of_pop = size
        for i in range(size_of_pop):
            result = self.run_random()
            popu = popu.append(result)
            popu = popu.reset_index(drop=True)

        popu.drop(popu.index[0], inplace=True)
        popu = popu.reset_index(drop=True)
        self.population = popu

    def determine_fitness(self):
        """
        Description:
        calculate the fitness of a population that is passed to it

        Parameters:
        df:pop

        Returns:
        df:pop_fitness

        Rick Comments:
            -if you get rid of the TFM then this function will be significantly influenced.
            Consider pro's and con's
        """

        # Create a df where fitness of pop can be stored
        fitness = pd.DataFrame(index=np.arange(1), columns=np.arange(1))

        # Determine size of population
        pop_size = self.population.shape[0]

        # loop over each tour in the population
        for j in range(pop_size):
            individual = self.population.iloc[[j]]
            individual_size = (
                individual.shape[1] - 1
            )  # determine how many stops there are along the way (the -1 is to disregard the return journey)

            individual_fitness = 0
            # loop over all stops for the given tour
            for i in range(individual_size):
                current_town = individual.loc[j, i]
                next_town = individual.loc[j, i + 1]
                dist_to_next = self.TFM.loc[current_town, next_town]
                individual_fitness += dist_to_next

            fitness.loc[j] = individual_fitness

        pop_fitness = fitness
        return pop_fitness

    def mutate(self, route):
        """
        Description:
        randomly switch two cities in a given tour

        Parameters:
        df:route

        Returns:
        df:mutation

        Rick Comments:
            -it is possible for a city to replace itself and no mutation to occur - improve on this
                ^^ the more cities there are, the less chance there is of this happening
            -When trying to switch values I ran into some serious problems with referencing/viewing vs copying of values.
                I ended up (secretly) dropping to an array to do the manipulations before (secretly) returning back to a df.
                I believe that I know how to do this properly now, with the .copy() function, but I should get some advise
                    on best practices here.
        """

        # determine length of route
        route_len = route.shape[1]

        # determine positions to be switched
        pos1 = np.random.randint(1, route_len - 1)
        pos2 = np.random.randint(1, route_len - 1)

        # sectretly drop down to an array in numpy because I dont know how to do this in a df
        route = np.array(route)

        # make the switch
        route[:, [pos1, pos2]] = route[:, [pos2, pos1]]

        # secretly convert back to a df before anybody realizes
        route = pd.DataFrame(route)

        mutation = route
        return mutation

    def breed_and_mutate(self, mrate=0.7):
        """
        Description:
        Crossover/Breed two 'parent' tours

        Parameters:
        int:mrate - the rate at which mutation should occur.
        default set to 0.7 becuase I want it to happen a lot

        Returns:
        none, but transforms the 'population' df

        Rick Comments:
        -I AM SO ASHAMED OF THE MESS I MADE HERE! I NEED TO LEARN HOW TO DO THIS BETTER AND THEN REDO IT! I HAVE SO MUCH TO LEARN!
            If anybody ever goes through this code... I am sorry for the torture that I have put you through!

        -Once again there is a chance of the start_point=end_point and nothing happening; no time to add logic now
        -How could I have worked with the parent df more elegantly?
            -how could I have looped twice between the relevant series in the df without repeating the loop?
        -a lot of repetitive loops are used!
            Think of how you could have user a 'derailing' approach to insertion instead starting from zero(see in-row comments)
            Think of how to loop between children istead of repeating code
        """

        # seperate parents from the parent df
        parent1 = self.parents.loc[0]
        parent2 = self.parents.loc[1]

        # randomly determine crossover start/stop points
        start_stop1 = np.random.randint(1, (len(parent1) - 2))
        start_stop2 = np.random.randint(1, (len(parent1) - 2))
        start_point = min(start_stop1, start_stop2)
        end_point = max(start_stop1, start_stop2)

        # prepare children df and clear the genes between the start and stop points
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(start_point, end_point):
            child1.loc[i] = "NO-GENE"

        for i in range(start_point, end_point):
            child2.loc[i] = "NO-GENE"

        # Insertion for child1 - I sort of cheated here by running through the parent genes from o-len instead of stating at 'start_point'
        for i in range(start_point, end_point):

            # Replace child1 empty genes with the gene from other parent
            for j in range(0, len(parent1)):

                if child1.str.contains(self.parents.loc[1, j]).any(axis=None):
                    # print('parent gene already in child - no crossover')
                    pass
                else:
                    # print('parent gene not in child - crossover')
                    child1[i] = self.parents.loc[1, j]
                    break

        # mutation for child1 (if triggered by mrate)
        if np.random.random() < mrate:
            # pass the first child through for mutation
            child1 = pd.DataFrame(child1).transpose()
            mutant1 = self.mutate(child1)

            # append the mutant to the population table
            self.population = self.population.append(mutant1)
            self.population = self.population.reset_index(drop=True)

        # if no mutation then append the child (this is now the child without mutation)
        else:
            self.population = self.population.append(child1)
            self.population = self.population.reset_index(drop=True)

        # I now repeat everythig for child2... there is definitely a better way of doing this!

        # Insertion for child2- I sort of cheated here by running through the parent genes
        # from o-len instead of stating at 'start_point'
        for i in range(start_point, end_point):

            # Replace child2 empty genes with the gene from other parent
            for j in range(0, len(parent1)):

                if child2.str.contains(self.parents.loc[0, j]).any(axis=None):
                    # print('parent gene already in child - no crossover')
                    pass

                else:
                    # print('parent gene not in child - crossover')
                    child2[i] = self.parents.loc[0, j]
                    break

        # mutation for child2 (if triggered by mrate)
        if np.random.random() < mrate:
            # pass the second child through for mutation
            child2 = pd.DataFrame(child2).transpose()
            mutant2 = self.mutate(child2)

            # append the mutant to the population table
            self.population = self.population.append(mutant2)
            self.population = self.population.reset_index(drop=True)

        # if no mutation then append the child (this is now the child without mutation)
        else:
            self.population = self.population.append(child2)
            self.population = self.population.reset_index(drop=True)

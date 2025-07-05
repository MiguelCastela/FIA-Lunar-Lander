import random
import copy
from re import T
from tracemalloc import stop
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
#PROB_CROSSOVER = 0.9

NUM_EVALS = 3
  
#PROB_MUTATION = 1.0/GENOTYPE_SIZE

PROB_MUTATION = 0.05
PROB_CROSSOVER = 0.5

STD_DEV = 0.1

ELITE_SIZE = 1



def network(shape, observation, ind):
    """
        In:
            shape: Number of neurons in each layer
            observation: Current observation of the whole environment
            ind: Genotype of the individual 
    """
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    #print(legs_touching)
    if legs_touching and on_landing_pad and stable:
        return True
    return False


def objective_function(observation):
    # Extract all state variables
    x = observation[0]  # horizontal position
    y = observation[1]  # vertical position
    vx = observation[2]  # horizontal velocity
    vy = observation[3]  # vertical velocity
    theta = observation[4]  # angle
    vtheta = observation[5]  # angular velocity
    contact_left = observation[6]
    contact_right = observation[7]
    
    
    # Define boolean conditions first (and convert to integers for math)
    stable_orientation = int(abs(theta) < 0.15)
    stable_angular_velocity = int(abs(vtheta) < 0.3)
    stable_horizontal_velocity = int(abs(vx) < 0.2)
    stable_vertical_velocity = int(abs(vy) < 0.3)
    safe_descent_rate = int(vy > -0.5)  # Less negative vertical velocity is better
    very_safe_descent = int(vy > -0.2)  # Extremely slow descent near landing
    centered_position = int(abs(x) < 0.2)
    legs_touching = int(contact_left == 1 and contact_right == 1)
    one_leg_touching = int((contact_left == 1 or contact_right == 1) and not legs_touching)
    near_ground = int(y < 0.5)
    very_near_ground = int(y < 0.2)  # Extremely close to ground
    stopped_vertical_velocity = int(abs(vy) < 0.1)
    stopped_horizontal_velocity = int(abs(vx) < 0.1)
    
    

    
    
    # Priorize stability if the environment is windy
    stability_multiplyer = 10 if ENABLE_WIND == True else 1
    
    # Height factor - stronger influence as we get closer to ground
    height_factor = 1.0 + (2.0/max(0.1, y))  # Increases as y decreases
    # 1. Stability component
    stability_score = (
        500 * stable_orientation +
        300 * stable_angular_velocity +
        300 * stable_horizontal_velocity
    ) * stability_multiplyer
    
    
    # 2. Positioning component - much stronger now
    positioning_score = (
        400 * centered_position + 
        300 * (1.0 - min(1.0, abs(x) / 0.5)) +  # Partial credit for getting closer
        # Encourage velocity toward center - stronger now
        200 * (1.0 - min(1.0, abs(vx + x*0.5) / 0.5))  # Reward velocity that counters position error
    ) * height_factor  # Much more important as we approach ground
    
    
    # 3. Descent rate control - new component with high priority
    descent_score = (
        400 * safe_descent_rate +
        600 * very_safe_descent +
        # Penalize high vertical velocity based on height
        300 * (1.0 - min(1.0, abs(vy) / 1.0))  # Reward slower descent in general
    ) * height_factor
    
    
    # 4. Landing component
    landing_score = 0
    stopped_reward = 0
    if near_ground:
        landing_score = (
            500 * legs_touching +
            200 * one_leg_touching +
            400 * stable_vertical_velocity +
            400 * centered_position  # Extra emphasis on position when landing
        )
        
        if very_near_ground and centered_position and stable_orientation and stable_horizontal_velocity and stable_vertical_velocity:
            # Reward stopped movement just hovering when everything else is good
            # Prevents endless hovering on the goal
            stopped_reward = (
                1000 * legs_touching +  # Much higher reward for actual landing
                500 * one_leg_touching +
                300 * stable_vertical_velocity +    
                100000 * int(stopped_horizontal_velocity and stopped_vertical_velocity and legs_touching) +  # Only give this reward when touching
                1000 * stopped_horizontal_velocity +
                1000 * stopped_vertical_velocity +
                -20000 * abs(vy) +  # Stronger penalty for vertical movement
                -1000 * abs(vx)
            )

    
    # Penalties for dangerous situations
    penalties = (
        -800 * int(abs(theta) > 0.6) +  # Severe angle penalty
        -500 * int(abs(vtheta) > 1.0) +  # High rotation speed penalty
        -1000 * int(vy < -1.0 and y < 1.0) +  # Much stronger crash penalty
        -800 * int(abs(x) > 0.8) * height_factor +  # Stronger penalty for being far from center
        -400 * int(abs(vx) > 0.8)  # Penalty for excessive horizontal speed
    )


    # Penalize based on steps used
    steps_penalty = -0.1 * (STEPS - 1) 
    penalties += steps_penalty


    total_score = (
        stability_score + 
        positioning_score + 
        descent_score + 
        landing_score * (stable_orientation * centered_position * safe_descent_rate) +
        stopped_reward
    )
    total_score += penalties
    
    # Return both a score and a boolean for successful landing
    return total_score, check_successful_landing(observation)


def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(prev_observation)


def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes

    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        total_fitness = 0
        success_count = 0
        num_trials = NUM_EVALS # Evaluate each individual multiple times
        
        for _ in range(num_trials):
            result = simulate(ind['genotype'], seed=None, env=env)
            total_fitness += result[0]
            success_count += int(result[1])
        
        ind['fitness'] = total_fitness / num_trials
        ind['success_rate'] = success_count / num_trials
                
        evaluatedQueue.put(ind)
    env.close()

    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop


def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def parent_selection_tournament_5(population):
    #TODO
    #Select an individual from the population
    # Tournament selection of 5
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    # ONLY REVERSE=TRUE IF THE OBJECTIVE FUNCTION IS MAXIMIZATION
    tournament.sort(key = lambda x: x['fitness'], reverse=True)
    return copy.deepcopy(tournament[0])
    
    """ 
        Choose the best with probability P
        Choose the second best with probability P*(1-P)
        Choose the third best with probability P*(1-P)*(1-P)
    """
    prob = 0.9 # (P)
    p = random.random() # [0, 1]
    # print(p, prob)
    if p < prob:
        return copy.deepcopy(tournament[0])
    elif p < prob*(1-prob):
        return copy.deepcopy(tournament[1])
    elif p < prob*(1-prob)*(1-prob):
        return copy.deepcopy(tournament[2])    

    return copy.deepcopy(random.choice(tournament))
    
    # RANDOM SELECTION
    return copy.deepcopy(random.choice(population))

def parent_selection_roulette(population):
    #TODO
    #Select an individual from the population
    # Roulette wheel selection
    total_fitness = sum([ind['fitness'] for ind in population])
    if total_fitness == 0:
        return copy.deepcopy(random.choice(population))
    
    # Calculate the probability of each individual
    probabilities = [ind['fitness']/total_fitness for ind in population]
    
    # Select an individual based on the probabilities
    selected_individual = random.choices(population, weights=probabilities, k=1)[0]
    
    return copy.deepcopy(selected_individual)

def parent_selection_tournament_2(population):
    #TODO
    #Select an individual from the population
    # Tournament selection of 2
    tournament_size = 2
    tournament = random.sample(population, tournament_size)
    # ONLY REVERSE=TRUE IF THE OBJECTIVE FUNCTION IS MAXIMIZATION
    tournament.sort(key = lambda x: x['fitness'], reverse=True)
    return copy.deepcopy(tournament[0])

PARENT_SELECTION = parent_selection_tournament_5


def crossover(p1, p2):
    #TODO
    #Create an offspring from the individuals p1 and p2
    # Single point crossover
    
    # Select a random point in the genotype
    cross_point = random.randint(0, GENOTYPE_SIZE-1)
    # Create the offspring 
    # Deep copy the genotype of p1 until the cross point
    genotype_slice_1 = copy.deepcopy(p1['genotype'][:cross_point])
    # Deep copy the genotype of p2 from the cross point
    genotype_slice_2 = copy.deepcopy(p2['genotype'][cross_point:])
    
    genotype_offspring = genotype_slice_1 + genotype_slice_2
    # Create the offspring
    offspring = {'genotype': genotype_offspring, 'fitness': None}
    # Return the offspring
    return offspring
      
    return p1


def mutation(p):
    #TODO
    #Mutate the individual p
    # Gaussian random mutation from N(-STD_DEV, STD_DEV)
    for i in range(len(p['genotype'])):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
            p['genotype'][i] = max(-1, min(1, p['genotype'][i]))
    return p    

    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    


def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                #p1 = parent_selection_tournament_5(population)
                #p2 = parent_selection_tournament_5(population)
                p1 = PARENT_SELECTION(population)
                p2 = PARENT_SELECTION(population)
                
                ni = crossover(p1, p2)

            else:
                ni = parent_selection_tournament_5(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests


def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests


if __name__ == '__main__':
    evolve = True
    evolve = False
    render_mode = None
    render_mode = 'human'
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(30):    
            #random.seed(seeds[i])
            random.seed(964)
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #validate individual
        bests = load_bests('log0.txt')
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = 1000

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
            print(f, s)
        print(fit/ntests, success/ntests)


        '''
        56653.992 0.154
        74103.92 0.295
        61151.43 0.183
        146582.12 0.969

        43808.66 0.143
        148110.92 0.972
        58213.836 0.17


        
        Experiencia 8
        1 trial 51562.22 0.144
        2 trials 15531.12 0.046
        3 trials 128967.16 0.838   //mega outlier
        4 trials 
        5 trials 82988.3 0.348
        '''
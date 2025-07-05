import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
RENDER_MODE = "human"
EPISODES = 1000

VERBOSE = True
def v_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)



Y = 2.0999999999999996
V_Y = 0.3
V_X = 0.3
V_THETA = 1.3
THETA = 0.55
X = 0.1
Y_WIND = 0.1
X_BOUNDARY = 1.0
MIN_THETA = 0.42999999999999994


Y = 1.5
Y_WIND = 0.2
V_Y = 0.2
V_X = 0.2
V_THETA = 0.3
THETA = 0.15
X = 0.2

env = gym.make("LunarLander-v3", render_mode = RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
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
 
    if legs_touching and on_landing_pad and stable:
        v_print("✅ Aterragem bem sucedida!")
        return True

    v_print("⚠️ Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
def query_perceptions(observation):
    x = observation[0]
    y = observation[1]
    v_x = observation[2]
    v_y = observation[3]
    theta = observation[4]
    v_theta = observation[5]
    left_leg_touching = observation[6]
    right_leg_touching = observation[7]
    return {
        "x": x,
        "y": y,
        "v_x": v_x,
        "v_y": v_y,
        "theta": theta,
        "v_theta": v_theta,
        "left leg touching": left_leg_touching,
        "right leg touching": right_leg_touching
    }


#Actions
def exec_action(perceptions):
    X_BOUNDARY = 1.0  # Distance where we start caring about alignment
    MIN_THETA = 0.05  # Minimum threshold (~3 degrees) when x=0
    
    # Calculate dynamic theta: closer to center = stricter requirement
    x_percentage = min(1.0, abs(perceptions["x"]) / X_BOUNDARY)
    dynamic_theta = MIN_THETA + (THETA - MIN_THETA) * x_percentage

    #Stabilize the ship's orientation to keep it aligned with the vertical axis while landing
    if perceptions["left leg touching"] == 1 and perceptions["right leg touching"] == 0 and perceptions["theta"] > 0:
        return [0, -0.6]
    elif perceptions["right leg touching"] == 1 and perceptions["left leg touching"] == 0:
        return [0, 0.6]
    elif perceptions["left leg touching"] == 1 and perceptions["right leg touching"] == 1:
        return [0, 0]
    
    #turn engines off if the ship goes too high
    elif perceptions["y"] > Y:
        return [0, 0]
    
    #Stabilize the ship’s orientation to keep it aligned with the vertical axis
    elif perceptions["theta"] <= -dynamic_theta and perceptions["v_y"] > -V_Y:
        return [0, -0.6]
    elif perceptions["theta"] > dynamic_theta and perceptions["v_y"] > -V_Y:
        return [0, 0.6]
    elif perceptions["theta"] > dynamic_theta and perceptions["v_y"] <= -V_Y:
        return [1, 0.6]
    elif perceptions["theta"] <= -dynamic_theta and perceptions["v_y"] <= -V_Y:
        return [1, -0.6]

    #Stabilize the ship’s angular velocity to keep it aligned with a stabilized orientation
    elif perceptions["v_theta"] <= -V_THETA and perceptions["v_y"] > -V_Y:
        return [0, -0.6]
    elif perceptions["v_theta"] > V_THETA and perceptions["v_y"] > -V_Y:
        return [0, 0.6]
    elif perceptions["v_theta"] > V_THETA and perceptions["v_y"] <= -V_Y:
        return [1, 0.6]
    elif perceptions["v_theta"] <= -V_THETA and perceptions["v_y"] <= -V_Y:
        return [1, -0.6]
    
    #:Stabilize the horizontal velocity, keeping it as low as possible using the secondary engines
    elif perceptions["v_x"] <= -V_X and perceptions["v_y"] > -V_Y:
        return [0, 0.6]
    elif perceptions["v_x"] > V_X and perceptions["v_y"] > -V_Y:
        return [0, -0.6]
    elif perceptions["v_x"] > V_X and perceptions["v_y"] <= -V_Y:
        return [1, -0.6]
    elif perceptions["v_x"] <= -V_X and perceptions["v_y"] <= -V_Y:
        return [1, 0.6]
    
    #Stabilize the horizontal position using the left and right engines
    elif perceptions["x"] == 0:
        return [0, 0]
    elif perceptions["x"] < -X:
        return [1, 0.6]
    elif perceptions["x"] > X:
        return [1, -0.6] 

    #(WIND OPTIMIZATION) : let the spaceship fall if low enough
    elif perceptions["y"] > Y_WIND and ENABLE_WIND == True:
        return [0, 0]    
    
    #Stabilize the vertical velocity, keeping it as low as possible using the main engine
    elif perceptions["v_y"] <= -V_Y:
        return [1, 0]
    elif perceptions["v_y"] > V_Y:
        return [0, 0]
    
    return [0, 0]


    
def reactive_agent(observation):
    ##TODO: Implemente aqui o seu agente reativo
    ##Substitua a linha abaixo pela sua implementação
    action = exec_action(query_perceptions(observation))
    return action 
    
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    v_print('observação:',observation)

    if keys[pygame.K_UP]:  
        action += np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action += np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action += np.array([0,1])

    return action
    
if __name__ == "__main__":
    print(f"""
            GLOBAL PARAMETERS:
            Y = {Y}
            V_Y = {V_Y}
            V_X = {V_X}
            V_THETA = {V_THETA}
            THETA = {THETA}
            X = {X}
            Y_WIND = {Y_WIND}
            X_BOUNDARY = {X_BOUNDARY}
            MIN_THETA = {MIN_THETA}        
            """)

    success = 0.0
    steps = 0.0
    for i in range(EPISODES):   
        st, su = simulate(steps=1000000, policy=reactive_agent)
        if su:
            steps += st
        success += su
        
        if su>0:
            v_print('Média de passos das aterragens bem sucedidas:', steps/(su*(i+1))*100)
        v_print('Taxa de sucesso:', success/(i+1)*100)
        
    print('Taxa de sucesso:', success/EPISODES*100)

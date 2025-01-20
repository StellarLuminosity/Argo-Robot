# Making quadrupeds Learning to walk: Step-by-Step Guide

 Add a compelling real-world example (e.g., a short video or GIF).
 Outline the excitement and potential applications (search-and-rescue, entertainment, etc.).


## Problem Overview
Dog, joystick -> objective tracking

## Ingredients and RL Framework
What is needed to create a control policy | What is a control policy
simulatore / environment (URDF)

photo e.g. input output to policy and robot, where is the neural network

## 4. Learning to Walk
### 4.1 Control Policy
- What is a policy?
- Neural network architecture
- Input/output visualization

### 4.2 Actions

The quadruped robot is equipped with 3 motors per leg, each controlled via position control. This setup results in a 12-dimensional action space, where each dimension corresponds to the reference position for one motor.

To simplify the learning process and "assist" the neural network, we opted to structure the action space as a residual added to a predefined homing position. The homing position represents the robot's default standing posture, with the legs arranged to ensure balance and stability. Instead of directly outputting absolute motor positions, the neural network predicts a residual adjustment relative to this homing posture.

This approach has two key benefits:

- **Simplified Exploration**: By constraining the network’s output to modify a stable baseline, the agent can focus on learning meaningful deviations, reducing the search space for valid actions.
- **Enhanced Stability**: The homing position acts as a natural fallback, preventing erratic movements during the early stages of training when the policy is still unrefined.

Here’s the implementation of the action transformation:

```python
action_total = self.q_homing + residual_action_nn
```

### 4.2 Observations

- ALCUNE SONO DIVERSE DAL TUO CODICE, PERÒ AVEVO UNA DOMANDA
- AGGIUNGIAMO INFO SU SCALING DELLE OBSERVATIONS QUI?

The observation space represents all the information that the robot can measure or estimate using sensors or sensor fusion techniques. This data is essential for the robot to understand its current situation, allowing it to make informed decisions and take the appropriate actions. A well-defined observation space is critical for the learning algorithm, as it provides the necessary context for the model to perform effectively.

In our case, the robot's observation space will be composed of both the robot's internal states and external inputs (i.e. user commands). 

The main components of the observation space taking into account internal states are as follows:

- **Base Linear Velocities**: $ Vx, Vy, Vz $

- **Base Rotational Velocities**: $ Wx, Wy, Wz $

- **Orientation Angles**: $ \textit{roll, pitch} $

- **Joint Positions**: $ q_{1..12} $

- **Joint Velocities**: $ \dot{q}_{1..12} $

- **Previous Actions**: $ a_{t-1} $

In addition to the robot's internal state, user commands are also incorporated into the observation space to allow for manual control. These commands come from an operator moving the robot through a joystick, so that their value will be in the range $[-1, 1]$. Thus, the observation space will be enlarged with the following user command inputs:

- **Reference Linear Velocities**: $Vx_{ref}, Vy_{ref}, Wz_{ref} \in [-1, 1]$

- **Reference Robot Altitude**: $z_{ref} \in [-1, 1]$ 

With all these components combined, the final observation space is of the following dimensionality: $ \mathbb{R}^{TBD} $  

### 4.3 Reward Design

The goal is to make our robot learning to walk following desired speed and altitude references given by an user. The core idea behind reinforcement learning is giving rewards when the agent behaves as expected, and punishing it when it behaves far from the desired behaviour.

So, the rewards chosen are:



### 4.4 Episode trmination condition

During training, episodes are terminated when specific criteria are met to ensure the robot remains in a healthy and functional state. The termination conditions include:

- $| \textit{roll} | < \textit{roll}_{\textit{min}}$: Robot roll is below a certain threshold.  
- $| \textit{pitch} | < \textit{pitch}_{\textit{min}}$: Robot pitch is below a certain threshold.  
- $z > z_{\textit{min}}$: Robot altitude is above a minimum value.  
- $\textit{steps} \geq \textit{max\_steps}$: Maximum number of steps reached.  

Here is an implementation for checking whether the robot is in a healthy state:

```python
# check whether robot current state is healthy
def is_healthy(self, obs, curr_step):
    
    roll = obs[xxx] # rad
    pitch = obs[xxx] # rad
    z = obs[xxx] # rad

    if (abs(roll)>self.roll_th or abs(pitch)>pitch_th or abs(z) < self.z_min or curr_step > self.max_steps):
        return False # dead
    else:
        return True # alive

```

### 4.5 Reset

Whenever a termination condition is met, the episode must be reset, and the robot should start over from the initial state. To encourage exploration and avoid overfitting, some randomness is introduced when reinitializing the robot's starting position.

Specifically, the initial joint positions and velocities are perturbed by adding small random noise:

$$
qpos = qpos_{init} + rand(low_{pos}, high_{pos})
$$

$$
qvel = qvel_{init} + rand(low_{vel}, high_{vel})
$$

Where:

- $\textit{qpos}$: Robot base pose and joint positions after reset.
- $\textit{qvel}$: Robot base velocities and joint velocities after reset.
- $\textit{rand}(\textit{low}, \textit{high})$: Uniform random noise between $\textit{low}$ and $\textit{high}$.
- $\textit{qpos}_{\textit{init}}$: Default positions.
- $\textit{qvel}_{\textit{init}}$: Default velocities.

Here is the code snippet for implementing this reset logic:

```python
# re-initialize robot after reset
qpos = self.qpos_init_sim + self.np_random.uniform(low=noise_pos_low, high=noise_pos_high, size=self.model.nq) 
qvel = self.qvel_init_sim + self.np_random.uniform(low=noise_vel_low, high=noise_vel_high, size=self.model.nv)
```

## 5. Training Process

- Core idea: snippet codice che mostra ciclo for: get_obs, step, reset
- PPO algorithm overview
- Training curves
- Common pitfalls
- Debug visualizations
- Magari dire qualcosa sulla questione dello scaling di azioni/rewards/azioni

## 6. From Simulation to Reality

While training and testing in simulated environments offers a fast and safe way to develop algorithms, real-world deployment often reveals significant discrepancies. This is known as **Sim2Real** gap. The discrepancies arise from the fact that simulations typically use idealized or simplified models of the environment and the robot. In reality, factors like sensor noise, unmodeled dynamics, imperfect actuators, environmental variabilities and physical wear are present.

Bridging this gap is crucial for deploying AI systems in robotics, as the algorithms need to generalize to new, unseen real-world conditions, especially in environments where real-world data is limited or too costly to acquire. Typical strategies include *"domain randomization"* or *"domain adaptation"*.

### 6.1 Domain randomization

Instead of performing time-consuming and expensive system identification or gathering extensive real-world data, domain randomization deliberately introduces random variations in the simulated environment’s parameters during training, allowing us to artificially create a diverse range of simulation environments. This strategy forces the agent to adapt to a broader set of possible conditions, which in turn helps the model generalize better to real-world scenarios, where conditions may vary due to factors like friction, mechanical noise, etc.

Several parameters in the simulation can be randomized, including:

- **Friction**: Varying the coefficient of friction on surfaces simulates different floor types, weather conditions, or wear on the robot’s legs. This allows the robot to learn strategies that adapt to a variety of environments.

- **Latency**: By randomly adjusting communication delays or actuator response times in the simulation, the robot can learn to handle real-world delays that may affect its control strategies.

- **Physical Parameters**: Modifying parameters such as mass, inertia, battery voltage, and motor friction helps the model account for real-world variations in these key aspects of the robot’s physical characteristics.

- **Sensor Noise**: Introducing random noise to sensory inputs mimics imperfections in real-world sensors, enabling the robot to better handle noisy or imprecise data during operation.

Basically, in training, every time there is a termination condition and the episode is reset, besides re-initializing the initial pose and velocities of the robot, there will be a re-initialization accounting also for domain randomization:

```python
# Perform randomization
env_friction = np.random.uniform(self.min_env_friction, self.max_env_friction)   
latency = np.random.uniform(self.min_latency, self.max_latency)      
mass = np.random.uniform(self.min_mass, self.max_mass)        
IMU_bias = np.random.uniform(self.min_IMU_bias, self.max_IMU_bias) 
IMU_std = np.random.uniform(self.min_IMU_std, self.max_IMU_std) 
```

By using domain randomization, robotic systems become more adaptable to the real world, and can handle scenarios that they were never directly trained on in simulation. However, a limitation of this approach is that the robot will learn a conservative policy able to generalize on different scenarios, rather than an optimal one tailored to specific conditions.

### 6.2 Adaptation strategies




## Key Works and Citations

- **Ashish Kumar (2022)**: [*Adapting Rapid Motor Adaptation for Bipedal Robots*](https://arxiv.org/pdf/2205.15299)
- **Ashish Kumar (2021)**: [*RMA: Rapid Motor Adaptation for Legged Robots*](https://arxiv.org/pdf/2107.04034)
- **Xue Bin Peng (2020)**: [*Learning Agile Robotic Locomotion Skills by
Imitating Animals*](https://arxiv.org/pdf/2004.00784)
- **Jie Tan (2018)**: [*Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*](https://arxiv.org/pdf/1804.10332)



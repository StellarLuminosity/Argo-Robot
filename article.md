# Making quadrupeds Learning to walk: Step-by-Step Guide

 Add a compelling real-world example (e.g., a short video or GIF).
 Outline the excitement and potential applications (search-and-rescue, entertainment, etc.).


## Problem Overview
Dog, joystick -> objective tracking

DICIAMO ANCHE DEL SALTO O NO? VEDI ANCHE REWARDS CHAPTER


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

- **Base Linear Velocities**: $ v_x, v_y, v_z $

- **Base Rotational Velocities**: $ w_x, w_y, w_z $

- **Orientation Angles**: $ \textit{roll, pitch} $

- **Joint Positions**: $ q_{1..12} $

- **Joint Velocities**: $ \dot{q}_{1..12} $

- **Previous Actions**: $ a_{t-1} $

In addition to the robot's internal state, user commands are also incorporated into the observation space to allow for manual control. These commands come from an operator moving the robot through a joystick, so that their value will be in the range $[-1, 1]$. Thus, the observation space will be enlarged with the following user command inputs:

- **Reference Velocities**: $v^{ref}_{x}, v^{ref}_{y}, w^{ref}_{z} \in [-1, 1]$

- **Reference Robot Altitude**: $z_{ref} \in [-1, 1]$ 

With all these components combined, the final observation space is of the following dimensionality: $ \mathbb{R}^{TBD} $  

### 4.3 Reward Design

The goal of the reward design is to guide the robot to walk effectively while adhering to user-specified references for speed and altitude. In reinforcement learning, the agent is encouraged to maximize its cumulative reward, which is designed to reflect the desired behavior. Rewards are given for achieving objectives, and penalties are applied when deviations occur. Below, we outline the specific reward terms used in our implementation, based on the provided code.

#### 1. **Linear Velocity Tracking Reward**

The robot is encouraged to track $v_x, v_y$ references commanded by the user.

$$
R_{lin\_vel} = \exp[-\|v^{ref}_{xy} - v_{xy}\|^2]
$$

Where:
- $v^{ref}_{xy} = [v^{ref}_{x}, v^{ref}_{y}]$ is the commanded velocity.
- $v_{xy} = [v_x, v_y]$ is the actual velocity.

#### 2. **Angular Velocity Tracking Reward**

The robot is encouraged to track $w_z$ reference commanded by the user.

$$
R_{ang\_vel} = \exp[-(w^{ref}_{z} - w_{z})^2]
$$

Where:
- $w_{cmd,z}$ is the commanded yaw velocity.
- $w_{base,z}$ is the actual yaw velocity.

#### 3. **Height Penalty**

The robot is encouraged to maintain a desired height as specified by the commanded altitude. A penalty is applied for deviations from this target height:

$$
R_{z} = (z - z_{ref})^2
$$

Where:
- $z$ is the current base height.
- $z_{ref}$ is the target height specified in the commands.

#### 4. **Pose Similarity Reward** ( --> PROVEREI A TOGLIERLA E VEDERE SE IMPARA)

To keep the robot's joint poses close to a default configuration, a penalty is applied for large deviations from the default joint positions:

$$
R_{pose\_similarity} = \|q - q_{default}\|^2
$$

Where:
- $q$ is the current joint position.
- $q_{default}$ is the default joint position.

#### 5. **Action Rate Penalty**

To ensure smooth control and discourage abrupt changes in actions, a penalty is applied based on the difference between consecutive actions:

$$
R_{action\_rate} = \|a_{t} - a_{t-1}\|^2
$$

Where:
- $a_t$ and $a_{t-1}$ are the actions at the current and previous time steps, respectively.

#### 6. **Vertical Velocity Penalty**

To discourage unnecessary movement along the vertical ($z$) axis, a penalty is applied to the squared $z$-axis velocity of the base when the robot is not actively jumping. The reward is:

$$
R_{lin\_vel\_z} = v_{z}^2
$$

Where:
- $v_{z}$ is the vertical velocity of the base.

#### 7. **Roll and Pitch Stabilization Penalty**

To ensure the robot maintains stability, a penalty is applied to discourage large roll and pitch deviations of the base. This reward is:

$$
R_{roll\_pitch} = roll^2 + pitch^2
$$

Where:
- $roll$ is the roll angle of the base.
- $pitch$ is the pitch angle of the base.

---

This design ensures that the robot learns a balanced policy that prioritizes tracking commands, maintaining stability, and acting smoothly while adhering to physical constraints.

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

### 6.2 Adaptation Strategies

To improve the generalization of the robot across various environments, the policy can be conditioned on environment parameters $(\mu)$. By doing so, the robot can adjust its actions based on both its internal state and the dynamics of the environment. However, in real-world scenarios, these environment parameters are not precisely known. If they were, the problem would already be solved. 

#### Latent Representation of Environment Parameters

Fortunately, in simulation, the environment parameters are fully available. In this way, at the beginning of each training episode, a random set of environment parameters $\mu$ is sampled according to a probability distribution $p(\mu)$. These parameters can include friction, latency, sensor noise and other factors that influence the dynamics. Predicting the exact system parameters $\mu$ is often unnecessary and impractical, as it may lead to overfitting and poor real-world performance. Instead, a low-dimensional latent embedding $z$ is used. Once $\mu$ is sampled, the environment parameters are encoded into a compact latent space $z$ using an encoder function $e$, represented as:

$$
z_t = e(\mu_t)
$$

Here, $z_t$ serves as a concise representation of the environment's dynamics. This latent variable is then fed as an additional input to the robot’s policy, enabling it to adapt its actions based on the environment:

$$
\pi(a_t \mid o_t, z_t)
$$

Where:
- $a_t$: Action to be taken by the robot.
- $o_t$: Observations.
- $z_t$: Latent encoding of the environment's dynamics.

#### Training the Policy

During training, both the encoder $e$ and policy $\pi$ are jointly optimized using gradient descent based on the reward signals, as a typical reinforcement learning problem. 

#### Real-World Deployment: Adaptation Module

In real-world deployment, the robot does not have access to the privileged environment parameters $\mu$. Instead, an *adaptation module* $(\phi)$ is employed to estimate the latent variable $\hat{z}_t$ online. This estimate is derived from the recent history of the robot's states $(x_{t-k:t-1})$ and actions $(a_{t-k:t-1})$:

$$
\hat{z}_t = \phi(x_{t-k:t-1}, a_{t-k:t-1})
$$

Unlike traditional system identification approaches that attempt to predict the precise environmental parameters $\mu$, this method directly estimates $\hat{z}_t$.

#### Training the Adaptation Module

The adaptation module $\phi$ is trained in simulation, where both the state-action history and the ground truth extrinsics vector $z_t$ are available. This is a typical supervised learning problem, where the objective is to minimize the **mean squared error (MSE)** between $\hat{z}_t$ and $z_t$:

$$
\text{MSE}(\hat{z}_t, z_t) = \| \hat{z}_t - z_t \|^2
$$

This training process ensures that the adaptation module learns to accurately predict $\hat{z}_t$ based on historical data.

#### Deployment

Once trained, the adaptation module and policy are ready for real-world deployment. The policy operates as follows:

$$
\pi(a_t \mid o_t, \hat{z}_t)
$$

The adaptation module $\phi$ runs asynchronously at a slower frequency, periodically updating $\hat{z}_t$. The policy uses the most recent $\hat{z}_t$ along with the current observations to determine the robot's actions. This design enables robust and efficient performance across diverse real-world environments while maintaining computational efficiency.

## 7. Key Works and Citations

- **Ashish Kumar (2022)**: [*Adapting Rapid Motor Adaptation for Bipedal Robots*](https://arxiv.org/pdf/2205.15299)
- **Ashish Kumar (2021)**: [*RMA: Rapid Motor Adaptation for Legged Robots*](https://arxiv.org/pdf/2107.04034)
- **Xue Bin Peng (2020)**: [*Learning Agile Robotic Locomotion Skills by
Imitating Animals*](https://arxiv.org/pdf/2004.00784)
- **Jie Tan (2018)**: [*Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*](https://arxiv.org/pdf/1804.10332)



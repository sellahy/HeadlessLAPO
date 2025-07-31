import gymnasium as gym
import numpy as np
from envs.gridworld import GridWorld
from src.gridworld.common.prepare_envs import get_eval_envs
from src.tiny_llama.config import Config as ModelConfig
from src.gridworld.common.transformer import Model
import torch
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from src.action_mapper import ActionMapper
import pdb


# ---- Utils begin ----

def goal_and_agent_pos_2_renderable_array(goal_pos : np.ndarray, agent_pos : np.ndarray, size : int) -> np.ndarray:
    """
    :param goal_pos: 1 dim np.ndarray with just 2 entries describing the indicies of the location of goal
    :param agent_pos: 1 dim np.ndarray with just 2 entries describing the indicies of the location of agent
    :param size: int describing width and height of the gridworld environment
    """
    # red is the color of the goal
    goal_col : np.array = np.array([255,0,0]) 
    # green is the color of the agent
    agent_col : np.array = np.array([0,255,0]) 

    # by default, all tiles have RGB value (255, 255, 255)
    grid : np.array = np.ones([size, size, 3]) * 255
    
    # color the goal
    grid[goal_pos[0], goal_pos[1]] = goal_col
    
    # color the agent
    grid[agent_pos[0], agent_pos[1]] = agent_col
    
    return grid

def renderable_array_2_goal_and_agent_pos(renderable_array : np.ndarray) -> (np.ndarray, np.ndarray, int):
    """
    convert array representing gridworld environment to just the goal position, agent position, and gridworld size

    :param renderable_array: np.ndarray representing a square, 2d gridworld environment 
                             in color. First dim is x coord, 2nd dim is y coord, and last 
                             dim is RGB value. Requires that shape is (size, size, 3) 
                             and rgb values are ints in range 0-255

    :return goal_pos: 1 dim np.ndarray with just 2 entries describing the indicies of the location of goal
    :return agent_pos: 1 dim np.ndarray with just 2 entries describing the indicies of the location of agent
    :return size: int describing width and height of the gridworld environment

    """
    assert len(renderable_array.shape == 3), f"expected renderable_array to have 3 dimensions, got {renderable_array.shape}"
    assert renderable_array.shape[0] == renderable_array.shape[1], f"expected grid of renderable_array to be square, but has shape {renderable_array.shape[0:2]}"
    assert renderable_array.shape[2] == 3, f"expected last dimension of renderable_array to be color channel with size 3, but got size {renderable_array.shape[2]}"
    # TODO: assert tha entries are ints with range 0 - 255

    # renderable_array has shape (size, size, 3). First dim is x coord, 2nd is y coord, 3rd is rgb value
    # goal_pos is position with red color. That is, (x,y) coord with rgb value == (255, 0, 0)

    goal_col : np.array = np.array([255,0,0])
    goal_pos : np.array = null
    agent_col : np.array = np.array([0,255,0])
    agent_pos : np.array = null
    size : int = renderable_array.shape[0]

    for i in range(renderable_array.shape[0]):
        for j in range(renderable_array.shape[1]):
            if (renderable_array[i,j] == goal_col).all():
                goal_pos = np.array([i,j])
            if (renderable_array[i,j] == agent_col).all():
                agent_pos = np.array([i,j])
            if agent_pos != null and goal_pos != null:
                break

    assert agent_pos != null, "no agent position was found in renderable_array"
    assert goal_pos != null, "no goal position was found in renderable_array"
    return goal_pos, agent_pos, size

def save_trajectory(agent_positions : list[np.ndarray], goal_pos : np.ndarray, size : int) -> None: # FIXME: need to readjust body now that agent_positions arg is an array of np.ndarrays
    """
    save trajectory represented as a dictionary. 
    
    agent_positions is required to have shape (num_timesteps, 2). The first dimension
    corresponds to the timestep of the agent position, the second dimension correponding
    to the position of the agent at a particular timestep. The first entry of the entry 
    at timestep t corresponds to the x coord of the agent at t, the second to the y 
    coord of the agent at t.
    
    goal_pos is required to have shape (2,). That is, it should have two entries, the first corresponding to the x coord of the goal, the second to the y coord of the goal

    the size is the width and height of the gridworld environment.
    """
    # assert len(agent_positions.shape) == 2, f"expected action_positions to be 2 dimensional, but found {len(agent_positions.shape)} dimensions"
    # assert agent_positions.shape[1] == 2, f"expected 2nd dim to have size 2, but has size {agent_positions.shape[1]}"

    # num_timesteps : int = agent_positions.shape[0]

    # return {"agent_positions" : agent_positions, "goal_pos" : goal_pos, "size" : size, "num_timesteps" : num_timesteps}

    np.save(os.path, np.vstack(agent_positions))

def get_model(config):
    # model setup
    model_config = ModelConfig(
        block_size=3 * config.seq_len + 5**config.action_seq_len,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_embd=config.d_model,
        bias=config.layer_norm_bias,
        rotary_percentage=config.rotary_percentage,
        parallel_residual=config.parallel_residual,
        shared_attention_norm=config.shared_attention_norm,
        _norm_class=config._norm_class,
        _mlp_class=config._mlp_class,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
    )

    model = Model(
        config=model_config,
        n_token=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
        action_seq_len=config.action_seq_len,
        num_states=config.grid_size**2,
    ).to(config.device)

    return model

# --- Utils end ---

@dataclass
class Config:
    # wandb params
    project: str = "Headless-AD"
    group: str = "gridworld-headless_ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "GridWorld"
    num_train_envs: int = 10_000
    learning_histories_path: Optional[str] = "trajectories"

    num_eval_envs: int = 100
    eval_every: int = 1_000
    log_every: int = 100

    num_train_steps: int = 30_000

    # Model Params
    seq_len: int = 100
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 64
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Training params
    batch_size: int = 64
    learning_rate: float = 3e-3
    beta1: float = 0.9
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    tau: float = 2.0
    sim_measure: str = "dot"
    get_action_type: str = "sample"
    use_action_set_prompt: bool = True
    rand_select_in_ucb: bool = True
    loss_type: str = "contrastive"
    rand_emb_type: str = "orthogonal"

    # New
    rotary_percentage: float = 1.0  # is default in the llama's configs at the bottom
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"

    # Device
    device: str = "cuda"
    autocast_dtype: str = "bf16"

    # Where to save data for experiment visualizations
    logs_dir: str = "logs"

    # Where to load model from
    load_dir = "model_checkpoints"

    # Where to save trajectories
    save_dir : str = "eval_trajs"

    # New
    action_seq_len: int = 3
    train_frac_acts: float = 0.4
    train_frac_goals: float = 0.85
    grid_size: int = 9
    num_episodes: int = 200
    q_learning_lr: float = 0.9933
    q_learning_discount: float = 0.6238
    check_cached: bool = False

    action_space_type: str = "for_headless"
    num_in_context_episodes: Optional[int] = None

    def __post_init__(self):
        self.job_type = self.action_space_type

        if self.num_in_context_episodes is None:
            self.num_in_context_episodes = 2 * self.num_episodes

        self.eval_seed = 1000 + self.train_seed

# the goal position. If None, will be chosen randomly.
goal_pos = np.array([2, 2])
# goal_pos = None

# actions in the env are sequences of left, right, up, down, and do nothing. action_seq_len determines how many actions in a row occur
# the amount of atomic actions constituting the action sequence.
action_seq_len = 3
# indices of action sequences that the environment will use.
# for action_seq_len 2, index 0 corresponds to [no-op, no-op], index 1 corresponds to [no-op, down], ... index 24 corresponds to [left, left]
# using all indices from 0 - 5^seq_len uses all actions
available_actions = np.arange(5**action_seq_len) 

# Create environment
env = gym.make(
    'GridWorld',
    goal_pos=goal_pos,
    available_actions=available_actions,
    action_seq_len=action_seq_len,
    render_mode="rgb_array"
)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=Config.eval_seed)
init_state = np.array([observation])

"""
10,000 seems to be num_train_steps
train_acts is a numpy array with shape (10000,50), all rows are the same at least at the first iter. Seems to contain arrays of indices of actions used for training
train_goals is a numpy array with shape (10000,2), all rows seem to be a 2d numpy array describing a goal pos. 
"""
# in their code, they split up the train and test actions and samples from the split arrays, mine just uses everything
eval_envs = get_eval_envs(config=Config, rng=np.random.default_rng(Config.train_seed), train_goals_split=available_actions, test_goals_split=available_actions, train_acts_split=available_actions, test_acts_split=available_actions) 

# in evaluate in context acts is a numpy.ndarray with shape (100,50). I assume this is (batch size, num actions per env)
key, (acts, goals) = list(eval_envs.items())[0] # just first first item instead of looping over everything

# load model for evaluation
model = get_model(config=Config)
model.load_state_dict(torch.load(os.path.join(Config.load_dir, "had_gridworld_step_30000.pt"), weights_only=True))
model.eval()

# Create action embeddings
act_mapper = ActionMapper(
    action_embed_dim=Config.token_embed_dim,
    num_actions=acts.shape[1],
    device=Config.device,
    sim_measure=Config.sim_measure,
    rand_emb_type=Config.rand_emb_type,
)
act_mapper.regenerate(seed=Config.eval_seed)

num_envs = 1 # in OG script, comes from SyncVectorEnv member var

# in evaluate_in_context, states is a torch.Tensor with shape (100,100). When istep is 1, states is all zeros with last row all 40's. I assume this corresponds to position of agent (40 would be index of middle of board)
states = torch.zeros(
    (Config.seq_len, num_envs), dtype=torch.long, device=Config.device
)
states[-1, :] = torch.from_numpy(init_state).to(Config.device).type(torch.long)
actions = torch.zeros( # in evaluate_in_context, actions is a torch.Tensor with shape (100, 100, 128). I assume this is (batch_size, seq_len, embed_dim). I assume each row is an action embedding of an action taken at a timestep for a batch. When istep is 1, this is all zeros
    (Config.seq_len, num_envs, Config.token_embed_dim),
    dtype=torch.float32,
    device=Config.device,
)
rewards = torch.zeros( # in evaluate_in_context, rewards is a torch.Tensor with shape (100, 100). I assume this is (batch size, seq_len). when istep is 1, this is all zeros.
    (Config.seq_len, num_envs), dtype=torch.long, device=Config.device
)
num_actions_per_env = torch.full(
    size=(num_envs,), fill_value=acts.shape[1], device=Config.device
)

actions_list = act_mapper._get_action_map_as_context( # in evaluate in context, this is a torch.tensor with shape (100, 50, 128). I assume this is (batch size, num actions per env, emded dim). I assume it stores the embeddings of each action in each row for the actions in each batch.
    num_actions_per_env=num_actions_per_env
)

# trajectory dictionary 
traj : list = [env.agent_pos] # start with the inital position recorded

breakpoint()
    
for istep in range(1, 10):
    """states.shape
torch.Size([64, 100]), presumbly 2nd index is config.seq_len
<class 'torch.Tensor'>
states is integers betwwen 0 and 80, presumably corresponding to indices of tiles of a 9 x 9 grid
(Pdb) actions.shape
torch.Size([64, 100])
<class 'torch.Tensor'>
actions is integers between 0 and 49, presumably corresponding to indices of 3 consecutive actions. The tensor presumably contains indices from a random selection of actions
(Pdb) rewards.shape
torch.Size([64, 100])
<class 'torch.Tensor'>
rewards is 1's and 0's, presumably cooresponding to ends of episodes

"""
    inp = (
    states.T[:, -istep:], # for each batch, gets the previous states of the agent. first it's many rows of [40] (one row for each batch), then it's many rows of [next_state_index, 40]
    actions.transpose(0, 1)[:, -istep:], # for each batch, gets the previous actions taken by the agent. First, it's many rows of [0 vector], then it's many rows of [0 vector, 0 vector]
    rewards.T[:, -istep:], # for each batch, gets the previous rewards of the agent. first it's many rows of [0] (one row for each batch), then it's many rows of [next_reward, 0]
    )
    # check for validity
    assert (istep < Config.seq_len and inp[0].shape[1] == istep) or (
        istep >= Config.seq_len and inp[0].shape[1] == Config.seq_len
    ), (
        inp[0].shape[1],
        istep,
    )
    
    # make prediction
    pred = model(*inp, actions_list=actions_list) # in evaluate in context, this is a torch.Tensor with shape (100, 1, 128). I assume this is (batch size, 1, embed dim)
    pred = pred[:, -1] # this gets rid of the 1 dim, so shape is now (100, 128) - 1 action embed prediction per batch
    


    action = env.action_space.sample() # random actions, not even using policy
    # TODO: get model to pick action. QUESTION: How to know if actions are better than random? probably need to limit duration of episode to size of grid and compare to random - expect less repeatition of actions by trained policy compared to random policy. Gridworld doesn't seem like a good showcase of Headless-AD's ability to reason about actions in context

    # want to measure avg return, avg regret, avg total regret, avg reward, avg min reward, avg max reward

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # record step
    traj.append(env.agent_pos)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

    breakpoint()

env.close()

traj = np.vstack(traj) # stack positions. Index with traj[i] to get (x,y) pos at timestep i
print(traj)

save_path = Path(Config.save_dir+"/eval_traj")
np.save(save_path, traj)






"""locals():
{'config': Config(project='Headless-AD', group='gridworld-headless_ad', job_type='for_headless', name=None, train_seed=0, eval_seed=1000, env_name='GridWorld', num_train_envs=10000, learning_histories_path='trajectories', num_eval_envs=100, eval_every=1000, log_every=100, num_train_steps=30000, seq_len=100, layer_norm_bias=True, token_embed_dim=128, d_model=512, num_layers=4, num_heads=64, dropout=0.0, attention_dropout=0.0, batch_size=64, learning_rate=0.003, beta1=0.9, weight_decay=0.0001, warmup_ratio=0.1, clip_grad_norm=5, tau=2.0, sim_measure='dot', get_action_type='sample', use_action_set_prompt=True, rand_select_in_ucb=True, loss_type='contrastive', rand_emb_type='orthogonal', rotary_percentage=1.0, parallel_residual=False, shared_attention_norm=False, _norm_class='FusedRMSNorm', _mlp_class='LLaMAMLP', device=device(type='cuda'), autocast_dtype='bf16', logs_dir='logs', action_seq_len=3, train_frac_acts=0.4, train_frac_goals=0.85, grid_size=9, num_episodes=200, q_learning_lr=0.9933, q_learning_discount=0.6238, check_cached=False, action_space_type='for_headless', num_in_context_episodes=400), 'accelerator': <accelerate.accelerator.Accelerator object at 0x7f01f0013cd0>, 'train_acts': array([[109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       ...,
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65]]), 'train_goals': array([[8, 6],
       [5, 1],
       [5, 5],
       ...,
       [8, 4],
       [5, 8],
       [1, 8]]), 'eval_envs': {'train_train': (array([[109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       ...,
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65]]), array([[1, 5],
       [1, 3],
       [8, 4],
       [1, 4],
       [2, 4],
       [1, 4],
       [7, 4],
       [4, 3],
       [7, 5],
       [8, 5],
       [2, 6],
       [3, 3],
       [1, 6],
       [7, 1],
       [7, 1],
       [7, 7],
       [2, 6],
       [7, 3],
       [5, 0],
       [0, 8],
       [1, 3],
       [0, 3],
       [4, 6],
       [7, 4],
       [3, 1],
       [1, 3],
       [4, 0],
       [8, 2],
       [8, 4],
       [8, 4],
       [3, 0],
       [5, 5],
       [1, 1],
       [8, 6],
       [6, 8],
       [1, 6],
       [1, 6],
       [1, 2],
       [5, 2],
       [0, 5],
       [5, 7],
       [1, 8],
       [8, 1],
       [1, 7],
       [0, 6],
       [8, 1],
       [3, 8],
       [2, 3],
       [4, 0],
       [6, 8],
       [4, 8],
       [2, 4],
       [1, 8],
       [3, 1],
       [2, 2],
       [8, 2],
       [0, 5],
       [5, 5],
       [2, 7],
       [7, 3],
       [0, 8],
       [7, 3],
       [5, 7],
       [4, 1],
       [3, 1],
       [0, 4],
       [3, 7],
       [1, 5],
       [7, 1],
       [8, 8],
       [8, 3],
       [4, 4],
       [4, 0],
       [7, 2],
       [8, 2],
       [8, 1],
       [6, 3],
       [6, 3],
       [5, 4],
       [7, 2],
       [0, 0],
       [4, 1],
       [6, 3],
       [2, 4],
       [3, 7],
       [3, 0],
       [1, 0],
       [2, 5],
       [2, 5],
       [7, 3],
       [4, 7],
       [8, 6],
       [3, 8],
       [4, 8],
       [1, 7],
       [0, 4],
       [2, 0],
       [1, 2],
       [5, 2],
       [3, 1]])), 'train_test': (array([[109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       ...,
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65],
       [109, 105, 112, ...,  83,  36,  65]]), array([[0, 7],
       [0, 7],
       [7, 0],
       [0, 7],
       [3, 2],
       [4, 5],
       [3, 2],
       [3, 4],
       [8, 0],
       [3, 2],
       [5, 3],
       [3, 2],
       [6, 0],
       [8, 7],
       [8, 0],
       [8, 7],
       [4, 5],
       [6, 5],
       [4, 5],
       [3, 4],
       [5, 3],
       [6, 5],
       [8, 7],
       [7, 6],
       [6, 2],
       [6, 0],
       [8, 0],
       [4, 5],
       [3, 6],
       [3, 6],
       [0, 7],
       [6, 2],
       [6, 5],
       [3, 2],
       [8, 0],
       [6, 5],
       [7, 6],
       [6, 5],
       [6, 2],
       [5, 3],
       [6, 0],
       [4, 5],
       [3, 4],
       [3, 2],
       [7, 0],
       [0, 7],
       [6, 0],
       [3, 4],
       [7, 6],
       [7, 0],
       [6, 5],
       [3, 6],
       [8, 7],
       [6, 0],
       [6, 5],
       [8, 0],
       [8, 7],
       [6, 2],
       [3, 4],
       [6, 0],
       [3, 6],
       [7, 6],
       [7, 6],
       [5, 3],
       [3, 6],
       [4, 5],
       [3, 6],
       [8, 0],
       [6, 0],
       [3, 6],
       [8, 0],
       [8, 0],
       [7, 0],
       [7, 6],
       [3, 4],
       [6, 5],
       [6, 2],
       [7, 0],
       [8, 7],
       [5, 3],
       [3, 2],
       [7, 0],
       [7, 0],
       [8, 7],
       [4, 5],
       [8, 7],
       [8, 7],
       [7, 6],
       [6, 5],
       [8, 0],
       [8, 0],
       [4, 5],
       [6, 0],
       [3, 2],
       [3, 6],
       [3, 4],
       [3, 2],
       [0, 7],
       [6, 2],
       [0, 7]])), 'perm_train_train': (array([[ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       ...,
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76]]), array([[4, 6],
       [1, 7],
       [5, 5],
       [8, 1],
       [1, 6],
       [4, 6],
       [8, 3],
       [3, 1],
       [2, 8],
       [3, 5],
       [5, 6],
       [7, 8],
       [1, 6],
       [1, 6],
       [5, 6],
       [8, 2],
       [0, 6],
       [2, 0],
       [8, 6],
       [6, 3],
       [1, 1],
       [8, 3],
       [1, 3],
       [0, 3],
       [5, 0],
       [7, 3],
       [2, 1],
       [5, 4],
       [1, 1],
       [5, 8],
       [6, 7],
       [6, 3],
       [6, 4],
       [8, 6],
       [8, 8],
       [8, 1],
       [0, 4],
       [8, 8],
       [7, 5],
       [3, 8],
       [5, 0],
       [2, 7],
       [2, 0],
       [3, 5],
       [2, 8],
       [2, 0],
       [0, 5],
       [1, 7],
       [0, 0],
       [0, 8],
       [5, 0],
       [7, 4],
       [2, 7],
       [4, 1],
       [5, 5],
       [3, 7],
       [4, 7],
       [1, 7],
       [6, 4],
       [1, 1],
       [4, 7],
       [3, 8],
       [2, 2],
       [5, 2],
       [5, 7],
       [3, 0],
       [4, 1],
       [1, 6],
       [5, 4],
       [1, 4],
       [6, 3],
       [5, 1],
       [4, 7],
       [2, 0],
       [1, 5],
       [7, 2],
       [2, 3],
       [7, 7],
       [3, 3],
       [5, 5],
       [2, 1],
       [0, 2],
       [7, 7],
       [2, 3],
       [8, 6],
       [0, 1],
       [5, 7],
       [7, 8],
       [3, 7],
       [7, 7],
       [1, 7],
       [3, 3],
       [7, 1],
       [5, 2],
       [4, 2],
       [3, 5],
       [0, 3],
       [2, 0],
       [4, 0],
       [2, 4]])), 'perm_train_test': (array([[ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       ...,
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76],
       [ 65, 109,  62, ...,  31,  45,  76]]), array([[3, 4],
       [8, 7],
       [6, 5],
       [6, 2],
       [0, 7],
       [0, 7],
       [5, 3],
       [3, 2],
       [7, 0],
       [5, 3],
       [6, 2],
       [6, 2],
       [6, 0],
       [6, 2],
       [7, 0],
       [3, 4],
       [6, 2],
       [8, 0],
       [6, 0],
       [5, 3],
       [3, 2],
       [4, 5],
       [7, 6],
       [6, 0],
       [6, 2],
       [7, 0],
       [6, 5],
       [3, 2],
       [7, 6],
       [3, 2],
       [7, 0],
       [0, 7],
       [3, 2],
       [7, 6],
       [3, 6],
       [3, 4],
       [8, 7],
       [6, 0],
       [0, 7],
       [3, 4],
       [6, 0],
       [5, 3],
       [5, 3],
       [6, 2],
       [3, 6],
       [7, 0],
       [8, 7],
       [8, 0],
       [6, 0],
       [3, 4],
       [6, 2],
       [8, 0],
       [8, 0],
       [7, 0],
       [3, 2],
       [3, 2],
       [4, 5],
       [5, 3],
       [3, 6],
       [0, 7],
       [3, 6],
       [4, 5],
       [3, 2],
       [0, 7],
       [5, 3],
       [8, 0],
       [6, 2],
       [5, 3],
       [0, 7],
       [0, 7],
       [3, 2],
       [4, 5],
       [3, 4],
       [0, 7],
       [0, 7],
       [4, 5],
       [6, 5],
       [6, 2],
       [6, 2],
       [5, 3],
       [3, 4],
       [3, 4],
       [8, 0],
       [7, 6],
       [6, 2],
       [7, 0],
       [8, 0],
       [3, 2],
       [5, 3],
       [5, 3],
       [6, 2],
       [3, 4],
       [5, 3],
       [3, 2],
       [6, 0],
       [5, 3],
       [7, 0],
       [7, 6],
       [8, 0],
       [0, 7]])), 'cut_test_train': (array([[117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       ...,
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12]]), array([[5, 4],
       [7, 5],
       [5, 6],
       [2, 6],
       [1, 1],
       [1, 1],
       [8, 2],
       [1, 8],
       [4, 4],
       [7, 1],
       [1, 2],
       [4, 4],
       [5, 6],
       [1, 0],
       [6, 7],
       [8, 6],
       [2, 6],
       [8, 2],
       [4, 3],
       [0, 0],
       [5, 4],
       [4, 3],
       [3, 1],
       [1, 5],
       [2, 3],
       [2, 1],
       [8, 1],
       [6, 1],
       [1, 4],
       [2, 7],
       [4, 6],
       [1, 1],
       [7, 1],
       [7, 1],
       [0, 4],
       [3, 0],
       [8, 6],
       [6, 3],
       [4, 4],
       [8, 3],
       [4, 2],
       [4, 8],
       [8, 8],
       [7, 4],
       [3, 0],
       [8, 4],
       [2, 8],
       [3, 1],
       [8, 4],
       [1, 2],
       [0, 3],
       [6, 7],
       [2, 7],
       [4, 1],
       [2, 7],
       [2, 3],
       [6, 1],
       [2, 8],
       [5, 8],
       [7, 4],
       [0, 6],
       [5, 1],
       [5, 0],
       [8, 2],
       [4, 3],
       [5, 8],
       [1, 1],
       [3, 3],
       [4, 0],
       [2, 0],
       [2, 6],
       [1, 0],
       [3, 5],
       [7, 8],
       [5, 4],
       [3, 1],
       [2, 1],
       [0, 6],
       [2, 8],
       [7, 8],
       [1, 8],
       [5, 5],
       [2, 8],
       [1, 2],
       [2, 1],
       [5, 8],
       [8, 8],
       [0, 1],
       [2, 5],
       [7, 4],
       [5, 2],
       [3, 5],
       [0, 0],
       [3, 1],
       [0, 1],
       [0, 5],
       [4, 0],
       [8, 8],
       [7, 2],
       [8, 6]])), 'cut_test_test': (array([[117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       ...,
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12],
       [117, 102,  17, ...,  67,   2,  12]]), array([[8, 7],
       [6, 2],
       [7, 0],
       [7, 6],
       [3, 6],
       [3, 6],
       [5, 3],
       [7, 0],
       [6, 0],
       [5, 3],
       [5, 3],
       [4, 5],
       [6, 2],
       [3, 2],
       [3, 2],
       [3, 2],
       [3, 4],
       [3, 2],
       [8, 7],
       [7, 0],
       [6, 0],
       [6, 5],
       [8, 7],
       [3, 6],
       [3, 6],
       [5, 3],
       [4, 5],
       [3, 2],
       [6, 2],
       [3, 4],
       [6, 2],
       [8, 7],
       [5, 3],
       [6, 2],
       [7, 0],
       [8, 0],
       [4, 5],
       [6, 2],
       [6, 0],
       [4, 5],
       [3, 4],
       [0, 7],
       [8, 7],
       [6, 2],
       [8, 7],
       [3, 2],
       [8, 0],
       [3, 4],
       [7, 6],
       [3, 4],
       [8, 0],
       [3, 6],
       [7, 6],
       [4, 5],
       [4, 5],
       [8, 0],
       [5, 3],
       [8, 7],
       [8, 7],
       [6, 2],
       [8, 0],
       [3, 6],
       [4, 5],
       [5, 3],
       [6, 2],
       [5, 3],
       [6, 5],
       [3, 4],
       [6, 2],
       [7, 6],
       [3, 6],
       [8, 0],
       [7, 0],
       [3, 4],
       [4, 5],
       [8, 7],
       [6, 0],
       [7, 6],
       [0, 7],
       [7, 6],
       [5, 3],
       [6, 0],
       [8, 7],
       [8, 0],
       [3, 4],
       [3, 2],
       [7, 6],
       [3, 6],
       [6, 2],
       [8, 0],
       [6, 2],
       [6, 0],
       [6, 0],
       [3, 2],
       [3, 6],
       [5, 3],
       [6, 5],
       [7, 0],
       [0, 7],
       [5, 3]])), 'test_train': (array([[117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       ...,
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114]]), array([[2, 7],
       [3, 5],
       [3, 1],
       [1, 8],
       [5, 2],
       [8, 4],
       [0, 1],
       [8, 1],
       [2, 1],
       [4, 2],
       [8, 3],
       [2, 2],
       [6, 8],
       [2, 3],
       [1, 8],
       [8, 2],
       [3, 0],
       [5, 5],
       [2, 5],
       [4, 3],
       [0, 5],
       [1, 4],
       [7, 2],
       [3, 7],
       [0, 4],
       [4, 6],
       [1, 8],
       [0, 6],
       [2, 8],
       [5, 1],
       [6, 4],
       [7, 5],
       [6, 3],
       [5, 1],
       [8, 4],
       [2, 6],
       [3, 7],
       [5, 5],
       [7, 5],
       [0, 2],
       [0, 0],
       [5, 5],
       [3, 8],
       [0, 8],
       [7, 4],
       [4, 1],
       [3, 1],
       [3, 0],
       [7, 8],
       [5, 5],
       [1, 7],
       [5, 4],
       [1, 8],
       [8, 1],
       [2, 6],
       [2, 1],
       [6, 8],
       [3, 3],
       [0, 5],
       [0, 2],
       [2, 2],
       [0, 0],
       [3, 7],
       [2, 1],
       [1, 0],
       [7, 4],
       [8, 4],
       [3, 3],
       [5, 2],
       [7, 8],
       [0, 5],
       [8, 1],
       [1, 7],
       [1, 6],
       [3, 3],
       [6, 3],
       [3, 5],
       [5, 0],
       [5, 6],
       [3, 5],
       [6, 4],
       [7, 3],
       [4, 4],
       [1, 7],
       [8, 5],
       [8, 4],
       [4, 1],
       [2, 1],
       [3, 5],
       [2, 6],
       [7, 5],
       [5, 0],
       [8, 3],
       [4, 0],
       [1, 3],
       [7, 1],
       [7, 8],
       [4, 1],
       [8, 5],
       [7, 3]])), 'test_test': (array([[117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       ...,
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114],
       [117, 102,  17, ...,  74,  27, 114]]), array([[7, 6],
       [6, 5],
       [5, 3],
       [6, 2],
       [5, 3],
       [7, 6],
       [3, 4],
       [7, 0],
       [6, 5],
       [7, 6],
       [8, 0],
       [7, 6],
       [4, 5],
       [3, 2],
       [3, 6],
       [6, 2],
       [8, 7],
       [4, 5],
       [3, 2],
       [3, 2],
       [8, 0],
       [3, 6],
       [4, 5],
       [7, 0],
       [6, 2],
       [8, 0],
       [0, 7],
       [7, 0],
       [7, 0],
       [6, 2],
       [4, 5],
       [0, 7],
       [4, 5],
       [0, 7],
       [5, 3],
       [8, 0],
       [8, 0],
       [3, 2],
       [3, 2],
       [3, 4],
       [0, 7],
       [0, 7],
       [6, 2],
       [3, 6],
       [3, 4],
       [3, 4],
       [0, 7],
       [3, 4],
       [4, 5],
       [0, 7],
       [3, 4],
       [3, 6],
       [5, 3],
       [6, 2],
       [7, 6],
       [7, 0],
       [5, 3],
       [3, 2],
       [6, 5],
       [7, 0],
       [6, 0],
       [4, 5],
       [3, 4],
       [6, 0],
       [5, 3],
       [8, 0],
       [7, 0],
       [8, 0],
       [8, 0],
       [5, 3],
       [8, 0],
       [8, 0],
       [6, 0],
       [6, 2],
       [3, 2],
       [4, 5],
       [8, 0],
       [7, 0],
       [4, 5],
       [4, 5],
       [8, 0],
       [3, 4],
       [8, 0],
       [4, 5],
       [3, 2],
       [0, 7],
       [8, 7],
       [6, 0],
       [3, 2],
       [3, 4],
       [8, 0],
       [3, 2],
       [7, 6],
       [3, 6],
       [0, 7],
       [3, 6],
       [0, 7],
       [4, 5],
       [4, 5],
       [3, 4]])), 'all_train': (array([[109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       ...,
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114]]), array([[3, 8],
       [7, 2],
       [5, 0],
       [6, 8],
       [7, 3],
       [2, 8],
       [4, 3],
       [3, 1],
       [2, 3],
       [5, 5],
       [5, 4],
       [3, 3],
       [2, 1],
       [6, 3],
       [8, 2],
       [2, 6],
       [6, 7],
       [4, 4],
       [0, 6],
       [7, 4],
       [2, 6],
       [3, 7],
       [1, 8],
       [8, 2],
       [0, 1],
       [1, 4],
       [8, 4],
       [5, 2],
       [4, 2],
       [2, 4],
       [7, 7],
       [0, 2],
       [1, 8],
       [3, 1],
       [8, 5],
       [0, 4],
       [7, 4],
       [1, 7],
       [7, 8],
       [5, 8],
       [1, 2],
       [5, 0],
       [2, 3],
       [1, 1],
       [1, 2],
       [1, 2],
       [3, 7],
       [8, 8],
       [2, 0],
       [5, 5],
       [1, 4],
       [1, 5],
       [2, 0],
       [1, 1],
       [2, 5],
       [0, 4],
       [4, 2],
       [2, 7],
       [1, 0],
       [7, 3],
       [5, 4],
       [0, 6],
       [4, 8],
       [6, 7],
       [5, 5],
       [5, 5],
       [1, 3],
       [1, 7],
       [4, 6],
       [2, 3],
       [0, 5],
       [3, 0],
       [8, 1],
       [6, 6],
       [2, 8],
       [4, 6],
       [1, 3],
       [0, 6],
       [5, 6],
       [7, 8],
       [7, 1],
       [8, 8],
       [0, 1],
       [0, 5],
       [4, 8],
       [4, 7],
       [5, 1],
       [8, 3],
       [8, 1],
       [0, 3],
       [2, 5],
       [1, 0],
       [0, 0],
       [6, 6],
       [2, 8],
       [7, 3],
       [4, 2],
       [8, 8],
       [8, 3],
       [5, 7]])), 'all_test': (array([[109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       ...,
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114],
       [109, 105, 112, ...,  74,  27, 114]]), array([[3, 6],
       [0, 7],
       [6, 0],
       [8, 7],
       [6, 5],
       [7, 0],
       [6, 0],
       [6, 5],
       [8, 7],
       [7, 6],
       [4, 5],
       [8, 0],
       [8, 7],
       [4, 5],
       [3, 2],
       [8, 0],
       [3, 6],
       [7, 6],
       [8, 0],
       [3, 6],
       [8, 0],
       [3, 2],
       [6, 5],
       [0, 7],
       [0, 7],
       [3, 6],
       [7, 0],
       [8, 7],
       [7, 6],
       [3, 2],
       [5, 3],
       [3, 6],
       [3, 6],
       [5, 3],
       [6, 5],
       [6, 5],
       [7, 6],
       [6, 5],
       [6, 5],
       [6, 5],
       [3, 4],
       [6, 0],
       [5, 3],
       [7, 0],
       [8, 7],
       [6, 5],
       [0, 7],
       [4, 5],
       [3, 4],
       [7, 6],
       [3, 2],
       [3, 2],
       [6, 2],
       [6, 5],
       [7, 6],
       [6, 2],
       [4, 5],
       [6, 2],
       [4, 5],
       [8, 7],
       [0, 7],
       [3, 2],
       [4, 5],
       [6, 5],
       [3, 4],
       [3, 6],
       [6, 2],
       [4, 5],
       [3, 2],
       [3, 4],
       [5, 3],
       [5, 3],
       [7, 0],
       [3, 4],
       [3, 4],
       [6, 2],
       [3, 4],
       [8, 7],
       [7, 6],
       [3, 6],
       [3, 6],
       [5, 3],
       [7, 6],
       [0, 7],
       [6, 5],
       [8, 7],
       [3, 2],
       [6, 5],
       [6, 0],
       [5, 3],
       [8, 0],
       [8, 0],
       [6, 0],
       [6, 0],
       [0, 7],
       [7, 6],
       [3, 4],
       [8, 0],
       [3, 4],
       [7, 6]]))}, 'q_learning_scores': {'train_train': 1.0, 'train_test': 1.0, 'perm_train_train': 0.98, 'perm_train_test': 1.0, 'cut_test_train': 0.94, 'cut_test_test': 0.99, 'test_train': 0.93, 'test_test': 0.89, 'all_train': 0.94, 'all_test': 0.85}, 'key': 'all_test', 'value': 0.34, 'dataset': <src.gridworld.common.data.SequenceDataset object at 0x7f02f30102e0>, 'shape0s': array([10000, 10000, 10000]), 'dataloader': <generator object next_dataloader at 0x7f01b43cbc80>, 'model_config': Config(block_size=425, n_layer=4, n_head=64, n_embd=512, rotary_percentage=1.0, parallel_residual=False, bias=True, dropout=0.0, attention_dropout=0.0, shared_attention_norm=False, _norm_class='FusedRMSNorm', _mlp_class='LLaMAMLP', n_query_groups=64, norm_eps=1e-05, intermediate_size=2048, condense_ratio=1), 'random_scores': {'train_train': 0.2, 'train_test': 0.29, 'perm_train_train': 0.25, 'perm_train_test': 0.25, 'cut_test_train': 0.21, 'cut_test_test': 0.39, 'test_train': 0.31, 'test_test': 0.36, 'all_train': 0.22, 'all_test': 0.34}, 'model': Model(
  (transformer): ModuleDict(
    (proj): Linear(in_features=128, out_features=512, bias=True)
    (reward_emb): Embedding(4, 128)
    (state_emb): Embedding(81, 128)
    (h): ModuleList(
      (0-3): 4 x Block(
        (norm_1): FusedRMSNorm()
        (attn): CausalSelfAttention(
          (attn): Linear(in_features=512, out_features=1536, bias=True)
          (proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm_2): FusedRMSNorm()
        (mlp): LLaMAMLP(
          (swiglu): SwiGLU(
            (w1): Linear(in_features=512, out_features=2048, bias=False)
            (w2): Linear(in_features=512, out_features=2048, bias=False)
            (w3): Linear(in_features=2048, out_features=512, bias=False)
          )
        )
      )
    )
    (ln_f): FusedRMSNorm()
    (out_proj): Linear(in_features=512, out_features=128, bias=True)
  )
), 'optim': AcceleratedOptimizer (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    initial_lr: 0.003
    lr: 0.0
    maximize: False
    weight_decay: 0.0001
), 'scheduler': <accelerate.scheduler.AcceleratedScheduler object at 0x7f01ba796f70>, 'act_mapper': ActionMapper(), 'best_opt_value': None, 'global_step': 1, 'batch_timer': <src.utils.misc.Timeit object at 0x7f01ba7961c0>, 'states': tensor([[57, 57, 58,  ..., 51, 41, 41],
        [25, 25, 32,  ..., 24, 25, 42],
        [31, 22, 40,  ..., 40, 31, 40],
        ...,
        [27,  9, 11,  ..., 46, 37, 37],
        [69, 69, 40,  ..., 43, 34, 35],
        [55, 72, 63,  ..., 24, 14, 25]], device='cuda:0'), 'actions': tensor([[39, 40, 45,  ..., 20,  4,  6],
        [28, 17, 13,  ...,  9, 23, 27],
        [18,  5, 12,  ..., 12,  5, 20],
        ...,
        [ 5, 46, 47,  ..., 35,  4,  3],
        [ 8, 13, 42,  ..., 25, 13,  7],
        [23,  1, 17,  ..., 44, 21, 25]], device='cuda:0'), 'rewards': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 1, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'), 'num_actions_per_env': tensor([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50], device='cuda:0'), 'act_mapper_timer': <src.utils.misc.Timeit object at 0x7f01ba365100>, 'action_embeds': tensor([[[-8.4907e-02, -4.4622e-02,  9.9621e-02,  ...,  2.5314e-02,
           1.5318e-02,  1.8582e-02],
         [ 8.1695e-02, -1.0327e-01,  5.6625e-02,  ..., -6.7547e-02,
          -1.4437e-01,  5.7191e-02],
         [-7.4834e-02, -1.7031e-01, -2.1793e-02,  ...,  5.4639e-02,
           1.2089e-01, -2.1210e-02],
         ...,
         [-1.5869e-02,  1.0151e-01, -3.9398e-02,  ..., -3.0799e-02,
           1.1656e-01,  1.0185e-02],
         [ 5.0944e-02,  3.6868e-02,  1.4852e-02,  ...,  6.5728e-02,
           1.9854e-01, -1.8787e-02],
         [-1.3129e-01,  1.4735e-01,  1.5945e-01,  ...,  6.3761e-02,
          -2.0732e-03,  1.6714e-02]],

        [[ 2.6509e-02, -2.4073e-02,  2.1827e-01,  ..., -1.1388e-01,
           7.4697e-02,  9.3815e-02],
         [ 1.3104e-01,  9.6293e-02,  1.1405e-02,  ..., -3.4735e-02,
           6.3965e-02, -5.1730e-02],
         [ 1.3141e-02,  3.1324e-02,  8.7598e-02,  ...,  3.2375e-02,
          -1.8412e-01, -4.2789e-02],
         ...,
         [-2.0105e-01,  3.3228e-02,  6.7171e-02,  ...,  4.3143e-02,
           1.8022e-01,  2.7382e-02],
         [-3.6965e-02,  4.0037e-02, -6.7072e-02,  ...,  4.1279e-03,
           7.9158e-02, -1.4921e-02],
         [-2.4315e-02, -6.9673e-02, -1.4470e-02,  ...,  1.5193e-01,
           1.1927e-01,  2.6032e-02]],

        [[-1.5425e-02, -2.2302e-02,  5.6525e-02,  ..., -1.0474e-01,
          -5.8329e-02,  6.4650e-02],
         [ 8.4568e-02,  2.6521e-02, -1.0032e-04,  ..., -1.0649e-02,
          -4.4705e-02,  7.8467e-02],
         [ 1.3327e-01,  9.4190e-02, -5.1129e-02,  ..., -1.1358e-01,
          -5.6747e-02, -5.5384e-02],
         ...,
         [ 1.3327e-01,  9.4190e-02, -5.1129e-02,  ..., -1.1358e-01,
          -5.6747e-02, -5.5384e-02],
         [ 8.4568e-02,  2.6521e-02, -1.0032e-04,  ..., -1.0649e-02,
          -4.4705e-02,  7.8467e-02],
         [-1.5869e-02,  1.0151e-01, -3.9398e-02,  ..., -3.0799e-02,
           1.1656e-01,  1.0185e-02]],

        ...,

        [[ 8.4568e-02,  2.6521e-02, -1.0032e-04,  ..., -1.0649e-02,
          -4.4705e-02,  7.8467e-02],
         [ 4.9767e-02,  5.4218e-02,  6.8235e-02,  ...,  3.1593e-02,
           9.9234e-03, -5.6154e-02],
         [ 5.1517e-03, -4.6956e-02,  5.7964e-02,  ...,  1.3221e-01,
          -5.2956e-02, -1.3833e-01],
         ...,
         [-7.3416e-02, -1.0413e-01,  1.4339e-01,  ..., -1.4656e-01,
          -5.1043e-02,  1.5564e-02],
         [ 5.0944e-02,  3.6868e-02,  1.4852e-02,  ...,  6.5728e-02,
           1.9854e-01, -1.8787e-02],
         [ 6.5746e-02,  5.1940e-02, -1.0514e-02,  ..., -6.7081e-02,
          -2.9261e-02,  7.2317e-02]],

        [[ 4.1113e-02, -8.5431e-02,  7.7799e-02,  ...,  1.6929e-01,
          -8.6387e-02,  2.8322e-02],
         [ 1.3141e-02,  3.1324e-02,  8.7598e-02,  ...,  3.2375e-02,
          -1.8412e-01, -4.2789e-02],
         [-1.8143e-02, -5.1010e-02,  2.7499e-03,  ...,  8.9079e-02,
           4.9910e-02,  5.1584e-02],
         ...,
         [ 4.1415e-02, -1.2662e-01,  1.5101e-02,  ..., -8.0342e-02,
           5.5061e-02,  7.2318e-03],
         [ 1.3141e-02,  3.1324e-02,  8.7598e-02,  ...,  3.2375e-02,
          -1.8412e-01, -4.2789e-02],
         [-6.3589e-02,  1.6398e-02, -2.7352e-02,  ..., -1.7188e-01,
           1.0772e-01,  2.4840e-02]],

        [[-3.6965e-02,  4.0037e-02, -6.7072e-02,  ...,  4.1279e-03,
           7.9158e-02, -1.4921e-02],
         [-9.5167e-02,  6.9592e-02, -1.8755e-02,  ..., -1.3356e-01,
           4.1287e-02,  6.4448e-02],
         [ 1.3104e-01,  9.6293e-02,  1.1405e-02,  ..., -3.4735e-02,
           6.3965e-02, -5.1730e-02],
         ...,
         [ 2.6733e-02, -2.8943e-02,  7.8055e-02,  ..., -1.2040e-01,
           1.4201e-01,  1.0661e-01],
         [-1.0206e-01, -1.9078e-01,  7.1386e-02,  ..., -1.1166e-01,
           4.8456e-02,  2.1568e-01],
         [ 4.1415e-02, -1.2662e-01,  1.5101e-02,  ..., -8.0342e-02,
           5.5061e-02,  7.2318e-03]]], device='cuda:0'), 'actions_list': tensor([[[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]],

        [[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]],

        [[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]],

        ...,

        [[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]],

        [[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]],

        [[-0.0570,  0.0480, -0.0380,  ...,  0.0241, -0.1779, -0.0069],
         [-0.0952,  0.0696, -0.0188,  ..., -0.1336,  0.0413,  0.0644],
         [ 0.0420, -0.0032,  0.0785,  ...,  0.0165,  0.1104,  0.0112],
         ...,
         [ 0.0052, -0.0470,  0.0580,  ...,  0.1322, -0.0530, -0.1383],
         [ 0.1233, -0.0410,  0.0262,  ..., -0.0657,  0.1565, -0.0235],
         [-0.1100, -0.1282, -0.0785,  ...,  0.1342,  0.0808, -0.0266]]],
       device='cuda:0'), 'pred_timer': <src.utils.misc.Timeit object at 0x7f01ba7d96a0>}
"""

"""globals()
{'__name__': '__main__', '__doc__': None, '__package__': 'src.gridworld.algorithms', '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f02f8bcdeb0>, '__spec__': ModuleSpec(name='src.gridworld.algorithms.headless_ad', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7f02f8bcdeb0>, origin='/fs/nexus-scratch/sellahy/headless-ad/src/gridworld/algorithms/headless_ad.py'), '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/fs/nexus-scratch/sellahy/headless-ad/src/gridworld/algorithms/headless_ad.py', '__cached__': '/fs/nexus-scratch/sellahy/headless-ad/src/gridworld/algorithms/__pycache__/headless_ad.cpython-39.pyc', 'gc': <module 'gc' (built-in)>, 'itertools': <module 'itertools' (built-in)>, 'mp': <module 'multiprocessing' from '/opt/local/stow/Python3-3.9.16/lib/python3.9/multiprocessing/__init__.py'>, 'os': <module 'os' from '/opt/local/stow/Python3-3.9.16/lib/python3.9/os.py'>, 'pickle': <module 'pickle' from '/opt/local/stow/Python3-3.9.16/lib/python3.9/pickle.py'>, 'shutil': <module 'shutil' from '/opt/local/stow/Python3-3.9.16/lib/python3.9/shutil.py'>, 'asdict': <function asdict at 0x7f02f6d025e0>, 'dataclass': <function dataclass at 0x7f02f6d023a0>, 'Dict': typing.Dict, 'List': typing.List, 'Optional': typing.Optional, 'Set': typing.Set, 'Tuple': typing.Tuple, 'gym': <module 'gymnasium' from '/fs/nexus-scratch/sellahy/had9/lib/python3.9/site-packages/gymnasium/__init__.py'>, 'np': <module 'numpy' from '/fs/nexus-scratch/sellahy/had9/lib/python3.9/site-packages/numpy/__init__.py'>, 'pyrallis': <module 'pyrallis' from '/fs/nexus-scratch/sellahy/had9/lib/python3.9/site-packages/pyrallis/__init__.py'>, 'torch': <module 'torch' from '/fs/nexus-scratch/sellahy/had9/lib/python3.9/site-packages/torch/__init__.py'>, 'Accelerator': <class 'accelerate.accelerator.Accelerator'>, 'SyncVectorEnv': <class 'gymnasium.vector.sync_vector_env.SyncVectorEnv'>, 'DataLoader': <class 'torch.utils.data.dataloader.DataLoader'>, 'tqdm': <class 'tqdm.std.tqdm'>, 'trange': <function trange at 0x7f021fc3ce50>, 'envs': <module 'envs' from '/fs/nexus-scratch/sellahy/headless-ad/envs/__init__.py'>, 'wandb': <module 'wandb' from '/fs/nexus-scratch/sellahy/had9/lib/python3.9/site-packages/wandb/__init__.py'>, 'ActionMapper': <class 'src.action_mapper.ActionMapper'>, 'SequenceDataset': <class 'src.gridworld.common.data.SequenceDataset'>, 'generate_dataset': <function generate_dataset at 0x7f01f6bba430>, 'solve_env': <function solve_env at 0x7f01f6bba280>, 'make_envs': <function make_envs at 0x7f01f6bbaca0>, 'Model': <class 'src.gridworld.common.transformer.Model'>, 'ModelConfig': <class 'src.tiny_llama.config.Config'>, 'Timeit': <class 'src.utils.misc.Timeit'>, 'set_seed': <function set_seed at 0x7f01fae4a700>, 'cosine_annealing_with_warmup': <function cosine_annealing_with_warmup at 0x7f01f1ac2c10>, 'log_in_context': <function log_in_context at 0x7f01efff3b80>, 'log_list': <function log_list at 0x7f01efff3d30>, 'log_raw_regrets': <function log_raw_regrets at 0x7f01f1ac2dc0>, 'pdb': <module 'pdb' from '/opt/local/stow/Python3-3.9.16/lib/python3.9/pdb.py'>, 'Config': <class '__main__.Config'>, 'make_env': <function make_env at 0x7f02f8b58670>, 'evaluate_in_context': <function evaluate_in_context at 0x7f01efffc1f0>, 'wandb_define_metrics': <function wandb_define_metrics at 0x7f01efffc280>, 'loss_fn': <function loss_fn at 0x7f01efffc310>, 'next_dataloader': <function next_dataloader at 0x7f01efffc3a0>, 'regrets_to_wandb': <function regrets_to_wandb at 0x7f01efffc430>, 'base_algo_scores': <function base_algo_scores at 0x7f01efffc4c0>, 'random_model_scores': <function random_model_scores at 0x7f01efffc550>, 'get_all_test_metric': <function get_all_test_metric at 0x7f01efffc5e0>, 'train': <function train at 0x7f01efffc790>}

"""
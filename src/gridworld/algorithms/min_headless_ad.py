import gc
import itertools
import multiprocessing as mp
import os
import pickle
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator
from gymnasium.vector import SyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import envs
import wandb
from src.action_mapper import ActionMapper
from src.gridworld.common.data import SequenceDataset
from src.gridworld.common.generate import generate_dataset, solve_env
from src.gridworld.common.prepare_envs import make_envs
from src.gridworld.common.transformer import Model
from src.tiny_llama.config import Config as ModelConfig
from src.utils.misc import Timeit, set_seed
from src.utils.schedule import cosine_annealing_with_warmup
from src.utils.wandb_logging import log_in_context, log_list, log_raw_regrets

if False:
    envs


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

    # Where to save model checkpoints
    save_dir : str = "model_checkpoints"

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


def make_env(env_name: str, acts: np.ndarray, goal: np.ndarray, act_seq_len: int):
    """
    This function creates an darkroom environment parameterized by the action sequences
    the length of action sequnce and a goal position.
    """

    def helper():
        return gym.make(
            env_name,
            goal_pos=goal.copy(),
            available_actions=acts.copy(),
            action_seq_len=act_seq_len,
        )

    return helper


@torch.no_grad()
def evaluate_in_context(
    config: Config,
    model: Model,
    acts: np.ndarray,
    goals: np.ndarray,
):
    model.eval()
    # Create env
    vec_env = SyncVectorEnv(
        [
            make_env(
                env_name=config.env_name,
                acts=a,
                goal=g,
                act_seq_len=config.action_seq_len,
            )
            for (a, g) in zip(acts, goals)
        ]
    )
    init_state, _ = vec_env.reset(seed=config.eval_seed)

    # reassign some variables
    num_envs = vec_env.num_envs

    # Create action embeddings
    act_mapper = ActionMapper(
        action_embed_dim=config.token_embed_dim,
        num_actions=acts.shape[1],
        device=config.device,
        sim_measure=config.sim_measure,
        rand_emb_type=config.rand_emb_type,
    )
    act_mapper.regenerate(seed=config.eval_seed)

    # Create context windows for storing the interaction histories
    states = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    states[-1, :] = torch.from_numpy(init_state).to(config.device).type(torch.long)
    actions = torch.zeros(
        (config.seq_len, num_envs, config.token_embed_dim),
        dtype=torch.float32,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    num_actions_per_env = torch.full(
        size=(num_envs,), fill_value=acts.shape[1], device=config.device
    )

    actions_list = act_mapper._get_action_map_as_context(
        num_actions_per_env=num_actions_per_env
    )

    tried_action_inds = []
    # create a list for accumulation of regrets
    all_returns = [[] for _ in range(num_envs)]
    all_lengths = [[] for _ in range(num_envs)]
    current_lengths = np.zeros(num_envs)
    current_returns = np.zeros(num_envs)
    entropies = []

    num_dones = np.zeros(num_envs, dtype=np.int32)
    tried_action_sets = [list() for _ in range(num_envs)]
    interm_tried_action_sets = [set() for _ in range(num_envs)]
    for istep in tqdm(itertools.count(start=1), desc="Eval ..."):
        inp = (
            states.T[:, -istep:],
            actions.transpose(0, 1)[:, -istep:],
            rewards.T[:, -istep:],
        )
        # check for validity
        assert (istep < config.seq_len and inp[0].shape[1] == istep) or (
            istep >= config.seq_len and inp[0].shape[1] == config.seq_len
        ), (
            inp[0].shape[1],
            istep,
        )
        # make prediction
        pred = model(*inp, actions_list=actions_list)
        pred = pred[:, -1]
        print(f"pred is {pred}")
        sys.exit()


def wandb_define_metrics() -> None:
    wandb.define_metric("data_gen/step")
    wandb.define_metric("data_gen/*", step_metric="data_gen/step")

    wandb.define_metric("final/step")
    wandb.define_metric("final/*", step_metric="final/step")

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("times/step")
    wandb.define_metric("times/*", step_metric="times/step")

    wandb.define_metric("num_uniques/step")
    wandb.define_metric("num_uniques/*", step_metric="num_uniques/step")
    wandb.define_metric("returns/step")
    wandb.define_metric("returns/*", step_metric="returns/step")

    wandb.define_metric("tried/step")
    wandb.define_metric("tried/*", step_metric="tried/step")


def next_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def base_algo_scores(config: Config, envs: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    base_algo_final_returns = {}
    with mp.Pool(processes=os.cpu_count()) as pool:
        for key, (acts, goals) in envs.items():
            returns, _ = solve_env(
                config=config, pool=pool, goals=goals, actions=acts, savedir=None
            )
            final_return = returns.mean(0)[-1]

            base_algo_final_returns[key] = final_return

    return base_algo_final_returns


def random_model_scores(
    config: Config,
    model_config: ModelConfig,
    accelerator: Accelerator,
    envs: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    real_num_in_context_episodes = config.num_in_context_episodes
    config.num_in_context_episodes = 10
    random_model = Model(
        config=model_config,
        n_token=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
        action_seq_len=config.action_seq_len,
        num_states=config.grid_size**2,
    ).to(config.device)

    random_model = accelerator.prepare(random_model)

    returns = {}

    for key, (acts, goals) in tqdm(envs.items()):
        (returns[key], _, _, _, _) = evaluate_in_context(
            config,
            model=random_model,
            acts=acts,
            goals=goals,
        )

        returns[key] = returns[key].mean(0)[-1]

    config.num_in_context_episodes = real_num_in_context_episodes
    return returns


@pyrallis.wrap()
def train(config: Config):
    # Clean up and then create directory
    if os.path.exists(config.logs_dir):
        shutil.rmtree(config.logs_dir)
    os.makedirs(config.logs_dir)

    # config.device = "cuda" if torch.cuda.is_available() else "cpu"

    config.autocast_dtype = (
        "bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )
    accelerator = Accelerator(mixed_precision=config.autocast_dtype)
    config.device = accelerator.device

    wandb.init(
        project=config.project,
        group=config.group,
        job_type=config.job_type,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    wandb_define_metrics()

    (train_acts, train_goals), eval_envs = make_envs(config=config)
    generate_dataset(config=config, actions=train_acts, goals=train_goals)
    q_learning_scores = base_algo_scores(config=config, envs=eval_envs)
    for key, value in q_learning_scores.items():
        wandb.log({f"q_learning/{key}": value})

    set_seed(seed=config.train_seed)

    dataset = SequenceDataset(
        runs_path=config.learning_histories_path, seq_len=config.seq_len
    )
    shape0s = np.array(
        [len(dataset._states), len(dataset._actions), len(dataset._rewards)]
    )
    assert np.all(shape0s == config.num_train_envs), (
        shape0s,
        config.num_train_envs,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        persistent_workers=True,
        drop_last=True,
    )

    # model & optimizer & scheduler setup
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

    random_scores = random_model_scores(
        config=config,
        model_config=model_config,
        accelerator=accelerator,
        envs=eval_envs,
    )


if __name__ == "__main__":
    train()

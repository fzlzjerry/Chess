import chess
import chess.engine
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import random
from typing import Optional, Tuple, Dict, Any
import os

# 设置可见的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 13), dtype=np.float32)
        self.engine_path = "/usr/bin/stockfish"  # 更新路径

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.board.reset()
        self.last_opponent_move = None
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        move = self._get_move_from_action(action)
        if move is None or move not in self.board.legal_moves:
            return self._get_obs(), -1, True, False, {}

        self.board.push(move)
        reward = self._get_reward(move)
        done = self.board.is_game_over()

        if not done:
            opponent_move = self.stockfish_move()
            if opponent_move:
                self.board.push(opponent_move)
                self.last_opponent_move = opponent_move

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        return get_obs(self.board, self.last_opponent_move)

    def _get_move_from_action(self, action: int) -> Optional[chess.Move]:
        return get_move_from_action(self.board, action)

    def _get_reward(self, move: chess.Move) -> float:
        reward = 0

        # 基本原则应用
        if self.board.is_checkmate():
            reward = 1000
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            reward = -10
        elif self.board.is_capture(move):
            piece = self.board.piece_at(move.to_square)
            reward = 0.1
            if piece:
                piece_values = {
                    chess.QUEEN: 9.0,
                    chess.ROOK: 5.0,
                    chess.BISHOP: 3.5,  # 主教略高于骑士
                    chess.KNIGHT: 3.0,
                    chess.PAWN: 1.0
                }
                reward += piece_values.get(piece.piece_type, 0)

        # 强调策略
        # 避免用一个小子换取三个兵
        if piece and piece.piece_type in [chess.BISHOP, chess.KNIGHT] and piece_values[piece.piece_type] > 3 * \
                piece_values[chess.PAWN]:
            reward -= 5

        # 避免用两个小子换取一个车和一个兵
        if piece and piece.piece_type in [chess.ROOK] and reward < piece_values[chess.BISHOP] + piece_values[
            chess.KNIGHT] - piece_values[chess.PAWN]:
            reward -= 5

        # 占据中心位置的奖励
        if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
            reward += 0.5

        reward -= 0.01 * self.board.fullmove_number
        return reward

    def stockfish_move(self) -> Optional[chess.Move]:
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            result = engine.play(self.board, chess.engine.Limit(time=0.1))
            return result.move


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 1024):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample_input = th.zeros(8, n_input_channels, observation_space.shape[0],
                                    observation_space.shape[1])  # 使用批次大小为8
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


def get_obs(board: chess.Board, last_opponent_move: Optional[chess.Move]) -> np.ndarray:
    board_planes = np.zeros((8, 8, 13), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            plane_index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            board_planes[i // 8, i % 8, plane_index] = 1
    if last_opponent_move:
        from_square = last_opponent_move.from_square
        to_square = last_opponent_move.to_square
        board_planes[from_square // 8, from_square % 8, 12] = -1
        board_planes[to_square // 8, to_square % 8, 12] = 1
    return board_planes


def get_move_from_action(board: chess.Board, action: int) -> Optional[chess.Move]:
    legal_moves = list(board.legal_moves)
    return legal_moves[action] if action < len(legal_moves) else None


def random_move(board: chess.Board) -> Optional[chess.Move]:
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None


def stockfish_move(board: chess.Board) -> Optional[chess.Move]:
    with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as engine:  # 确保路径正确
        result = engine.play(board, chess.engine.Limit(time=0.1))
        return result.move


def evaluate_model(model: PPO, episodes: int = 100) -> None:
    wins, draws, losses = 0, 0, 0

    for _ in range(episodes):
        board = chess.Board()
        done = False
        last_opponent_move = None
        while not done:
            if board.turn == chess.WHITE:
                obs = get_obs(board, last_opponent_move)[None]
                obs = th.tensor(obs).float().to('cuda:0')  # 使用第一个GPU
                action, _ = model.predict(obs, deterministic=True)
                move = get_move_from_action(board, action[0])
            else:
                move = stockfish_move(board)

            if move is None or move not in board.legal_moves:
                break

            board.push(move)
            done = board.is_game_over()

            if board.turn == chess.BLACK:
                last_opponent_move = move

        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    print(f"Results after {episodes} episodes:")
    print(f"Wins: {wins}")
    print(f"Draws: {draws}")
    print(f"Losses: {losses}")


class CustomStopTrainingOnNoModelImprovement(StopTrainingOnNoModelImprovement):
    def __init__(self, max_no_improvement_evals: int = 10, min_evals: int = 0, verbose: int = 0):
        super().__init__(max_no_improvement_evals, min_evals, verbose)
        self.num_no_improvement_evals = 0

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.num_no_improvement_evals > 0:
            print(
                f"No improvement for {self.num_no_improvement_evals} evaluations. Patience left: {self.max_no_improvement_evals - self.num_no_improvement_evals}")
        return result

def init_process(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def run(rank, size):
    init_process(rank, size)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=1024),
    )

    # 创建和包装环境
    def make_env():
        def _init():
            env = ChessEnv()
            env = Monitor(env)
            return env

        return _init

    num_cpu = 4  # 使用四个CPU
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # 初始化模型
    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0001, n_steps=4096,
                ent_coef=0.01, device=f'cuda:{rank}')

    # 定义早停回调
    stop_callback = CustomStopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
    eval_env = ChessEnv()
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, eval_freq=20000,
                                 best_model_save_path='./logs/', verbose=1)

    model.learn(total_timesteps=10000000, callback=eval_callback)

    # 保存模型
    model.save("chess_model")

    # 评估模型
    evaluate_model(model, episodes=100)


if __name__ == '__main__':
    size = 4  # Number of GPUs
    mp.spawn(run, args=(size,), nprocs=size, join=True)

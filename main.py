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
import random
from typing import Optional, Tuple, Dict, Any

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 13), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
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
        if self.board.is_checkmate():
            reward = 1000
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            reward = -10
        elif self.board.is_capture(move):
            piece = self.board.piece_at(move.to_square)
            reward = 0.1
            if piece:
                if piece.piece_type == chess.QUEEN:
                    reward += 9.0
                elif piece.piece_type == chess.ROOK:
                    reward += 5.0
                elif piece.piece_type in {chess.BISHOP, chess.KNIGHT}:
                    reward += 3.0
                elif piece.piece_type == chess.PAWN:
                    reward += 1.0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        if move.to_square in center_squares:
            reward += 0.5
        reward -= 0.01 * self.board.fullmove_number
        return reward

    def stockfish_move(self) -> Optional[chess.Move]:
        with chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish") as engine:
            result = engine.play(self.board, chess.engine.Limit(time=0.1))
            return result.move

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample_input = th.zeros(1, n_input_channels, observation_space.shape[0], observation_space.shape[1])
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))

def get_obs(board: chess.Board, last_opponent_move: Optional[chess.Move]) -> np.ndarray:
    board_planes = np.zeros((8, 8, 13), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            plane_index = {
                chess.PAWN: 0,
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK: 3,
                chess.QUEEN: 4,
                chess.KING: 5
            }[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
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

def evaluate_model(model: PPO, episodes: int = 100) -> None:
    wins, draws, losses = 0, 0, 0

    for _ in range(episodes):
        board = chess.Board()
        done = False
        last_opponent_move = None
        while not done:
            if board.turn == chess.WHITE:
                obs = get_obs(board, last_opponent_move)[None]
                obs = th.tensor(obs).float()
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

def stockfish_move(board: chess.Board) -> Optional[chess.Move]:
    with chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish") as engine:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        return result.move

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

# 创建和包装环境
env = ChessEnv()
env = Monitor(env)  # 使用 Monitor 包装环境
obs, _ = env.reset()
print(obs.shape)  # 应输出 (8, 8, 13)

model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0001, n_steps=2048, ent_coef=0.01)

# 定义早停回调
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, verbose=1)
eval_env = ChessEnv()
eval_env = Monitor(eval_env)  # 使用 Monitor 包装评估环境
eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path='./logs/', verbose=1)

model.learn(total_timesteps=2000000, callback=eval_callback)  # 增加训练步数

# 保存模型
model.save("chess_model")

# 评估模型
evaluate_model(model, episodes=100)

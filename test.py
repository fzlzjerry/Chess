import chess
import chess.engine
import torch as th
import numpy as np
from stable_baselines3 import PPO
from typing import Optional, Tuple

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

def stockfish_move(board: chess.Board) -> Optional[chess.Move]:
    try:
        with chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish") as engine:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            return result.move
    except Exception as e:
        print(f"Error during Stockfish move: {e}")
        return None

def evaluate_model(model: PPO, episodes: int = 100) -> None:
    wins, draws, losses = 0, 0, 0

    for episode in range(episodes):
        board = chess.Board()
        done = False
        last_opponent_move = None
        print(f"Starting episode {episode + 1}")
        while not done:
            if board.turn == chess.WHITE:
                obs = get_obs(board, last_opponent_move)[None]
                obs = th.tensor(obs).float()
                action, _ = model.predict(obs, deterministic=True)
                move = get_move_from_action(board, action[0])
                print(f"White move: {move}")
            else:
                move = stockfish_move(board)
                print(f"Stockfish move: {move}")

            if move is None or move not in board.legal_moves:
                print("Invalid move encountered.")
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

if __name__ == "__main__":
    # 加载模型
    model = PPO.load("logs/best_model.zip")

    # 评估模型
    evaluate_model(model, episodes=10)

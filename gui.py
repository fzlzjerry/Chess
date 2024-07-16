import tkinter as tk
import chess
import chess.svg
import torch as th
import numpy as np
from stable_baselines3 import PPO
from typing import Optional, Tuple
from PIL import Image, ImageTk
import io
import cairosvg

# Load the trained model
model = PPO.load("chess_model")


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
    with chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish") as engine:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        return result.move


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.board = chess.Board()
        self.last_opponent_move = None
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=480, height=480)
        self.canvas.pack()

        self.update_board()

        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x, y = event.x // 60, event.y // 60
        square = chess.square(x, 7 - y)

        if self.board.turn == chess.WHITE:
            move = None
            for legal_move in self.board.legal_moves:
                if legal_move.from_square == square:
                    move = legal_move
                    break

            if move and move in self.board.legal_moves:
                self.board.push(move)
                self.last_opponent_move = move
                self.update_board()

                # Model's turn
                obs = get_obs(self.board, self.last_opponent_move)[None]
                obs = th.tensor(obs).float()
                action, _ = model.predict(obs, deterministic=True)
                model_move = get_move_from_action(self.board, action[0])

                if model_move and model_move in self.board.legal_moves:
                    self.board.push(model_move)
                    self.update_board()

    def update_board(self):
        svg_data = chess.svg.board(self.board).encode("utf-8")
        png_data = cairosvg.svg2png(bytestring=svg_data)
        img = Image.open(io.BytesIO(png_data))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()

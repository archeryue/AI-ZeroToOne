# Go (围棋) Web Game

Play Go against AI opponents on 9x9, 13x13, or 19x19 boards.

## Quick Start

**Backend** (port 8001):
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Frontend** (port 3001):
```bash
cd frontend
npm install
npm run dev -- --port 3001
```

Open http://localhost:3001 to play.

## Game Modes

| AI | Strategy | Strength |
|---|---|---|
| **Random** | Picks random legal moves | Baseline |
| **Greedy** | Captures + territory + center control heuristic | Beginner |
| **MCTS** | Monte Carlo Tree Search with random rollouts (100-2000 sims) | Intermediate |
| **Human** | Two-player mode | — |

## Features

- Board sizes: 9x9, 13x13, 19x19
- Play as Black or White
- Real-time play via WebSocket
- Move history with coordinate notation
- Capture counting
- Pass, undo, resign
- Tromp-Taylor area scoring with territory visualization
- Last move indicator
- Hover preview for legal moves

## Architecture

Same pattern as the ChineseChess game:

- **Backend**: FastAPI + WebSocket (Python)
- **Frontend**: Next.js 16 + React 19 + TypeScript + Tailwind CSS
- **Engine**: Pure Python Go rules (board, capture, ko, scoring)
- **AI**: Pluggable AI interface — ready for AlphaZero integration later

## Future: AlphaZero Integration

After training our AlphaZero model (see `../AlphaZero/PLAN.md`), we can deploy it as a new AI opponent by adding an `alphazero_ai.py` that loads the trained model and uses MCTS with the neural network for move selection.

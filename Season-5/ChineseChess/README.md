# ChineseChess (Xiangqi)

A full-stack Chinese Chess web application with multiple AI opponents, built with Next.js and FastAPI.

## Project Structure

```
ChineseChess/
├── frontend/              # Next.js + React + TypeScript
│   └── src/
│       ├── app/           # Game page
│       ├── components/    # Board, Piece, GameStatus, GameControls, MoveHistory
│       ├── hooks/         # WebSocket hook
│       └── lib/           # API client, types, constants
└── backend/               # FastAPI + Python
    ├── api/               # REST routes, WebSocket handler, game manager
    ├── ai/                # AI engines (Random, Greedy, Minimax, NNUE)
    ├── engine/            # Game logic (board, pieces, rules, move)
    └── tests/
```

## Quick Start

### 1. Build the C++ Engine (required for NNUE AI)

```bash
cd Season-5/ChessRL/engine_c

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

pip install setuptools pybind11
python setup.py build_ext --inplace
```

### 2. Start the Backend

```bash
cd Season-5/ChineseChess/backend

# Use the same venv, or install dependencies
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.

### 3. Start the Frontend

```bash
cd Season-5/ChineseChess/frontend

npm install
npm run dev
```

Frontend runs at `http://localhost:3000`. Open it in your browser to play.

## Game Modes

| AI Type  | Description                                      |
| -------- | ------------------------------------------------ |
| Random   | Picks a random legal move                        |
| Greedy   | Material-based evaluation, beginner level        |
| Minimax  | Alpha-beta search, depth 1-4, intermediate level |
| NNUE     | Neural network eval + C++ search, strongest      |
| Human    | Two-player mode, no AI                           |

You can choose to play as Red (first move) or Black, and adjust AI search depth (1-4).

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS
- **Backend**: FastAPI, uvicorn, WebSocket
- **AI Engine**: C++ with pybind11 (NNUE v2 from ChessRL)

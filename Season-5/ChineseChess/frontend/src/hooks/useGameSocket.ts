"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE } from "@/lib/constants";
import { GameState, WSMessage } from "@/lib/types";

export function useGameSocket(gameId: string | null) {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!gameId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      const msg: WSMessage = JSON.parse(event.data);
      if (msg.type === "game_state") {
        setGameState(msg.data as GameState);
      } else if (msg.type === "error") {
        setError((msg.data as { message: string }).message);
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [gameId]);

  const sendMove = useCallback(
    (fromRow: number, fromCol: number, toRow: number, toCol: number) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "move",
            data: {
              from_row: fromRow,
              from_col: fromCol,
              to_row: toRow,
              to_col: toCol,
            },
          })
        );
      }
    },
    []
  );

  const sendUndo = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "undo" }));
    }
  }, []);

  const sendResign = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "resign" }));
    }
  }, []);

  return { gameState, connected, error, sendMove, sendUndo, sendResign };
}

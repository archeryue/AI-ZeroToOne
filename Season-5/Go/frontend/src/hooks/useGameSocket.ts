"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { WS_BASE } from "@/lib/constants";
import type { GameState, WSMessage } from "@/lib/types";

export function useGameSocket(gameId: string | null) {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);
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

      switch (msg.type) {
        case "game_state":
          setGameState(msg.data as GameState);
          setAiThinking(false);
          break;
        case "ai_thinking":
          setAiThinking(true);
          break;
        case "ai_move":
          setAiThinking(false);
          break;
        case "error":
          setError(msg.message || "Unknown error");
          setAiThinking(false);
          break;
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      setConnected(false);
      setError("WebSocket connection failed");
    };

    return () => {
      ws.close();
    };
  }, [gameId]);

  const sendMove = useCallback((row: number, col: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "move", row, col }));
    }
  }, []);

  const sendPass = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "pass" }));
    }
  }, []);

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

  return {
    gameState,
    setGameState,
    connected,
    aiThinking,
    error,
    sendMove,
    sendPass,
    sendUndo,
    sendResign,
  };
}

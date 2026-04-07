"use client";

import type { MoveRecord } from "@/lib/types";
import { useRef, useEffect } from "react";

interface MoveHistoryProps {
  moves: MoveRecord[];
  boardSize: number;
}

function formatMove(m: MoveRecord, boardSize: number): string {
  if (m.position === "pass") return "Pass";
  const [r, c] = m.position.split(",").map(Number);
  const colLabel = String.fromCharCode(65 + (c >= 8 ? c + 1 : c));
  const rowLabel = boardSize - r;
  return `${colLabel}${rowLabel}${m.captured > 0 ? ` (x${m.captured})` : ""}`;
}

export default function MoveHistory({ moves, boardSize }: MoveHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [moves.length]);

  if (moves.length === 0) {
    return <div className="text-sm text-gray-400 italic">No moves yet</div>;
  }

  // Pair moves (black, white) per row
  const pairs: [MoveRecord, MoveRecord | null][] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push([moves[i], moves[i + 1] || null]);
  }

  return (
    <div ref={scrollRef} className="max-h-60 overflow-y-auto text-sm">
      <table className="w-full">
        <thead>
          <tr className="text-gray-400 text-xs">
            <th className="w-8 text-left">#</th>
            <th className="text-left">Black</th>
            <th className="text-left">White</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map(([black, white], i) => (
            <tr key={i} className="hover:bg-gray-50">
              <td className="text-gray-400">{i + 1}</td>
              <td className="font-mono">{formatMove(black, boardSize)}</td>
              <td className="font-mono">{white ? formatMove(white, boardSize) : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

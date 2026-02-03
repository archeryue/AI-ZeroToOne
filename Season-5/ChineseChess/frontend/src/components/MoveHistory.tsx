"use client";

import { MoveRecord } from "@/lib/types";
import { PIECE_NAMES_RED, PIECE_NAMES_BLACK } from "@/lib/constants";

interface MoveHistoryProps {
  moves: MoveRecord[];
  board: number[][];
}

function formatMove(move: MoveRecord, index: number): string {
  const isRed = index % 2 === 0;
  const moveNum = Math.floor(index / 2) + 1;
  const prefix = isRed ? `${moveNum}. ` : "";
  return `${prefix}${move.ucci}`;
}

export default function MoveHistory({ moves }: MoveHistoryProps) {
  if (moves.length === 0) {
    return (
      <div className="text-stone-500 text-sm italic">No moves yet</div>
    );
  }

  // Group moves into pairs (red, black)
  const pairs: [MoveRecord, MoveRecord | null][] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push([moves[i], moves[i + 1] || null]);
  }

  return (
    <div className="max-h-64 overflow-y-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-stone-400 border-b border-stone-700">
            <th className="w-8 text-left py-1">#</th>
            <th className="text-left py-1">Red</th>
            <th className="text-left py-1">Black</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map(([red, black], i) => (
            <tr key={i} className="border-b border-stone-800">
              <td className="text-stone-500 py-1">{i + 1}</td>
              <td className="text-red-400 py-1 font-mono">{red.ucci}</td>
              <td className="text-blue-400 py-1 font-mono">
                {black?.ucci || ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

import { PuzzleData } from '../types'

type Props = {
  puzzles: PuzzleData[]
  onSelect: (puzzle: PuzzleData) => void
}

export function PuzzleList({ puzzles, onSelect }: Props) {
  return (
    <div className="puzzle-list">
      <p>Select a puzzle:</p>
      {puzzles.map((p, i) => (
        <button key={i} onClick={() => onSelect(p)}>
          {p.name}
        </button>
      ))}
    </div>
  )
}


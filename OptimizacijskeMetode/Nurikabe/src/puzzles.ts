import puzzle5x5 from '../data/nurikabe5x5.json'
import puzzle10x10 from '../data/nurikabe10x10.json'
import puzzle10x10_2 from '../data/nurikabe10x10-2.json'
import puzzle10x10_3 from '../data/nurikabe10x10-3.json'
import puzzle10x10_4 from '../data/nurikabe10x10-4.json'
import puzzle10x10_5 from '../data/nurikabe10x10-5.json'
import { PuzzleData, GameState, CellState } from './types'

export const puzzleData: PuzzleData[] = [
  puzzle5x5,
  puzzle10x10,
  puzzle10x10_2,
  puzzle10x10_3,
  puzzle10x10_4,
  puzzle10x10_5
]

export function createGameState(data: PuzzleData): GameState {
  return {
    name: data.name,
    grid: data.puzzle.map(row =>
      row.map(value => ({
        value,
        state: 'gray' as CellState
      }))
    )
  }
}


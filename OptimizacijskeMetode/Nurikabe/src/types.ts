export type CellState = 'gray' | 'white' | 'black'

export type Cell = {
  value: number      // 0 = no number, >0 = island size hint
  state: CellState   // current display state
}

export type GameState = {
  name: string
  grid: Cell[][]
}

export type PuzzleData = {
  name: string
  puzzle: number[][]
}


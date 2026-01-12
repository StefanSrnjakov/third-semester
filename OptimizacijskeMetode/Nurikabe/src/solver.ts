import { GameState, CellState } from './types'
import {
  markNumberedCellsWhite,
  fillMandatoryBlack,
  fillMandatoryWhite,
  fillUnreachable,
  preventTwoByTwo,
  expandIsolatedWhite,
  expandIsolatedBlack,
  hasNo2x2BlackSquare,
  areBlackCellsConnected,
  hasOneNumberPerIsland,
  isIslandSizeCorrect,
  hasNoGrayCells
} from './utils'

function cloneGame(game: GameState): GameState {
  return {
    ...game,
    grid: game.grid.map(row => row.map(cell => ({ ...cell })))
  }
}

function gamesEqual(a: GameState, b: GameState): boolean {
  for (let r = 0; r < a.grid.length; r++) {
    for (let c = 0; c < a.grid[0].length; c++) {
      if (a.grid[r][c].state !== b.grid[r][c].state) return false
    }
  }
  return true
}

function countGrayCells(game: GameState): number {
  let count = 0
  for (const row of game.grid) {
    for (const cell of row) {
      if (cell.state === 'gray') count++
    }
  }
  return count
}

/**
 * Check that white islands don't have multiple numbered cells
 * and don't exceed their expected size.
 */
function areWhiteIslandsValid(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const island: { row: number; col: number }[] = []
        const stack = [{ row: r, col: c }]
        
        while (stack.length > 0) {
          const pos = stack.pop()!
          if (visited[pos.row][pos.col]) continue
          if (grid[pos.row][pos.col].state !== 'white') continue
          
          visited[pos.row][pos.col] = true
          island.push(pos)
          
          if (pos.row > 0) stack.push({ row: pos.row - 1, col: pos.col })
          if (pos.row < rows - 1) stack.push({ row: pos.row + 1, col: pos.col })
          if (pos.col > 0) stack.push({ row: pos.row, col: pos.col - 1 })
          if (pos.col < cols - 1) stack.push({ row: pos.row, col: pos.col + 1 })
        }
        
        const numberedCells = island.filter(p => grid[p.row][p.col].value > 0)
        if (numberedCells.length > 1) return false
        
        if (numberedCells.length === 1) {
          const expectedSize = grid[numberedCells[0].row][numberedCells[0].col].value
          if (island.length > expectedSize) return false
        }
      }
    }
  }
  
  return true
}

/**
 * Count the number of separate black regions in the grid.
 */
function countBlackRegions(game: GameState): number {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  let regionCount = 0
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'black' && !visited[r][c]) {
        regionCount++
        const stack = [{ row: r, col: c }]
        while (stack.length > 0) {
          const pos = stack.pop()!
          if (visited[pos.row][pos.col]) continue
          if (grid[pos.row][pos.col].state !== 'black') continue
          
          visited[pos.row][pos.col] = true
          
          if (pos.row > 0) stack.push({ row: pos.row - 1, col: pos.col })
          if (pos.row < rows - 1) stack.push({ row: pos.row + 1, col: pos.col })
          if (pos.col > 0) stack.push({ row: pos.row, col: pos.col - 1 })
          if (pos.col < cols - 1) stack.push({ row: pos.row, col: pos.col + 1 })
        }
      }
    }
  }
  
  return regionCount
}

/**
 * Check that all black regions can potentially connect via gray cells.
 * Each isolated black region must have at least one gray neighbor.
 */
function canBlackRegionsConnect(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'black' && !visited[r][c]) {
        let hasGrayNeighbor = false
        const stack = [{ row: r, col: c }]
        
        while (stack.length > 0) {
          const pos = stack.pop()!
          if (visited[pos.row][pos.col]) continue
          if (grid[pos.row][pos.col].state !== 'black') continue
          
          visited[pos.row][pos.col] = true
          
          const neighbors = [
            { row: pos.row - 1, col: pos.col },
            { row: pos.row + 1, col: pos.col },
            { row: pos.row, col: pos.col - 1 },
            { row: pos.row, col: pos.col + 1 }
          ]
          for (const n of neighbors) {
            if (n.row >= 0 && n.row < rows && n.col >= 0 && n.col < cols) {
              if (grid[n.row][n.col].state === 'gray') hasGrayNeighbor = true
              if (grid[n.row][n.col].state === 'black' && !visited[n.row][n.col]) {
                stack.push(n)
              }
            }
          }
        }
        
        if (!hasGrayNeighbor) return false
      }
    }
  }
  
  return true
}

/**
 * Check for contradictions before puzzle is complete:
 * - 2x2 black squares
 * - Islands too big or with multiple numbers
 * - Isolated black regions that can't connect
 */
function isPartiallyValid(game: GameState): boolean {
  if (!hasNo2x2BlackSquare(game)) return false
  if (!areWhiteIslandsValid(game)) return false
  
  const blackRegionCount = countBlackRegions(game)
  if (blackRegionCount > 1 && !canBlackRegionsConnect(game)) return false
  
  return true
}

/**
 * PROPAGATION: Apply all constraint rules until no more changes.
 * Returns null if a contradiction is found.
 */
function propagate(game: GameState): GameState | null {
  let current = cloneGame(game)
  let changed = true
  
  while (changed) {
    if (!isPartiallyValid(current)) return null
    
    const before = cloneGame(current)
    
    current = markNumberedCellsWhite(current)
    current = fillMandatoryBlack(current)
    current = fillMandatoryWhite(current)
    current = preventTwoByTwo(current)
    current = expandIsolatedWhite(current)
    current = expandIsolatedBlack(current)
    current = fillUnreachable(current)
    
    changed = !gamesEqual(before, current)
  }
  
  if (!isPartiallyValid(current)) return null
  return current
}

function isSolved(game: GameState): boolean {
  return (
    hasNoGrayCells(game) &&
    hasNo2x2BlackSquare(game) &&
    areBlackCellsConnected(game) &&
    hasOneNumberPerIsland(game) &&
    isIslandSizeCorrect(game)
  )
}

/**
 * HEURISTIC: Pick the best cell to branch on.
 * Prefers cells near islands and in constrained positions.
 */
function findBranchCell(game: GameState): { row: number; col: number } | null {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  
  let bestCell: { row: number; col: number } | null = null
  let bestScore = -1
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state !== 'gray') continue
      
      let score = 0
      const neighbors = [
        { row: r - 1, col: c },
        { row: r + 1, col: c },
        { row: r, col: c - 1 },
        { row: r, col: c + 1 }
      ]
      
      let whiteNeighbors = 0
      let blackNeighbors = 0
      let grayNeighbors = 0
      
      for (const n of neighbors) {
        if (n.row >= 0 && n.row < rows && n.col >= 0 && n.col < cols) {
          if (grid[n.row][n.col].state === 'white') whiteNeighbors++
          else if (grid[n.row][n.col].state === 'black') blackNeighbors++
          else grayNeighbors++
        }
      }
      
      score += whiteNeighbors * 10
      score += blackNeighbors * 5
      score += (4 - grayNeighbors) * 3
      if (r === 0 || r === rows - 1) score += 2
      if (c === 0 || c === cols - 1) score += 2
      
      if (score > bestScore) {
        bestScore = score
        bestCell = { row: r, col: c }
      }
    }
  }
  
  return bestCell
}

/**
 * DFS WITH BACKTRACKING:
 * 1. Propagate all constraints
 * 2. If stuck, pick a cell and try black/white
 * 3. If contradiction, backtrack and try the other option
 */
function dfs(game: GameState): GameState | null {
  const propagated = propagate(game)
  if (propagated === null) return null
  if (isSolved(propagated)) return propagated
  if (countGrayCells(propagated) === 0) return null
  
  const branchCell = findBranchCell(propagated)
  if (!branchCell) return null
  
  for (const tryState of ['black', 'white'] as CellState[]) {
    if (propagated.grid[branchCell.row][branchCell.col].value > 0 && tryState === 'black') {
      continue
    }
    
    const attempt = cloneGame(propagated)
    attempt.grid[branchCell.row][branchCell.col].state = tryState
    
    const result = dfs(attempt)
    if (result !== null) return result
  }
  
  return null
}

export function solveNurikabe(game: GameState): GameState {
  const result = dfs(game)
  if (result) return result
  
  console.warn('No solution found!')
  return game
}

type StepCallback = (state: GameState, info: { depth: number; action: string }) => void

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function propagateAnimated(
  game: GameState,
  depth: number,
  onStep: StepCallback,
  delay: number,
  signal?: AbortSignal
): Promise<GameState | null> {
  let current = cloneGame(game)
  let changed = true
  let iteration = 0
  
  while (changed) {
    if (signal?.aborted) return null
    
    if (!isPartiallyValid(current)) {
      onStep(current, { depth, action: '❌ Contradiction found!' })
      await sleep(delay)
      return null
    }
    
    const before = cloneGame(current)
    
    current = markNumberedCellsWhite(current)
    current = fillMandatoryBlack(current)
    current = fillMandatoryWhite(current)
    current = preventTwoByTwo(current)
    current = expandIsolatedWhite(current)
    current = expandIsolatedBlack(current)
    current = fillUnreachable(current)
    
    changed = !gamesEqual(before, current)
    
    if (changed) {
      iteration++
      onStep(current, { depth, action: `Propagating... (pass ${iteration})` })
      await sleep(delay)
    }
  }
  
  if (!isPartiallyValid(current)) return null
  return current
}

async function dfsAnimated(
  game: GameState,
  depth: number,
  onStep: StepCallback,
  delay: number,
  signal?: AbortSignal
): Promise<GameState | null> {
  if (signal?.aborted) return null
  
  onStep(game, { depth, action: `Depth ${depth}: Starting propagation` })
  await sleep(delay)
  
  const propagated = await propagateAnimated(game, depth, onStep, delay, signal)
  if (propagated === null) return null
  
  if (isSolved(propagated)) {
    onStep(propagated, { depth, action: '✅ Solution found!' })
    return propagated
  }
  
  if (countGrayCells(propagated) === 0) return null
  
  const branchCell = findBranchCell(propagated)
  if (!branchCell) return null
  
  for (const tryState of ['black', 'white'] as CellState[]) {
    if (signal?.aborted) return null
    
    if (propagated.grid[branchCell.row][branchCell.col].value > 0 && tryState === 'black') {
      continue
    }
    
    const attempt = cloneGame(propagated)
    attempt.grid[branchCell.row][branchCell.col].state = tryState
    
    onStep(attempt, { 
      depth, 
      action: `Depth ${depth}: Trying cell (${branchCell.row},${branchCell.col}) = ${tryState}` 
    })
    await sleep(delay * 2)
    
    const result = await dfsAnimated(attempt, depth + 1, onStep, delay, signal)
    
    if (result !== null) return result
    
    onStep(propagated, { depth, action: `Depth ${depth}: Backtracking...` })
    await sleep(delay)
  }
  
  return null
}

/**
 * ANIMATED SOLVER: Same algorithm but yields intermediate states for visualization.
 * Uses async/await with delays to show the solving process step by step.
 */
export async function solveNurikabeAnimated(
  game: GameState,
  onStep: StepCallback,
  delay: number = 100,
  signal?: AbortSignal
): Promise<GameState | null> {
  return dfsAnimated(game, 0, onStep, delay, signal)
}

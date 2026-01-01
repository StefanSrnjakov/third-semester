import { GameState, Cell, CellState } from './types'

/**
 * NURIKABE RULES:
 * 1. No 2x2 black squares allowed
 * 2. All black cells must be connected (one sea)
 * 3. Each island has exactly as many white cells as its number
 * 4. Each island contains exactly one numbered cell
 * 5. Islands cannot touch orthogonally
 */

type Position = { row: number; col: number }

function getNeighbors(row: number, col: number, rows: number, cols: number): Position[] {
  const neighbors: Position[] = []
  if (row > 0) neighbors.push({ row: row - 1, col })
  if (row < rows - 1) neighbors.push({ row: row + 1, col })
  if (col > 0) neighbors.push({ row, col: col - 1 })
  if (col < cols - 1) neighbors.push({ row, col: col + 1 })
  return neighbors
}

function floodFill(
  grid: Cell[][],
  startRow: number,
  startCol: number,
  targetState: 'black' | 'white',
  visited: boolean[][]
): Position[] {
  const rows = grid.length
  const cols = grid[0].length
  const region: Position[] = []
  const stack: Position[] = [{ row: startRow, col: startCol }]

  while (stack.length > 0) {
    const { row, col } = stack.pop()!
    if (visited[row][col]) continue
    if (grid[row][col].state !== targetState) continue

    visited[row][col] = true
    region.push({ row, col })

    for (const neighbor of getNeighbors(row, col, rows, cols)) {
      if (!visited[neighbor.row][neighbor.col]) {
        stack.push(neighbor)
      }
    }
  }

  return region
}

export function hasNo2x2BlackSquare(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      if (
        grid[r][c].state === 'black' &&
        grid[r + 1][c].state === 'black' &&
        grid[r][c + 1].state === 'black' &&
        grid[r + 1][c + 1].state === 'black'
      ) {
        return false
      }
    }
  }
  return true
}

export function areBlackCellsConnected(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0

  let firstBlack: Position | null = null
  let totalBlackCount = 0

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'black') {
        totalBlackCount++
        if (!firstBlack) {
          firstBlack = { row: r, col: c }
        }
      }
    }
  }

  if (totalBlackCount === 0) return true

  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  const region = floodFill(grid, firstBlack!.row, firstBlack!.col, 'black', visited)

  return region.length === totalBlackCount
}

export function isIslandSizeCorrect(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const island = floodFill(grid, r, c, 'white', visited)
        const numberedCells = island.filter(pos => grid[pos.row][pos.col].value > 0)
        
        if (numberedCells.length !== 1) continue
        
        const expectedSize = grid[numberedCells[0].row][numberedCells[0].col].value
        if (island.length !== expectedSize) {
          return false
        }
      }
    }
  }
  return true
}

export function hasOneNumberPerIsland(game: GameState): boolean {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const island = floodFill(grid, r, c, 'white', visited)
        const numberedCells = island.filter(pos => grid[pos.row][pos.col].value > 0)
        
        if (numberedCells.length !== 1) {
          return false
        }
      }
    }
  }
  return true
}

export function doIslandsNotTouch(game: GameState): boolean {
  return hasOneNumberPerIsland(game)
}

export function hasNoGrayCells(game: GameState): boolean {
  const { grid } = game
  for (const row of grid) {
    for (const cell of row) {
      if (cell.state === 'gray') {
        return false
      }
    }
  }
  return true
}

export function isValidCell(game: GameState, row: number, col: number): boolean {
  const { grid } = game
  const cell = grid[row][col]
  
  if (cell.state === 'gray') return true
  if (cell.value > 0 && cell.state !== 'white') return false
  
  const rows = grid.length
  const cols = grid[0].length
  
  for (let dr = -1; dr <= 0; dr++) {
    for (let dc = -1; dc <= 0; dc++) {
      const r = row + dr
      const c = col + dc
      if (r >= 0 && r < rows - 1 && c >= 0 && c < cols - 1) {
        if (
          grid[r][c].state === 'black' &&
          grid[r + 1][c].state === 'black' &&
          grid[r][c + 1].state === 'black' &&
          grid[r + 1][c + 1].state === 'black'
        ) {
          return false
        }
      }
    }
  }
  
  return true
}

export function isValidMatrix(game: GameState): boolean {
  return (
    hasNoGrayCells(game) &&
    hasNo2x2BlackSquare(game) &&
    areBlackCellsConnected(game) &&
    hasOneNumberPerIsland(game) &&
    isIslandSizeCorrect(game)
  )
}

export function getRuleViolations(game: GameState): string[] {
  const violations: string[] = []
  
  if (!hasNoGrayCells(game)) violations.push('Puzzle not complete: gray cells remain')
  if (!hasNo2x2BlackSquare(game)) violations.push('Rule violated: 2x2 black square exists')
  if (!areBlackCellsConnected(game)) violations.push('Rule violated: black cells are not all connected')
  if (!hasOneNumberPerIsland(game)) violations.push('Rule violated: island has zero or multiple numbers')
  if (!isIslandSizeCorrect(game)) violations.push('Rule violated: island size does not match its number')
  
  return violations
}

/** Mark all numbered cells as white - they are the starting points of islands */
export function markNumberedCellsWhite(game: GameState): GameState {
  const newGrid = game.grid.map(row =>
    row.map(cell => {
      if (cell.value > 0 && cell.state === 'gray') {
        return { ...cell, state: 'white' as CellState }
      }
      return cell
    })
  )
  return { ...game, grid: newGrid }
}

/**
 * Fill cells that must be black:
 * - Neighbors of "1" islands (complete islands)
 * - Cells between two numbered cells (would connect different islands)
 * - Neighbors of completed islands
 */
export function fillMandatoryBlack(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const cellsToBlack: Set<string> = new Set()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid[r][c]

      if (cell.value === 1) {
        const neighbors = getNeighbors(r, c, rows, cols)
        for (const n of neighbors) {
          if (grid[n.row][n.col].state === 'gray' && grid[n.row][n.col].value === 0) {
            cellsToBlack.add(`${n.row},${n.col}`)
          }
        }
      }

      if (cell.state === 'gray' && cell.value === 0) {
        const neighbors = getNeighbors(r, c, rows, cols)
        const numberedNeighbors = neighbors.filter(n => grid[n.row][n.col].value > 0)
        if (numberedNeighbors.length >= 2) {
          cellsToBlack.add(`${r},${c}`)
        }
      }
    }
  }

  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const island = floodFill(grid, r, c, 'white', visited)
        const numberedCells = island.filter(pos => grid[pos.row][pos.col].value > 0)
        if (numberedCells.length !== 1) continue
        
        const expectedSize = grid[numberedCells[0].row][numberedCells[0].col].value
        if (island.length === expectedSize) {
          for (const pos of island) {
            const neighbors = getNeighbors(pos.row, pos.col, rows, cols)
            for (const n of neighbors) {
              if (grid[n.row][n.col].state === 'gray' && grid[n.row][n.col].value === 0) {
                cellsToBlack.add(`${n.row},${n.col}`)
              }
            }
          }
        }
      }
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cellsToBlack.has(`${r},${c}`)) {
        return { ...cell, state: 'black' as CellState }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

/** If an incomplete island has only one direction to expand, it must expand there */
export function fillMandatoryWhite(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  const cellsToWhite: Set<string> = new Set()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const island = floodFill(grid, r, c, 'white', visited)
        const numberedCells = island.filter(pos => grid[pos.row][pos.col].value > 0)
        if (numberedCells.length !== 1) continue
        
        const expectedSize = grid[numberedCells[0].row][numberedCells[0].col].value
        if (island.length >= expectedSize) continue
        
        const grayNeighbors: Position[] = []
        for (const pos of island) {
          const neighbors = getNeighbors(pos.row, pos.col, rows, cols)
          for (const n of neighbors) {
            if (grid[n.row][n.col].state === 'gray') {
              if (!grayNeighbors.some(g => g.row === n.row && g.col === n.col)) {
                grayNeighbors.push(n)
              }
            }
          }
        }
        
        if (grayNeighbors.length === 1) {
          cellsToWhite.add(`${grayNeighbors[0].row},${grayNeighbors[0].col}`)
        }
      }
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cellsToWhite.has(`${r},${c}`)) {
        return { ...cell, state: 'white' as CellState }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

/** Cells that no island can reach (too far from all islands) must be black */
export function fillUnreachable(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0

  const visitedIslands = Array.from({ length: rows }, () => Array(cols).fill(false))
  const islandMap: Map<string, Position[]> = new Map()
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visitedIslands[r][c]) {
        const island = floodFill(grid, r, c, 'white', visitedIslands)
        const numbered = island.find(pos => grid[pos.row][pos.col].value > 0)
        if (numbered) {
          islandMap.set(`${numbered.row},${numbered.col}`, island)
        }
      }
    }
  }

  const islandReach: Set<string> = new Set()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid[r][c]
      if (cell.value > 0 && cell.state === 'white') {
        const island = islandMap.get(`${r},${c}`) || [{ row: r, col: c }]
        const remainingCapacity = cell.value - island.length
        
        if (remainingCapacity <= 0) continue
        
        const reachable = getReachableCellsFromIsland(grid, island, remainingCapacity, rows, cols)
        for (const pos of reachable) {
          islandReach.add(`${pos.row},${pos.col}`)
        }
      }
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cell.state === 'gray' && cell.value === 0) {
        if (!islandReach.has(`${r},${c}`)) {
          return { ...cell, state: 'black' as CellState }
        }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

function getReachableCellsFromIsland(
  grid: Cell[][],
  island: Position[],
  remainingCapacity: number,
  rows: number,
  cols: number
): Position[] {
  const reachable: Position[] = []
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  
  for (const pos of island) {
    visited[pos.row][pos.col] = true
  }
  
  const queue: { row: number; col: number; dist: number }[] = []
  
  for (const pos of island) {
    const neighbors = getNeighbors(pos.row, pos.col, rows, cols)
    for (const n of neighbors) {
      if (grid[n.row][n.col].state === 'gray' && !visited[n.row][n.col]) {
        queue.push({ row: n.row, col: n.col, dist: 1 })
      }
    }
  }
  
  while (queue.length > 0) {
    const { row, col, dist } = queue.shift()!
    
    if (visited[row][col]) continue
    if (dist > remainingCapacity) continue
    if (grid[row][col].state !== 'gray') continue
    
    visited[row][col] = true
    reachable.push({ row, col })
    
    const neighbors = getNeighbors(row, col, rows, cols)
    for (const n of neighbors) {
      if (!visited[n.row][n.col] && grid[n.row][n.col].state === 'gray') {
        queue.push({ row: n.row, col: n.col, dist: dist + 1 })
      }
    }
  }
  
  return reachable
}

/** If making a gray cell black would create a 2x2 black square, it must be white */
export function preventTwoByTwo(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const cellsToWhite: Set<string> = new Set()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state !== 'gray') continue
      
      const neighbors = getNeighbors(r, c, rows, cols)
      const hasBlackNeighbor = neighbors.some(n => grid[n.row][n.col].state === 'black')
      if (!hasBlackNeighbor) continue
      
      const wouldCreate2x2 = [
        [[0, 1], [1, 0], [1, 1]],
        [[0, -1], [1, -1], [1, 0]],
        [[-1, 0], [-1, 1], [0, 1]],
        [[-1, -1], [-1, 0], [0, -1]],
      ].some(checks => {
        return checks.every(([dr, dc]) => {
          const nr = r + dr
          const nc = c + dc
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) return false
          return grid[nr][nc].state === 'black'
        })
      })

      if (wouldCreate2x2) {
        cellsToWhite.add(`${r},${c}`)
      }
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cellsToWhite.has(`${r},${c}`)) {
        return { ...cell, state: 'white' as CellState }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

/** Isolated white cells without a number must connect to an island - if only one path, take it */
export function expandIsolatedWhite(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  const cellsToWhite: Set<string> = new Set()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'white' && !visited[r][c]) {
        const region = floodFill(grid, r, c, 'white', visited)
        const hasNumber = region.some(pos => grid[pos.row][pos.col].value > 0)
        
        if (hasNumber) continue
        
        const grayNeighbors: Position[] = []
        for (const pos of region) {
          const neighbors = getNeighbors(pos.row, pos.col, rows, cols)
          for (const n of neighbors) {
            if (grid[n.row][n.col].state === 'gray') {
              if (!grayNeighbors.some(g => g.row === n.row && g.col === n.col)) {
                grayNeighbors.push(n)
              }
            }
          }
        }
        
        if (grayNeighbors.length === 1) {
          cellsToWhite.add(`${grayNeighbors[0].row},${grayNeighbors[0].col}`)
        }
      }
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cellsToWhite.has(`${r},${c}`)) {
        return { ...cell, state: 'white' as CellState }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

/** Isolated black regions must connect to the main sea - if only one path, take it */
export function expandIsolatedBlack(game: GameState): GameState {
  const { grid } = game
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false))
  const blackRegions: Position[][] = []
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c].state === 'black' && !visited[r][c]) {
        blackRegions.push(floodFill(grid, r, c, 'black', visited))
      }
    }
  }
  
  if (blackRegions.length <= 1) return game
  
  let mainSeaIndex = 0
  let maxSize = 0
  for (let i = 0; i < blackRegions.length; i++) {
    if (blackRegions[i].length > maxSize) {
      maxSize = blackRegions[i].length
      mainSeaIndex = i
    }
  }
  
  const cellsToBlack: Set<string> = new Set()
  
  for (let i = 0; i < blackRegions.length; i++) {
    if (i === mainSeaIndex) continue
    
    const region = blackRegions[i]
    const grayNeighbors: Position[] = []
    
    for (const pos of region) {
      const neighbors = getNeighbors(pos.row, pos.col, rows, cols)
      for (const n of neighbors) {
        if (grid[n.row][n.col].state === 'gray' && grid[n.row][n.col].value === 0) {
          if (!grayNeighbors.some(g => g.row === n.row && g.col === n.col)) {
            grayNeighbors.push(n)
          }
        }
      }
    }
    
    if (grayNeighbors.length === 1) {
      cellsToBlack.add(`${grayNeighbors[0].row},${grayNeighbors[0].col}`)
    }
  }

  const newGrid = grid.map((row, r) =>
    row.map((cell, c) => {
      if (cellsToBlack.has(`${r},${c}`)) {
        return { ...cell, state: 'black' as CellState }
      }
      return cell
    })
  )

  return { ...game, grid: newGrid }
}

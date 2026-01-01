import { useState, useRef } from 'react'
import { GameState, PuzzleData, CellState } from './types'
import { puzzleData, createGameState } from './puzzles'
import { PuzzleList } from './components/PuzzleList'
import { Grid } from './components/Grid'
import { solveNurikabe, solveNurikabeAnimated } from './solver'
import { isValidMatrix, getRuleViolations, markNumberedCellsWhite, fillMandatoryBlack, fillMandatoryWhite, fillUnreachable, preventTwoByTwo, expandIsolatedWhite, expandIsolatedBlack } from './utils'

type ValidationResult = { valid: boolean; violations: string[] }

function App() {
  const [game, setGame] = useState<GameState | null>(null)
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null)
  const [solveTime, setSolveTime] = useState<number | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState(100)
  const abortControllerRef = useRef<AbortController | null>(null)

  const handleCellClick = (rowIdx: number, colIdx: number) => {
    if (!game) return
    setValidationResult(null)
    
    setGame(prev => {
      if (!prev) return prev
      const newGrid = prev.grid.map((row, r) =>
        row.map((cell, c) => {
          if (r === rowIdx && c === colIdx) {
            const nextState: CellState = 
              cell.state === 'gray' ? 'white' :
              cell.state === 'white' ? 'black' : 'gray'
            return { ...cell, state: nextState }
          }
          return cell
        })
      )
      return { ...prev, grid: newGrid }
    })
  }

  const handleSolve = () => {
    if (!game || isAnimating) return
    setValidationResult(null)
    setSolveTime(null)
    
    const startTime = performance.now()
    const solved = solveNurikabe(game)
    const endTime = performance.now()
    
    setSolveTime(endTime - startTime)
    setGame(solved)
  }

  const handleVisualize = async () => {
    if (!game || isAnimating) return
    
    setValidationResult(null)
    setSolveTime(null)
    setIsAnimating(true)
    
    abortControllerRef.current = new AbortController()
    const startTime = performance.now()
    
    const result = await solveNurikabeAnimated(
      game,
      (state) => {
        setGame(state)
      },
      animationSpeed,
      abortControllerRef.current.signal
    )
    
    const endTime = performance.now()
    
    if (result) {
      setSolveTime(endTime - startTime)
      setGame(result)
    }
    
    setIsAnimating(false)
    abortControllerRef.current = null
  }

  const handleStopAnimation = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsAnimating(false)
    }
  }

  const handleCheck = () => {
    if (!game) return
    const valid = isValidMatrix(game)
    const violations = getRuleViolations(game)
    setValidationResult({ valid, violations })
  }

  const handleInitialIslands = () => {
    if (!game) return
    setValidationResult(null)
    setGame(markNumberedCellsWhite(game))
  }

  const handleFillMandatoryBlack = () => {
    if (!game) return
    setValidationResult(null)
    setGame(fillMandatoryBlack(game))
  }

  const handleFillMandatoryWhite = () => {
    if (!game) return
    setValidationResult(null)
    setGame(fillMandatoryWhite(game))
  }

  const handleFillUnreachable = () => {
    if (!game) return
    setValidationResult(null)
    setGame(fillUnreachable(game))
  }

  const handlePreventTwoByTwo = () => {
    if (!game) return
    setValidationResult(null)
    setGame(preventTwoByTwo(game))
  }

  const handleExpandIsolatedWhite = () => {
    if (!game) return
    setValidationResult(null)
    setGame(expandIsolatedWhite(game))
  }

  const handleExpandIsolatedBlack = () => {
    if (!game) return
    setValidationResult(null)
    setGame(expandIsolatedBlack(game))
  }

  const loadPuzzle = (data: PuzzleData) => {
    setGame(createGameState(data))
    setValidationResult(null)
    setSolveTime(null)
  }

  return (
    <div className="container">
      <h1>Nurikabe</h1>
      
      {!game ? (
        <PuzzleList puzzles={puzzleData} onSelect={loadPuzzle} />
      ) : (
        <Grid 
          game={game} 
          onCellClick={handleCellClick} 
          onBack={() => { setGame(null); setValidationResult(null); setSolveTime(null) }}
          onSolve={handleSolve}
          onVisualize={handleVisualize}
          onStopAnimation={handleStopAnimation}
          isAnimating={isAnimating}
          animationSpeed={animationSpeed}
          onSpeedChange={setAnimationSpeed}
          onCheck={handleCheck}
          onInitialIslands={handleInitialIslands}
          onFillMandatoryBlack={handleFillMandatoryBlack}
          onFillMandatoryWhite={handleFillMandatoryWhite}
          onFillUnreachable={handleFillUnreachable}
          onPreventTwoByTwo={handlePreventTwoByTwo}
          onExpandIsolatedWhite={handleExpandIsolatedWhite}
          onExpandIsolatedBlack={handleExpandIsolatedBlack}
          validationResult={validationResult}
          solveTime={solveTime}
        />
      )}
    </div>
  )
}

export default App

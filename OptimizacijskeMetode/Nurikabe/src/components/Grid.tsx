import { GameState } from '../types'

type Props = {
  game: GameState
  onCellClick: (row: number, col: number) => void
  onBack: () => void
  onSolve: () => void
  onVisualize: () => void
  onStopAnimation: () => void
  isAnimating: boolean
  animationSpeed: number
  onSpeedChange: (speed: number) => void
  onCheck: () => void
  onInitialIslands: () => void
  onFillMandatoryBlack: () => void
  onFillMandatoryWhite: () => void
  onFillUnreachable: () => void
  onPreventTwoByTwo: () => void
  onExpandIsolatedWhite: () => void
  onExpandIsolatedBlack: () => void
  validationResult: { valid: boolean; violations: string[] } | null
  solveTime: number | null
}

export function Grid({ game, onCellClick, onBack, onSolve, onVisualize, onStopAnimation, isAnimating, animationSpeed, onSpeedChange, onCheck, onInitialIslands, onFillMandatoryBlack, onFillMandatoryWhite, onFillUnreachable, onPreventTwoByTwo, onExpandIsolatedWhite, onExpandIsolatedBlack, validationResult, solveTime }: Props) {
  return (
    <div className="puzzle-view">
      <div className="toolbar">
        <button className="back-btn" onClick={onBack} disabled={isAnimating}>
          ← Back
        </button>
        <button className="check-btn" onClick={onCheck} disabled={isAnimating}>
          Check
        </button>
        <button className="solve-btn" onClick={onSolve} disabled={isAnimating}>
          Solve
        </button>
        {isAnimating ? (
          <button className="stop-btn" onClick={onStopAnimation}>
            ■ Stop
          </button>
        ) : (
          <button className="visualize-btn" onClick={onVisualize}>
            ▶ Visualize
          </button>
        )}
      </div>
      
      <div className="speed-control">
        <label>Speed:</label>
        <input
          type="range"
          min="10"
          max="500"
          value={510 - animationSpeed}
          onChange={(e) => onSpeedChange(510 - Number(e.target.value))}
          disabled={isAnimating}
        />
        <span className="speed-label">
          {animationSpeed <= 50 ? 'Fast' : animationSpeed <= 150 ? 'Medium' : animationSpeed <= 300 ? 'Slow' : 'Very Slow'}
        </span>
      </div>
      
      <div className="helper-toolbar">
        <span className="toolbar-label">Islands:</span>
        <button className="helper-btn white-btn" onClick={onInitialIslands}>
          Initial Islands
        </button>
        <button className="helper-btn white-btn" onClick={onFillMandatoryWhite}>
          Expand Forced
        </button>
        <button className="helper-btn white-btn" onClick={onPreventTwoByTwo}>
          Prevent 2x2
        </button>
        <button className="helper-btn white-btn" onClick={onExpandIsolatedWhite}>
          Expand Isolated
        </button>
      </div>
      
      <div className="helper-toolbar">
        <span className="toolbar-label">Water:</span>
        <button className="helper-btn black-btn" onClick={onFillMandatoryBlack}>
          Fill Mandatory
        </button>
        <button className="helper-btn black-btn" onClick={onFillUnreachable}>
          Fill Unreachable
        </button>
        <button className="helper-btn black-btn" onClick={onExpandIsolatedBlack}>
          Connect Sea
        </button>
      </div>
      
      <h2>{game.name}</h2>
      <p className="hint">Click cells to toggle: gray → white → black</p>
      
      {solveTime !== null && (
        <div className="solve-time">
          ⏱ Solved in <strong>{solveTime < 1000 ? `${solveTime.toFixed(2)}ms` : `${(solveTime / 1000).toFixed(2)}s`}</strong>
        </div>
      )}
      
      {validationResult && (
        <div className={`validation-result ${validationResult.valid ? 'valid' : 'invalid'}`}>
          {validationResult.valid ? (
            <span>✓ Puzzle is valid!</span>
          ) : (
            <div>
              <span>✗ Invalid:</span>
              <ul>
                {validationResult.violations.map((v, i) => (
                  <li key={i}>{v}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      
      <div className="grid">
        {game.grid.map((row, rowIdx) => (
          <div key={rowIdx} className="row">
            {row.map((cell, colIdx) => (
              <div
                key={colIdx}
                className={`cell ${cell.state}`}
                onClick={() => onCellClick(rowIdx, colIdx)}
              >
                {cell.value > 0 ? cell.value : ''}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

import React from 'react';
import type { SimulationParams } from '../types';

interface ControlsProps {
  paramsArr: SimulationParams[];
  setParamsArr: React.Dispatch<React.SetStateAction<SimulationParams[]>>;
  onStart: () => void;
  isRunning: boolean;
  colors: string[];
}

export const Controls: React.FC<ControlsProps> = ({ paramsArr, setParamsArr, onStart, isRunning, colors }) => {

  const handleChange = (typeId: number, field: keyof SimulationParams, value: string) => {
    let numVal = parseFloat(value);
    if (isNaN(numVal)) numVal = 0;
    const newArr = [...paramsArr];
    newArr[typeId] = { ...newArr[typeId], [field]: numVal };
    setParamsArr(newArr);
  };

  return (
    <div>
      <div className="controls-grid">
        {paramsArr.map((params, idx) => (
          <div key={idx} className="species-control" style={{ borderTop: `4px solid ${colors[idx]}` }}>
            <h3>Species {idx + 1}</h3>

            <div className="input-group">
              <label>Starting Pop (st): </label>
              <input type="number" value={params.st} onChange={e => handleChange(idx, 'st', e.target.value)} />
            </div>

            <div className="input-group">
              <label>Reproduction (R): </label>
              <input type="number" step="0.01" value={params.r} onChange={e => handleChange(idx, 'r', e.target.value)} />
            </div>

            <div className="input-group">
              <label>Death Prob (S): </label>
              <input type="number" step="0.01" value={params.s} onChange={e => handleChange(idx, 's', e.target.value)} />
            </div>

            <div className="input-group">
              <label>Capacity Coef (K): </label>
              <input type="number" step="0.0001" value={params.k} onChange={e => handleChange(idx, 'k', e.target.value)} />
            </div>
          </div>
        ))}
      </div>

      <button onClick={onStart}>
        {isRunning ? 'Restart Simulation' : 'Start Simulation'}
      </button>
    </div>
  );
};

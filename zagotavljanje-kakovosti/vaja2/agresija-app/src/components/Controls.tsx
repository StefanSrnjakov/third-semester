import React from 'react';
import type { SimulationParams } from '../types';

interface ControlsProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  onStart: () => void;
  isRunning: boolean;
}

export const Controls: React.FC<ControlsProps> = ({ params, setParams, onStart, isRunning }) => {

  const handleChange = (field: keyof SimulationParams, value: string) => {
    let numVal = parseFloat(value);
    if (isNaN(numVal)) numVal = 0;
    setParams({ ...params, [field]: numVal });
  };

  return (
    <div>
      <div className="controls-grid">
        <div className="global-control">
          <h3>Simulation Parameters</h3>

          <div className="input-group">
            <label>Initial Peaceful: </label>
            <input type="number" value={params.initialPeaceful} onChange={e => handleChange('initialPeaceful', e.target.value)} />
          </div>

          <div className="input-group">
            <label>Initial Aggressive: </label>
            <input type="number" value={params.initialAggressive} onChange={e => handleChange('initialAggressive', e.target.value)} />
          </div>

          <div className="input-group">
            <label>Food Pairs (per gen): </label>
            <input type="number" value={params.foodPairs} onChange={e => handleChange('foodPairs', e.target.value)} />
          </div>

          <div className="input-group">
            <label>Simulation Speed: </label>
            <input type="range" min="1" max="100" value={params.speed} onChange={e => handleChange('speed', e.target.value)} />
            <span>{params.speed}</span>
          </div>
        </div>
      </div>

      <button onClick={onStart}>
        {isRunning ? 'Restart Simulation' : 'Start Simulation'}
      </button>
    </div>
  );
};

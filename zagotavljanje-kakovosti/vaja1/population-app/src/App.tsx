import { useState, useCallback } from 'react';
import { useSimulation } from './hooks/useSimulation';
import type { SimulationParams } from './types';
import { SimulationView } from './components/SimulationView';
import { GraphView } from './components/GraphView';
import { Controls } from './components/Controls';
import './index.css';

const DEFAULT_PARAMS: SimulationParams[] = [
  { st: 50, r: 0.05, s: 0.01, k: 0.0001 },
  { st: 50, r: 0.08, s: 0.02, k: 0.0002 },
  { st: 50, r: 0.1, s: 0.05, k: 0.00025 }
];

function App() {
  const [paramsArr, setParamsArr] = useState<SimulationParams[]>(DEFAULT_PARAMS);
  const [isRunning, setIsRunning] = useState(false);

  const { creatures, history, initSimulation, SIMULATION_WIDTH, SIMULATION_HEIGHT, COLORS } = useSimulation(paramsArr, isRunning);

  const handleStart = useCallback(() => {
    setIsRunning(false);
    setTimeout(() => {
      initSimulation();
      setIsRunning(true);
    }, 50);
  }, [initSimulation]);

  return (
    <div className="app-container">
      <h1>Population Growth Simulation</h1>

      <div className="top-panes">
        <SimulationView
          creatures={creatures}
          width={SIMULATION_WIDTH}
          height={SIMULATION_HEIGHT}
          colors={COLORS}
        />
        <GraphView
          history={history}
          width={SIMULATION_WIDTH}
          height={SIMULATION_HEIGHT}
          colors={COLORS}
        />
      </div>

      <Controls
        paramsArr={paramsArr}
        setParamsArr={setParamsArr}
        onStart={handleStart}
        isRunning={isRunning}
        colors={COLORS}
      />
    </div>
  );
}

export default App;

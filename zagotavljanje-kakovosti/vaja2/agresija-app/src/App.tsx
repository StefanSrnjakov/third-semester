import { useState, useCallback } from 'react';
import { useSimulation } from './hooks/useSimulation';
import type { SimulationParams } from './types';
import { SimulationView } from './components/SimulationView';
import { GraphView } from './components/GraphView';
import { Controls } from './components/Controls';
import './index.css';

const DEFAULT_PARAMS: SimulationParams = {
  initialPeaceful: 50,
  initialAggressive: 50,
  foodPairs: 60,
  speed: 2
};

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  const [isRunning, setIsRunning] = useState(false);

  const { creatures, foodPairs, history, initSimulation, SIMULATION_WIDTH, SIMULATION_HEIGHT } = useSimulation(params, isRunning);

  const handleStart = useCallback(() => {
    setIsRunning(false);
    setTimeout(() => {
      initSimulation();
      setIsRunning(true);
    }, 50);
  }, [initSimulation]);

  return (
    <div className="app-container">
      <h1>Evolucija Agresije (Game Theory)</h1>

      <div className="top-panes">
        <SimulationView
          creatures={creatures}
          foodPairs={foodPairs}
          width={SIMULATION_WIDTH}
          height={SIMULATION_HEIGHT}
        />
        <GraphView
          history={history}
          width={SIMULATION_WIDTH}
          height={SIMULATION_HEIGHT}
        />
      </div>

      <Controls
        params={params}
        setParams={setParams}
        onStart={handleStart}
        isRunning={isRunning}
      />
    </div>
  );
}

export default App;

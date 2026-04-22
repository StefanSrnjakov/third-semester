import React, { useState } from 'react';
import { SimulationParams } from './types';
import { useSimulation } from './hooks/useSimulation';
import { SimulationCanvas } from './components/SimulationCanvas';
import { HistoryChart } from './components/HistoryChart';
import { COLORS, SIM_CONSTANTS } from './constants';

const DEFAULT: SimulationParams = {
  width: 40,
  height: 40,
  initialPrey: 35,
  initialPredators: 10,
  initialFood: 60,
  selectedTerrain: 'lake',
  simulationSpeed: 10,
  hungerTickBase: 0.25,
  thirstTickBase: 0.3,
  reproductionInc: 0.2,
  perceptionMult: 3,
  agingMult: 1.0,
};

const App: React.FC = () => {
  const [params, setParams] = useState<SimulationParams>(DEFAULT);
  const [isRunning, setIsRunning] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { state, init } = useSimulation(params, isRunning);

  const populationData = state.deathHistory.map(h => ({
    day: h.day,
    prey: h.preyTraits.population,
    predator: h.predatorTraits.population,
    food: h.foodCount
  }));

  const birthData = state.deathHistory.map(h => ({
    day: h.day,
    prey: h.preyTraits.births,
    predator: h.predatorTraits.births
  }));

  return (
    <div style={{ display: 'flex', gap: '20px', padding: '20px', fontFamily: 'system-ui, sans-serif', background: '#f0f2f5', minHeight: '100vh' }}>
      {/* Sidebar Controls */}
      <div style={{ width: '320px', flexShrink: 0, padding: '20px', background: '#fff', borderRadius: '12px', border: '1px solid #e1e4e8', height: 'fit-content', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
        <h2 style={{ marginTop: 0, color: '#1a1a1a', fontSize: '1.25em' }}>Ecosystem Lab v4.1</h2>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <button onClick={() => { init(); setIsRunning(true); }} style={{ flex: 1, padding: '12px', background: '#2980b9', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', transition: 'all 0.2s' }}>
            Initialize
          </button>
          <button onClick={() => setIsRunning(!isRunning)} style={{ flex: 1, padding: '12px', background: isRunning ? '#e74c3c' : '#27ae60', border: 'none', color: 'white', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>
            {isRunning ? 'Pause' : 'Resume'}
          </button>
        </div>

        <div style={{ borderBottom: '1px solid #eee', marginBottom: '15px' }}></div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <label style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#444' }}>
            Terrain Strategy
            <select value={params.selectedTerrain} onChange={e => setParams({ ...params, selectedTerrain: e.target.value as any })} style={{ width: '100%', padding: '8px', marginTop: '6px', borderRadius: '4px', border: '1px solid #ccc' }}>
              <option value="lake">Large Lake</option>
              <option value="river">River Crossing</option>
              <option value="multi-lake">Multiple Lakes</option>
              <option value="puddles">Scattered Puddles</option>
            </select>
          </label>

          <label style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#444' }}>
            Prey Population ({params.initialPrey})
            <input type="range" min="5" max="150" value={params.initialPrey} onChange={e => setParams({ ...params, initialPrey: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
          </label>

          <label style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#444' }}>
            Predator Count ({params.initialPredators})
            <input type="range" min="0" max="50" value={params.initialPredators} onChange={e => setParams({ ...params, initialPredators: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
          </label>

          <label style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#444' }}>
            Food Abundance ({params.initialFood})
            <input type="range" min="10" max="200" value={params.initialFood} onChange={e => setParams({ ...params, initialFood: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
          </label>

          <label style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#444' }}>
            Simulation Speed ({params.simulationSpeed}x)
            <input type="range" min="1" max="200" value={params.simulationSpeed} onChange={e => setParams({ ...params, simulationSpeed: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
          </label>

          {/* Advanced Section */}
          <div style={{ marginTop: '10px' }}>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{ width: '100%', padding: '8px', background: '#f0f2f5', border: '1px solid #ddd', borderRadius: '6px', fontSize: '0.85em', cursor: 'pointer', fontWeight: 'bold', color: '#555', display: 'flex', justifyContent: 'space-between' }}
            >
              {showAdvanced ? '▼ Hide Advanced' : '▶ Advanced Parameters'}
            </button>

            {showAdvanced && (
              <div style={{ marginTop: '15px', padding: '15px', background: '#fcfcfc', border: '1px solid #eee', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <label style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#666' }}>
                  Hunger Rate ({params.hungerTickBase.toFixed(2)})
                  <input type="range" min="0.05" max="1.0" step="0.05" value={params.hungerTickBase} onChange={e => setParams({ ...params, hungerTickBase: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
                </label>
                <label style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#666' }}>
                  Thirst Rate ({params.thirstTickBase.toFixed(2)})
                  <input type="range" min="0.05" max="1.0" step="0.05" value={params.thirstTickBase} onChange={e => setParams({ ...params, thirstTickBase: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
                </label>
                <label style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#666' }}>
                  Reproduction Chance ({params.reproductionInc.toFixed(2)})
                  <input type="range" min="0.01" max="1.0" step="0.01" value={params.reproductionInc} onChange={e => setParams({ ...params, reproductionInc: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
                </label>
                <label style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#666' }}>
                  Perception Unit ({params.perceptionMult.toFixed(1)})
                  <input type="range" min="1.0" max="10.0" step="0.5" value={params.perceptionMult} onChange={e => setParams({ ...params, perceptionMult: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
                </label>
                <label style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#666' }}>
                  Aging Speed ({params.agingMult.toFixed(2)})
                  <input type="range" min="0.1" max="5.0" step="0.1" value={params.agingMult} onChange={e => setParams({ ...params, agingMult: +e.target.value })} style={{ width: '100%', marginTop: '6px' }} />
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Real-time Summary Card */}
        <div style={{ marginTop: '25px', padding: '15px', background: '#f8f9fa', borderRadius: '10px', border: '1px solid #eee' }}>
          <h4 style={{ margin: '0 0 12px 0', borderBottom: '1px solid #ddd', paddingBottom: '6px' }}>Current State (Day: {Math.floor(state.ticks / SIM_CONSTANTS.TICKS_PER_DAY)})</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.85em' }}>
            <div style={{ color: COLORS.prey, fontWeight: 'bold' }}>Rabbits: {state.creatures.filter(c => c.type === 'prey').length}</div>
            <div style={{ color: COLORS.predator, fontWeight: 'bold' }}>Foxes: {state.creatures.filter(c => c.type === 'predator').length}</div>
            <div style={{ gridColumn: 'span 2' }}>Veggie Density: {state.food.length} units</div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>

        {/* TOP ROW: Map & Basic Population Info */}
        <div style={{ display: 'flex', gap: '20px' }}>
          <div style={{ flex: '2', minWidth: 0 }}>
            <SimulationCanvas grid={state.grid} creatures={state.creatures} food={state.food} />
          </div>
          <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <HistoryChart
              title="Population Trends"
              yLabel="Size"
              data={populationData}
              lines={[
                { name: 'Prey', dataKey: 'prey', color: COLORS.prey },
                { name: 'Predator', dataKey: 'predator', color: COLORS.predator },
                { name: 'Plants', dataKey: 'food', color: COLORS.food },
              ]}
            />
            <HistoryChart
              title="Daily Births"
              yLabel="Count"
              data={birthData}
              lines={[
                { name: 'Prey Births', dataKey: 'prey', color: '#2ecc71' },
                { name: 'Predator Births', dataKey: 'predator', color: '#e67e22' },
              ]}
            />
          </div>
        </div>

        {/* MIDDLE ROW: Trait Evolution Graphs */}
        <div style={{ display: 'flex', gap: '20px' }}>
          <HistoryChart
            title="Prey Genes (Evolution)"
            data={state.deathHistory.map(h => ({ day: h.day, ...h.preyTraits }))}
            lines={[
              { name: 'Speed', dataKey: 'avgSpeed', color: '#3498db' },
              { name: 'Size', dataKey: 'avgSize', color: '#e74c3c' },
              { name: 'Perception', dataKey: 'avgPerception', color: '#f1c40f' },
            ]}
          />
          <HistoryChart
            title="Predator Genes (Evolution)"
            data={state.deathHistory.map(h => ({ day: h.day, ...h.predatorTraits }))}
            lines={[
              { name: 'Speed', dataKey: 'avgSpeed', color: '#2980b9' },
              { name: 'Size', dataKey: 'avgSize', color: '#c0392b' },
              { name: 'Perception', dataKey: 'avgPerception', color: '#d35400' },
            ]}
          />
        </div>

        {/* THIRD ROW: Mortality Analysis */}
        <div style={{ display: 'flex', gap: '20px' }}>
          <HistoryChart
            title="Mortality (Prey)"
            data={state.deathHistory.map(h => ({ day: h.day, ...h.prey }))}
            lines={[
              { name: 'Starvation', dataKey: 'starvation', color: '#8e44ad' },
              { name: 'Thirst', dataKey: 'thirst', color: '#3498db' },
              { name: 'Predation', dataKey: 'eaten', color: '#e74c3c' },
              { name: 'Age', dataKey: 'age', color: '#95a5a6' },
            ]}
          />
          <HistoryChart
            title="Mortality (Predators)"
            data={state.deathHistory.map(h => ({ day: h.day, ...h.predator }))}
            lines={[
              { name: 'Starvation', dataKey: 'starvation', color: '#8e44ad' },
              { name: 'Thirst', dataKey: 'thirst', color: '#3498db' },
              { name: 'Age', dataKey: 'age', color: '#95a5a6' },
            ]}
          />
        </div>

        {/* BOTTOM ROW: Detailed Live List */}
        <div style={{ background: '#fff', border: '1px solid #ddd', borderRadius: '12px', padding: '20px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
          <h3 style={{ marginTop: 0, fontSize: '1.1em', color: '#2c3e50' }}>Live Inhabitant Registry</h3>
          <div style={{ maxHeight: '250px', overflowY: 'auto', borderRadius: '8px', border: '1px solid #f0f0f0' }}>
            <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '0.85em' }}>
              <thead style={{ position: 'sticky', top: 0, background: '#f8f9fa', borderBottom: '2px solid #eee' }}>
                <tr>
                  <th style={{ padding: '12px' }}>ID</th>
                  <th style={{ padding: '12px' }}>Type</th>
                  <th style={{ padding: '12px' }}>Age Status</th>
                  <th style={{ padding: '12px' }}>Needs (H / T / R)</th>
                  <th style={{ padding: '12px' }}>Gender</th>
                  <th style={{ padding: '12px' }}>Traits (S / Sz / P)</th>
                </tr>
              </thead>
              <tbody>
                {state.creatures
                  .sort((a, b) => b.age - a.age)
                  .map(c => (
                    <tr key={c.id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '10px', fontFamily: 'monospace', fontWeight: 'bold', color: '#7f8c8d' }}>{c.id.toUpperCase()}</td>
                      <td style={{ padding: '10px' }}>
                        <span style={{ padding: '3px 8px', borderRadius: '12px', background: c.type === 'predator' ? '#fededb' : '#e2f9e1', color: c.type === 'predator' ? '#e74c3c' : '#27ae60', fontWeight: 'bold' }}>
                          {c.type === 'predator' ? 'Fox' : 'Rabbit'}
                        </span>
                      </td>
                      <td style={{ padding: '10px' }}>{c.age.toFixed(1)} / {Math.round(c.maxAge)}d</td>
                      <td style={{ padding: '10px' }}>
                        <span style={{ color: c.hunger > SIM_CONSTANTS.NEED_PRIORITY.HUNGER ? '#e74c3c' : '#7f8c8d' }}>{Math.round(c.hunger)}</span> /
                        <span style={{ color: c.thirst > SIM_CONSTANTS.NEED_PRIORITY.THIRST ? '#3498db' : '#7f8c8d' }}> {Math.round(c.thirst)}</span> /
                        <span style={{ color: c.reproductionDesire > SIM_CONSTANTS.NEED_PRIORITY.REPRODUCTION ? '#27ae60' : '#7f8c8d' }}> {Math.round(c.reproductionDesire)}%</span>
                      </td>
                      <td style={{ padding: '10px' }}>{c.gender === 'male' ? 'Male' : 'Female'}</td>
                      <td style={{ padding: '10px' }}>{c.speed.toFixed(1)} / {c.size.toFixed(1)} / {c.perception.toFixed(1)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;

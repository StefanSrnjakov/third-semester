import React, { useState } from 'react';
import { useSimulation } from './hooks/useSimulation';
import { SimulationParams } from './types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const App: React.FC = () => {
  const [params, setParams] = useState<SimulationParams>({
    initialCount: 20,
    initialFood: 50,
    initialEnergy: 1000,
    initialSpeed: 1,
    initialSize: 1,
    initialPerception: 5,
    foodScenario: 'constant',
    width: 600,
    height: 400,
    simulationSpeed: 1,
  });

  const [isRunning, setIsRunning] = useState(false);
  const { state, init } = useSimulation(params, isRunning);

  const handleStart = () => {
    init();
    setIsRunning(true);
  };

  const activeCreatures = state.creatures
    .filter(c => !c.isDead && (!c.isReturning || c.y > 10))
    .slice(0, 10);

  return (
    <div style={{ padding: '10px' }}>
      <h1>Evolucija</h1>
      
      <div style={{ display: 'flex', gap: '20px' }}>
        {/* Controls */}
        <div style={{ border: '1px solid #ccc', padding: '10px', minWidth: '220px' }}>
          <h3>Nastavitve</h3>
          <div>
            <label>Bitij: </label><br/>
            <input type="number" value={params.initialCount} onChange={e => setParams({ ...params, initialCount: +e.target.value })} style={{ width: '80%' }}/><br/>
            
            <label>Hrana: </label><br/>
            <input type="number" value={params.initialFood} onChange={e => setParams({ ...params, initialFood: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Zacetna energija: </label><br/>
            <input type="number" value={params.initialEnergy} onChange={e => setParams({ ...params, initialEnergy: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Hitrost: </label><br/>
            <input type="number" step="0.1" value={params.initialSpeed} onChange={e => setParams({ ...params, initialSpeed: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Velikost: </label><br/>
            <input type="number" step="0.1" value={params.initialSize} onChange={e => setParams({ ...params, initialSize: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Zaznava: </label><br/>
            <input type="number" value={params.initialPerception} onChange={e => setParams({ ...params, initialPerception: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Sirina polja: </label><br/>
            <input type="number" value={params.width} onChange={e => setParams({ ...params, width: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Visina polja: </label><br/>
            <input type="number" value={params.height} onChange={e => setParams({ ...params, height: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Hitrost simulacije: </label><br/>
            <input type="number" min="1" max="100" value={params.simulationSpeed} onChange={e => setParams({ ...params, simulationSpeed: +e.target.value })} style={{ width: '80%' }}/><br/>

            <label>Scenarij: </label><br/>
            <select value={params.foodScenario} onChange={e => setParams({ ...params, foodScenario: e.target.value as any })} style={{ width: '80%' }}>
              <option value="constant">Konstantna hrana</option>
              <option value="decreasing">Padajoča hrana</option>
            </select><br/><br/>

            <button onClick={handleStart} style={{ padding: '8px', cursor: 'pointer' }}>Zacetek</button>
            <button onClick={() => setIsRunning(!isRunning)} style={{ marginLeft: '10px', padding: '8px', cursor: 'pointer' }}>
                {isRunning ? 'Pavza' : 'Nadaljuj'}
            </button>
          </div>

          <div style={{ marginTop: '20px' }}>
            <strong>Gen:</strong> {state.generation}<br/>
            <strong>Zivih:</strong> {state.creatures.filter(c => !c.isDead && (!c.isReturning || c.y > 10)).length}<br/>
            <strong>Hrana:</strong> {state.foods.length}
          </div>
        </div>

        {/* Animation & List */}
        <div style={{ flex: 1 }}>
          <div style={{ position: 'relative', width: params.width, height: params.height, border: '1px solid black', backgroundColor: 'white' }}>
            {state.foods.map(f => (
              <div key={f.id} style={{ position: 'absolute', left: f.x, top: f.y, width: '4px', height: '4px', background: 'green', borderRadius: '50%' }} />
            ))}
            {state.creatures.map(c => {
                 const isSafe = c.isReturning && c.y <= 10;
                 if (isSafe && !c.isDead) return null;
                 return (
                    <div
                        key={c.id}
                        style={{
                            position: 'absolute',
                            left: c.x,
                            top: c.y,
                            width: c.size * 5,
                            height: c.size * 5,
                            background: c.isDead ? 'red' : c.isReturning ? 'blue' : 'black',
                            borderRadius: '50%',
                            opacity: 0.7,
                        }}
                    />
                 );
            })}
          </div>

          <div style={{ marginTop: '20px' }}>
            <h3>Aktivna Bitja (Top 10)</h3>
            <div style={{ height: '300px', overflowY: 'auto', border: '1px solid #ddd' }}>
              <table className="creature-list-table" style={{ margin: 0 }}>
                <thead style={{ position: 'sticky', top: 0, backgroundColor: 'white' }}>
                  <tr>
                    <th>ID</th>
                    <th>Energija</th>
                    <th>Hitrost</th>
                    <th>Velikost</th>
                    <th>Zaznava</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {activeCreatures.map(c => (
                    <tr key={c.id}>
                      <td>{c.id}</td>
                      <td>{Math.round(c.energy)}</td>
                      <td>{c.speed.toFixed(2)}</td>
                      <td>{c.size.toFixed(2)}</td>
                      <td>{c.perception.toFixed(2)}</td>
                      <td>{c.isReturning ? 'Vraca se' : 'Išče'}</td>
                    </tr>
                  ))}
                  {/* Fill empty rows for stability if fewer than 10 */}
                  {Array.from({ length: Math.max(0, 10 - activeCreatures.length) }).map((_, i) => (
                    <tr key={`empty-${i}`}>
                      <td colSpan={6}>&nbsp;</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>Grafov</h3>
        <div style={{ display: 'flex', gap: '20px' }}>
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
                <LineChart width={450} height={250} data={[...state.history]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="generation" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="count" stroke="blue" name="Populacija" />
                </LineChart>
            </div>
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
                <LineChart width={450} height={250} data={[...state.history]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="generation" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="avgSpeed" stroke="orange" name="Avg Hitrost" />
                    <Line type="monotone" dataKey="avgSize" stroke="green" name="Avg Velikost" />
                    <Line type="monotone" dataKey="avgPerception" stroke="purple" name="Avg Zaznava" />
                </LineChart>
            </div>
        </div>
      </div>
    </div>
  );
};

export default App;

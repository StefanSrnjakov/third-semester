import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Props {
  title: string;
  history: { day: number, age: number, thirst: number, starvation: number, eaten?: number }[];
}

export const DeathChart: React.FC<Props> = ({ title, history }) => {
  if (!history || history.length === 0) return <div style={{ flex: 1, padding: '20px', background: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}>Tracking {title}...</div>;

  return (
    <div style={{ flex: 1, background: '#fff', border: '1px solid #ddd', borderRadius: '8px', padding: '15px' }}>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '1em', textAlign: 'center' }}>{title}</h3>
      
      <div style={{ width: '100%', height: '180px' }}>
        <ResponsiveContainer>
          <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" tick={{ fontSize: 12 }} label={{ value: 'Time (Days)', position: 'insideBottomRight', offset: -5, fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} label={{ value: 'Deaths Count', angle: -90, position: 'insideLeft', fontSize: 12 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            
            <Line type="monotone" name="Old Age" dataKey="age" stroke="#95a5a6" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" name="Thirst" dataKey="thirst" stroke="#3498db" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" name="Starvation" dataKey="starvation" stroke="#e67e22" strokeWidth={2} dot={false} isAnimationActive={false} />
            {history[0].eaten !== undefined && (
              <Line type="monotone" name="Eaten" dataKey="eaten" stroke="#e74c3c" strokeWidth={2} dot={false} isAnimationActive={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

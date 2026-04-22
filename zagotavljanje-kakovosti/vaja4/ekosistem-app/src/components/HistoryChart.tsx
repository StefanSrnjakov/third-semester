import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface LineConfig {
  name: string;
  dataKey: string;
  color: string;
}

interface Props {
  title: string;
  data: any[];
  lines: LineConfig[];
  yLabel?: string;
  height?: number;
}

export const HistoryChart: React.FC<Props> = ({ title, data, lines, yLabel = 'Value', height = 180 }) => {
  if (!data || data.length === 0) return <div style={{ flex: 1, padding: '20px', background: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}>History for {title} arriving...</div>;

  return (
    <div style={{ flex: 1, background: '#fff', border: '1px solid #ddd', borderRadius: '8px', padding: '15px' }}>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '1em', textAlign: 'center' }}>{title}</h3>
      <div style={{ width: '100%', height: `${height}px` }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 10 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: '10px' }} />
            {lines.map(l => (
              <Line key={l.dataKey} type="monotone" name={l.name} dataKey={l.dataKey} stroke={l.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

import React, { useRef, useEffect } from 'react';
import type { Creature, FoodPair } from '../types';

interface SimulationViewProps {
  creatures: Creature[];
  foodPairs: FoodPair[];
  width: number;
  height: number;
}

const COLORS = {
  Peaceful: '#3b82f6', // Blue
  Aggressive: '#ef4444' // Red
};

export const SimulationView: React.FC<SimulationViewProps> = ({ creatures, foodPairs, width, height }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // Draw food pairs
    foodPairs.forEach(f => {
      ctx.beginPath();
      ctx.arc(f.x - 5, f.y, 4, 0, Math.PI * 2);
      ctx.arc(f.x + 5, f.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981'; // Green food
      ctx.fill();
    });

    // Draw creatures
    creatures.forEach(c => {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[c.type];
      ctx.fill();
    });
  }, [creatures, foodPairs, width, height]);

  const peacefulCount = creatures.filter(c => c.type === 'Peaceful').length;
  const aggressiveCount = creatures.filter(c => c.type === 'Aggressive').length;

  return (
    <div>
      <h2>
        Simulation Area
        <span style={{ color: COLORS.Peaceful, marginLeft: '10px' }}>Peaceful: {peacefulCount}</span>
        <span style={{ color: COLORS.Aggressive, marginLeft: '10px' }}>Aggressive: {aggressiveCount}</span>
      </h2>
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
};

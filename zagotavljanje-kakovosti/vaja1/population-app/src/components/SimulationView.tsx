import React, { useRef, useEffect } from 'react';
import type { Creature } from '../types';

interface SimulationViewProps {
  creatures: Creature[];
  width: number;
  height: number;
  colors: string[];
}

export const SimulationView: React.FC<SimulationViewProps> = ({ creatures, width, height, colors }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    creatures.forEach(c => {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = colors[c.typeId];
      ctx.fill();
    });
  }, [creatures, width, height, colors]);

  const counts = [0, 0, 0];
  creatures.forEach(c => {
    if (counts[c.typeId] !== undefined) {
      counts[c.typeId]++;
    }
  });

  return (
    <div>
      <h2>
        Simulation Area ({creatures.length} total) - 
        <span style={{ color: colors[0], marginLeft: '10px' }}>Sp1: {counts[0]}</span>
        <span style={{ color: colors[1], marginLeft: '10px' }}>Sp2: {counts[1]}</span>
        <span style={{ color: colors[2], marginLeft: '10px' }}>Sp3: {counts[2]}</span>
      </h2>
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
};

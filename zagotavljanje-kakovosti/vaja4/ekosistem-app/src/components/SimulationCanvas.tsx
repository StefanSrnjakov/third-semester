import React, { useRef, useEffect, useState } from 'react';
import { Tile, Creature, Food } from '../types';
import { COLORS } from '../constants';

interface Props {
  grid: Tile[][];
  creatures: Creature[];
  food: Food[];
}

export const SimulationCanvas: React.FC<Props> = ({ grid, creatures, food }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDrag, setIsDrag] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const TILE = 20;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || grid.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(offset.x, offset.y);

    // Draw Terrain (Grid)
    grid.forEach((row, y) => {
      row.forEach((tile, x) => {
        ctx.fillStyle = COLORS[tile.type];
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      });
    });

    // Draw Food (Plants)
    food.forEach(f => {
      if (f.isEaten) return;
      ctx.fillStyle = COLORS.food;
      ctx.beginPath();
      ctx.arc(f.x * TILE + TILE / 2, f.y * TILE + TILE / 2, TILE / 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw Creatures (Prey and Predators)
    creatures.forEach(c => {
      if (c.isDead) return;

      const cx = c.x * TILE + TILE / 2;
      const cy = c.y * TILE + TILE / 2;
      const r = (c.size * TILE) / 3.5;

      ctx.save();

      // Glow/Border for Reproduction Readiness
      if (c.reproductionDesire > 70) {
        ctx.strokeStyle = '#ff385c';
        ctx.lineWidth = 3;
        ctx.beginPath();
        if (c.gender === 'female') ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
        else ctx.rect(cx - r - 2, cy - r - 2, (r + 2) * 2, (r + 2) * 2);
        ctx.stroke();
      }

      ctx.fillStyle = c.type === 'predator' ? COLORS.predator : COLORS.prey;
      ctx.beginPath();
      if (c.gender === 'female') {
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
      } else {
        ctx.rect(cx - r, cy - r, r * 2, r * 2);
      }
      ctx.fill();

      // Action Icon
      const icons: any = {
        searchingMate: '❤️',
        searchingWater: '💧',
        searchingFood: c.type === 'predator' ? '⚔️' : '🥕',
        chasing: '⚔️',
        fleeing: '🏃',
        idle: ''
      };

      const icon = icons[c.currentAction];
      if (icon) {
        ctx.font = '12px serif';
        ctx.textAlign = 'center';
        ctx.fillText(icon, cx, cy - r - 10);
      }

      ctx.strokeStyle = '#2c3e50';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    });

    ctx.restore();
  }, [grid, creatures, food, offset]);

  const canvasWidth = grid[0] ? grid[0].length * TILE : 800;
  const canvasHeight = grid.length ? grid.length * TILE : 800;

  return (
    <div style={{ position: 'relative', overflow: 'hidden', width: `${canvasWidth}px`, height: `${canvasHeight}px`, border: '1px solid #ccc', borderRadius: '8px', cursor: isDrag ? 'grabbing' : 'grab' }}>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={e => { setIsDrag(true); setLastPos({ x: e.clientX, y: e.clientY }); }}
        onMouseMove={e => {
          if (!isDrag) return;
          setOffset(prev => ({ x: prev.x + (e.clientX - lastPos.x), y: prev.y + (e.clientY - lastPos.y) }));
          setLastPos({ x: e.clientX, y: e.clientY });
        }}
        onMouseUp={() => setIsDrag(false)}
        onMouseLeave={() => setIsDrag(false)}
      />

      {/* Legend Overlay */}
      <div style={{ position: 'absolute', top: '10px', left: '10px', background: 'rgba(255,255,255,0.85)', padding: '12px', borderRadius: '10px', pointerEvents: 'none', border: '1px solid #eee', fontSize: '0.8em', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', backdropFilter: 'blur(4px)' }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px', borderBottom: '1px solid #ddd', paddingBottom: '4px' }}>Inhabitant Legend</div>
        <div style={{ display: 'grid', gridTemplateColumns: '20px 1fr', gap: '4px 8px', alignItems: 'center' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#7f8c8d' }}></div> <span>Female (Circle)</span>
          <div style={{ width: '12px', height: '12px', background: '#7f8c8d' }}></div> <span>Male (Square)</span>
          <div style={{ width: '14px', height: '14px', border: '2px solid #ff385c', borderRadius: '50%' }}></div> <span>Ready to Mate</span>
          <hr style={{ gridColumn: 'span 2', width: '100%', margin: '4px 0', border: 'none', borderTop: '1px solid #eee' }} />
          <span>⚔️</span> <span>Hunting / Chase</span>
          <span>🏃</span> <span>Fleeing (Escape)</span>
          <span>🥕</span> <span>Searching Food</span>
          <span>💧</span> <span>Searching Water</span>
          <span>❤️</span> <span>Seeking Partner</span>
        </div>
      </div>
    </div>
  );
};

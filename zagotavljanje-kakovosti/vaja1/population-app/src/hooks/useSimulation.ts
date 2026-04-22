import { useState, useEffect, useRef } from 'react';
import type { SimulationParams, Creature } from '../types';

const COLORS = ['#3b82f6', '#ef4444', '#10b981']; // Blue, Red, Green
const SIMULATION_WIDTH = 600;
const SIMULATION_HEIGHT = 400;

export function useSimulation(paramsArr: SimulationParams[], isRunning: boolean) {
  const [creatures, setCreatures] = useState<Creature[]>([]);
  const [history, setHistory] = useState<{ [key: number]: number[] }>({ 0: [], 1: [], 2: [] });
  const [cycle, setCycle] = useState(0);
  const intervalRef = useRef<number | undefined>(undefined);

  const initSimulation = () => {
    const newCreatures: Creature[] = [];
    const newHistory: { [key: number]: number[] } = { 0: [], 1: [], 2: [] };

    paramsArr.forEach((params, typeId) => {
      newHistory[typeId] = [params.st];
      for (let i = 0; i < params.st; i++) {
        newCreatures.push(createCreature(typeId));
      }
    });

    setCreatures(newCreatures);
    setHistory(newHistory);
    setCycle(0);
  };

  const createCreature = (typeId: number): Creature => ({
    id: Math.random().toString(36).substring(2, 9),
    typeId,
    x: Math.random() * SIMULATION_WIDTH,
    y: Math.random() * SIMULATION_HEIGHT,
    vx: (Math.random() - 0.5) * 4,
    vy: (Math.random() - 0.5) * 4
  });

  const updateSimulation = () => {
    if (!isRunning) return;

    setCreatures(prevCreatures => {
      const currentCounts = [0, 0, 0];
      prevCreatures.forEach(c => currentCounts[c.typeId]++);

      const nextCreatures: Creature[] = [];

      prevCreatures.forEach(c => {
        const typeParams = paramsArr[c.typeId];

        const deathProb = Math.min(1, typeParams.s + (typeParams.k * currentCounts[c.typeId]));

        if (Math.random() < deathProb) {
          return;
        }

        let nx = c.x + c.vx;
        let ny = c.y + c.vy;
        let nvx = c.vx;
        let nvy = c.vy;

        if (nx <= 0 || nx >= SIMULATION_WIDTH) { nvx *= -1; nx = nx <= 0 ? 0 : SIMULATION_WIDTH; }
        if (ny <= 0 || ny >= SIMULATION_HEIGHT) { nvy *= -1; ny = ny <= 0 ? 0 : SIMULATION_HEIGHT; }

        nextCreatures.push({ ...c, x: nx, y: ny, vx: nvx, vy: nvy });

        if (Math.random() < typeParams.r) {
          nextCreatures.push(createCreature(c.typeId));
        }
      });

      // Update history every N frames to avoid huge arrays
      if (cycle % 5 === 0) {
        setHistory(prev => {
          const newH = { ...prev };
          newH[0] = [...newH[0], nextCreatures.filter(c => c.typeId === 0).length];
          newH[1] = [...newH[1], nextCreatures.filter(c => c.typeId === 1).length];
          newH[2] = [...newH[2], nextCreatures.filter(c => c.typeId === 2).length];

          // Limit history length to keep graph rendering fast
          if (newH[0].length > 200) {
            newH[0] = newH[0].slice(1);
            newH[1] = newH[1].slice(1);
            newH[2] = newH[2].slice(1);
          }
          return newH;
        });
      }

      setCycle(s => s + 1);
      return nextCreatures;
    });
  };

  useEffect(() => {
    if (isRunning) {
      // 50ms tick ~= 20 FPS
      intervalRef.current = window.setInterval(updateSimulation, 50);
    }
    return () => {
      if (intervalRef.current !== undefined) window.clearInterval(intervalRef.current);
    };
  }, [isRunning, paramsArr]);

  return { creatures, history, initSimulation, SIMULATION_WIDTH, SIMULATION_HEIGHT, COLORS };
}

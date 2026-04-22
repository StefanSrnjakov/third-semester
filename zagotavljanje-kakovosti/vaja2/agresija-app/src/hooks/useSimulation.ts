import { useState, useEffect, useRef } from 'react';
import type { SimulationParams, Creature, CreatureType, FoodPair } from '../types';

const SIMULATION_WIDTH = 600;
const SIMULATION_HEIGHT = 400;

export function useSimulation(params: SimulationParams, isRunning: boolean) {
  const [, setRenderTrigger] = useState(0);

  const stateRef = useRef({
    creatures: [] as Creature[],
    foodPairs: [] as FoodPair[],
    history: { Peaceful: [], Aggressive: [] } as { Peaceful: number[]; Aggressive: number[] },
    phase: 'spawn' as 'spawn' | 'moveToFood' | 'eatAndResolve' | 'returnToEdge',
    generation: 0
  });

  const intervalRef = useRef<number | undefined>(undefined);

  const createCreature = (type: CreatureType): Creature => {
    let x = 0, y = 0;
    if (Math.random() > 0.5) {
      x = Math.random() > 0.5 ? 0 : SIMULATION_WIDTH;
      y = Math.random() * SIMULATION_HEIGHT;
    } else {
      x = Math.random() * SIMULATION_WIDTH;
      y = Math.random() > 0.5 ? 0 : SIMULATION_HEIGHT;
    }

    return {
      id: Math.random().toString(36).substring(2, 9),
      type, x, y, targetX: x, targetY: y, hasEaten: 0
    };
  };

  const initSimulation = () => {
    const newCreatures: Creature[] = [];

    for (let i = 0; i < params.initialPeaceful; i++) newCreatures.push(createCreature('Peaceful'));
    for (let i = 0; i < params.initialAggressive; i++) newCreatures.push(createCreature('Aggressive'));

    stateRef.current = {
      creatures: newCreatures,
      foodPairs: [],
      history: { Peaceful: [params.initialPeaceful], Aggressive: [params.initialAggressive] },
      phase: 'spawn',
      generation: 0
    };
    setRenderTrigger(v => v + 1);
  };

  const updateSimulation = () => {
    if (!isRunning) return;
    const state = stateRef.current;
    const moveAmount = 5 * params.speed; // Speed acts as movement multiplier

    if (state.phase === 'spawn') {
      const newFood: FoodPair[] = [];
      for (let i = 0; i < params.foodPairs; i++) {
        newFood.push({
          id: `f${i}`,
          x: 50 + Math.random() * (SIMULATION_WIDTH - 100),
          y: 50 + Math.random() * (SIMULATION_HEIGHT - 100),
          creaturesTargeting: 0
        });
      }

      const nextCreatures = [...state.creatures];
      nextCreatures.sort(() => Math.random() - 0.5);

      nextCreatures.forEach(c => {
        c.hasEaten = 0;
        const availableFood = newFood.filter(f => f.creaturesTargeting < 2);
        if (availableFood.length > 0) {
          const targetFood = availableFood[Math.floor(Math.random() * availableFood.length)];
          targetFood.creaturesTargeting++;
          c.targetX = targetFood.x;
          c.targetY = targetFood.y;
        } else {
          c.targetX = c.x;
          c.targetY = c.y;
        }
      });

      state.foodPairs = newFood;
      state.creatures = nextCreatures;
      state.phase = 'moveToFood';
    }
    else if (state.phase === 'moveToFood') {
      let allReached = true;
      state.creatures.forEach(c => {
        const dx = c.targetX - c.x;
        const dy = c.targetY - c.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > moveAmount) {
          allReached = false;
          c.x += (dx / dist) * moveAmount;
          c.y += (dy / dist) * moveAmount;
        } else {
          c.x = c.targetX;
          c.y = c.targetY;
        }
      });

      if (allReached) state.phase = 'eatAndResolve';
    }
    else if (state.phase === 'eatAndResolve') {
      const groups = new Map<string, Creature[]>();

      state.creatures.forEach(c => {
        if (c.targetX === c.x && c.targetY === c.y) {
          if (c.x <= 0 || c.x >= SIMULATION_WIDTH || c.y <= 0 || c.y >= SIMULATION_HEIGHT) return;
        }

        const key = `${Math.round(c.x)},${Math.round(c.y)}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key)!.push(c);
      });

      groups.forEach(group => {
        if (group.length === 1) {
          group[0].hasEaten = 2;
        } else if (group.length === 2) {
          const [c1, c2] = group;
          if (c1.type === 'Peaceful' && c2.type === 'Peaceful') {
            c1.hasEaten = 1; c2.hasEaten = 1;
          } else if (c1.type === 'Aggressive' && c2.type === 'Aggressive') {
            c1.hasEaten = 0; c2.hasEaten = 0;
          } else {
            const agg = c1.type === 'Aggressive' ? c1 : c2;
            const peace = c1.type === 'Peaceful' ? c1 : c2;
            agg.hasEaten = 1.5; peace.hasEaten = 0.5;
          }
        }
      });

      state.creatures.forEach(c => {
        if (Math.random() > 0.5) {
          c.targetX = c.x > SIMULATION_WIDTH / 2 ? SIMULATION_WIDTH : 0;
          c.targetY = Math.random() * SIMULATION_HEIGHT;
        } else {
          c.targetX = Math.random() * SIMULATION_WIDTH;
          c.targetY = c.y > SIMULATION_HEIGHT / 2 ? SIMULATION_HEIGHT : 0;
        }
      });

      state.foodPairs = [];
      state.phase = 'returnToEdge';
    }
    else if (state.phase === 'returnToEdge') {
      let allReached = true;
      state.creatures.forEach(c => {
        const dx = c.targetX - c.x;
        const dy = c.targetY - c.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > moveAmount) {
          allReached = false;
          c.x += (dx / dist) * moveAmount;
          c.y += (dy / dist) * moveAmount;
        } else {
          c.x = c.targetX;
          c.y = c.targetY;
        }
      });

      if (allReached) {
        const survivors: Creature[] = [];
        state.creatures.forEach(c => {
          if (c.hasEaten === 0) return;
          if (c.hasEaten === 1) survivors.push({ ...c, hasEaten: 0 });
          else if (c.hasEaten === 2) {
            survivors.push({ ...c, hasEaten: 0 });
            survivors.push(createCreature(c.type));
          } else if (c.hasEaten === 1.5) {
            survivors.push({ ...c, hasEaten: 0 });
            if (Math.random() < 0.5) survivors.push(createCreature(c.type));
          } else if (c.hasEaten === 0.5) {
            if (Math.random() < 0.5) survivors.push({ ...c, hasEaten: 0 });
          }
        });

        const pCount = survivors.filter(c => c.type === 'Peaceful').length;
        const aCount = survivors.filter(c => c.type === 'Aggressive').length;

        const p = [...state.history.Peaceful, pCount];
        const a = [...state.history.Aggressive, aCount];
        if (p.length > 200) { p.shift(); a.shift(); }
        state.history = { Peaceful: p, Aggressive: a };

        state.creatures = survivors;
        state.generation++;
        state.phase = 'spawn';
      }
    }

    setRenderTrigger(v => v + 1);
  };

  useEffect(() => {
    if (isRunning) {
      // Lower interval delay as speed increases for an additive effect
      const intervalDelay = Math.max(10, 50 - params.speed);
      intervalRef.current = window.setInterval(updateSimulation, intervalDelay);
    }
    return () => clearInterval(intervalRef.current);
  }, [isRunning, params]);

  return {
    creatures: [...stateRef.current.creatures],
    foodPairs: [...stateRef.current.foodPairs],
    history: stateRef.current.history,
    initSimulation,
    SIMULATION_WIDTH,
    SIMULATION_HEIGHT
  };
}

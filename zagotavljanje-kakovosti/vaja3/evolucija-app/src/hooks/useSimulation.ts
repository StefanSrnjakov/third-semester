import { useState, useEffect, useRef } from 'react';
import { Creature, Food, SimulationParams, HistoryPoint } from '../types';

// Constants for simulation tuning
const SIM_CONSTANTS = {
  ID_LENGTH: 9,                    // Length of the generated random IDs
  FOOD_SPAWN_MARGIN: 10,           // Margin from edges when spawning food
  MUTATION_CHANCE: 0.1,            // 10% probability of a trait mutating
  MUTATION_RANGE: 0.2,             // Mutation spread (e.g., 0.2 allows +/- 10% change)
  MUTATION_OFFSET: 0.1,            // Offset to center the mutation range at 0
  MIN_TRAIT_VALUE: 0.1,            // Minimum possible value for any trait
  ENERGY_COST_MULTIPLIER: 0.05,    // Global energy consumption scale factor
  RETURN_HOME_ENERGY_THRESHOLD: 0.4, // Energy level (40%) at which a creature seeks home
  FOOD_DETECTION_RANGE_FACTOR: 20, // Multiplier for perception to define food detection range
  HUNTING_RANGE_FACTOR: 15,        // Multiplier for perception to define hunting range
  FLEEING_RANGE_FACTOR: 10,        // Multiplier for perception to define fleeing range
  SIZE_ADVANTAGE_THRESHOLD: 1.2,   // Required size ratio to hunt or scare another creature
  IDLE_WANDER_OFFSET: 40,          // Max target offset for random movement when no targets are nearby
  SPEED_MULTIPLIER: 4,             // Conversion factor from speed trait to pixels per tick
  COLLISION_DISTANCE: 10,          // Distance for eating or killing events
  SAFE_ZONE_HEIGHT: 10,            // Y-coordinate threshold for the safe starting zone
  MIN_FOOD_COUNT: 10,              // Floor for food count in decreasing scenarios
  BASE_TICK_INTERVAL: 33,          // Base simulation step duration in milliseconds (~30 FPS)
};

export function useSimulation(params: SimulationParams, isRunning: boolean) {
  const [, setTrigger] = useState(0);
  const stateRef = useRef({
    creatures: [] as Creature[],
    foods: [] as Food[],
    history: [] as HistoryPoint[],
    generation: 0,
    currentFoodCount: params.initialFood,
  });

  const generateId = () => Math.random().toString(36).substr(2, SIM_CONSTANTS.ID_LENGTH);

  const createCreature = (x: number, y: number, traits: Partial<Creature>): Creature => ({
    id: generateId(),
    x, y,
    energy: params.initialEnergy,
    speed: traits.speed || params.initialSpeed,
    size: traits.size || params.initialSize,
    perception: traits.perception || params.initialPerception,
    foodEaten: 0,
    isReturning: false,
    isDead: false,
  });

  const init = () => {
    const creatures: Creature[] = [];
    for (let i = 0; i < params.initialCount; i++) {
      // Start at the edge (y = 0)
      creatures.push(createCreature(Math.random() * params.width, 0, {}));
    }
    stateRef.current = {
      creatures,
      foods: [],
      history: [],
      generation: 0,
      currentFoodCount: params.initialFood,
    };
    spawnFood();
    setTrigger(t => t + 1);
  };

  const spawnFood = () => {
    const state = stateRef.current;
    const newFoods: Food[] = [];
    for (let i = 0; i < state.currentFoodCount; i++) {
      newFoods.push({
        id: generateId(),
        x: Math.random() * params.width,
        y: Math.random() * (params.height - 2 * SIM_CONSTANTS.FOOD_SPAWN_MARGIN) + SIM_CONSTANTS.FOOD_SPAWN_MARGIN,
      });
    }
    state.foods = newFoods;
  };

  const mutate = (val: number) => {
    if (Math.random() < SIM_CONSTANTS.MUTATION_CHANCE) {
      const change = (Math.random() * SIM_CONSTANTS.MUTATION_RANGE - SIM_CONSTANTS.MUTATION_OFFSET) * val;
      return Math.max(SIM_CONSTANTS.MIN_TRAIT_VALUE, val + change);
    }
    return val;
  };

  const tick = () => {
    if (!isRunning) return;
    const state = stateRef.current;

    state.creatures.forEach(c => {
      if (c.isDead || (c.isReturning && c.y <= SIM_CONSTANTS.SAFE_ZONE_HEIGHT)) return;

      const cost = (Math.pow(c.size, 3) * Math.pow(c.speed, 2) + c.perception) * SIM_CONSTANTS.ENERGY_COST_MULTIPLIER;
      c.energy -= cost;
      if (c.energy <= 0) {
        c.isDead = true;
        return;
      }

      const shouldReturnHome = c.foodEaten >= 2 || (c.foodEaten >= 1 && c.energy < params.initialEnergy * SIM_CONSTANTS.RETURN_HOME_ENERGY_THRESHOLD);
      if (shouldReturnHome) {
        c.isReturning = true;
      }

      let targetX = c.x;
      let targetY = c.y;
      let foundTarget = false;

      if (c.isReturning) {
        targetX = c.x;
        targetY = 0;
        foundTarget = true;
      } else {
        let closestFood: Food | null = null;
        let minDist = c.perception * SIM_CONSTANTS.FOOD_DETECTION_RANGE_FACTOR;
        for (const f of state.foods) {
          const d = Math.sqrt(Math.pow(f.x - c.x, 2) + Math.pow(f.y - c.y, 2));
          if (d < minDist) {
            minDist = d;
            closestFood = f;
          }
        }

        if (closestFood) {
          targetX = closestFood.x;
          targetY = closestFood.y;
          foundTarget = true;
        }

        if (!foundTarget) {
          for (const other of state.creatures) {
            if (other === c || other.isDead || other.isReturning) continue;
            const d = Math.sqrt(Math.pow(other.x - c.x, 2) + Math.pow(other.y - c.y, 2));
            const canHunt = d < c.perception * SIM_CONSTANTS.HUNTING_RANGE_FACTOR && c.size > SIM_CONSTANTS.SIZE_ADVANTAGE_THRESHOLD * other.size;
            if (canHunt) {
              targetX = other.x;
              targetY = other.y;
              foundTarget = true;
              break;
            }
          }
        }
      }

      for (const other of state.creatures) {
        if (other === c || other.isDead) continue;
        const d = Math.sqrt(Math.pow(other.x - c.x, 2) + Math.pow(other.y - c.y, 2));
        const shouldFlee = d < c.perception * SIM_CONSTANTS.FLEEING_RANGE_FACTOR && other.size > SIM_CONSTANTS.SIZE_ADVANTAGE_THRESHOLD * c.size;
        if (shouldFlee) {
          targetX = c.x - (other.x - c.x);
          targetY = c.y - (other.y - c.y);
          foundTarget = true;
          break;
        }
      }

      if (!foundTarget) {
        targetX += (Math.random() - 0.5) * SIM_CONSTANTS.IDLE_WANDER_OFFSET;
        targetY += (Math.random() - 0.5) * SIM_CONSTANTS.IDLE_WANDER_OFFSET;
      }

      const dx = targetX - c.x;
      const dy = targetY - c.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const move = Math.min(dist, c.speed * SIM_CONSTANTS.SPEED_MULTIPLIER);
      c.x += (dx / dist) * move;
      c.y += (dy / dist) * move;

      c.x = Math.max(0, Math.min(params.width, c.x));
      c.y = Math.max(0, Math.min(params.height, c.y));

      state.foods = state.foods.filter(f => {
        const d = Math.sqrt(Math.pow(f.x - c.x, 2) + Math.pow(f.y - c.y, 2));
        if (d < SIM_CONSTANTS.COLLISION_DISTANCE) {
          c.foodEaten++;
          return false;
        }
        return true;
      });

      state.creatures.forEach(other => {
        if (other === c || other.isDead || other.isReturning) return;
        const d = Math.sqrt(Math.pow(other.x - c.x, 2) + Math.pow(other.y - c.y, 2));
        if (d < SIM_CONSTANTS.COLLISION_DISTANCE && c.size > SIM_CONSTANTS.SIZE_ADVANTAGE_THRESHOLD * other.size) {
          other.isDead = true;
          c.foodEaten++;
        }
      });
    });

    const allDone = state.creatures.every(c => c.isDead || (c.isReturning && c.y <= SIM_CONSTANTS.SAFE_ZONE_HEIGHT));
    if (allDone && state.creatures.length > 0) {
      nextGeneration();
    }
  };

  const nextGeneration = () => {
    const state = stateRef.current;
    const survivors = state.creatures.filter(c => !c.isDead && c.isReturning && c.y <= SIM_CONSTANTS.SAFE_ZONE_HEIGHT && c.foodEaten >= 1);

    if (survivors.length > 0 || state.generation > 0) {
      state.history.push({
        generation: state.generation,
        count: survivors.length,
        avgSpeed: survivors.length ? survivors.reduce((a, b) => a + b.speed, 0) / survivors.length : 0,
        avgSize: survivors.length ? survivors.reduce((a, b) => a + b.size, 0) / survivors.length : 0,
        avgPerception: survivors.length ? survivors.reduce((a, b) => a + b.perception, 0) / survivors.length : 0,
      });
    }

    const nextCreatures: Creature[] = [];
    survivors.forEach(c => {
      nextCreatures.push(createCreature(Math.random() * params.width, 0, { speed: c.speed, size: c.size, perception: c.perception }));
      if (c.foodEaten >= 2) {
        nextCreatures.push(createCreature(Math.random() * params.width, 0, {
          speed: mutate(c.speed),
          size: mutate(c.size),
          perception: mutate(c.perception),
        }));
      }
    });

    state.creatures = nextCreatures;
    state.generation++;

    if (params.foodScenario === 'decreasing') {
      state.currentFoodCount = Math.max(SIM_CONSTANTS.MIN_FOOD_COUNT, state.currentFoodCount - 1);
    }

    spawnFood();
  };

  useEffect(() => {
    if (!isRunning) return;

    const timer = setInterval(() => {
      const ticksPerRender = Math.max(1, Math.floor(params.simulationSpeed / 2));

      for (let i = 0; i < ticksPerRender; i++) {
        tick();
      }
      setTrigger(t => t + 1);
    }, SIM_CONSTANTS.BASE_TICK_INTERVAL);

    return () => clearInterval(timer);
  }, [isRunning, params.simulationSpeed, params.width, params.height, params.foodScenario]);

  return { state: stateRef.current, init };
}

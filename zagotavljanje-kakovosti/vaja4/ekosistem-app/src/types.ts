export type TileType = 'water' | 'land';
export type Gender = 'male' | 'female';
export type CreatureType = 'predator' | 'prey';

export interface Tile {
  x: number;
  y: number;
  type: TileType;
}

export interface Creature {
  id: string;
  type: CreatureType;
  x: number;
  y: number;
  hunger: number; // 0 to 100, 100 is dead
  thirst: number; // 0 to 100, 100 is dead
  reproductionDesire: number; // 0 to 100
  age: number;
  maxAge: number;
  speed: number;
  size: number;
  perception: number;
  gender: Gender;
  isDead: boolean;
  currentAction: 'idle' | 'chasing' | 'fleeing' | 'searchingFood' | 'searchingWater' | 'searchingMate';
  wanderTarget?: { x: number; y: number }; // For smooth wandering
}

export interface Food {
  id: string;
  x: number;
  y: number;
  isEaten: boolean;
}

export interface DeathStats {
  age: number;
  thirst: number;
  starvation: number;
  eaten: number;
}

export interface TraitStats {
  avgSpeed: number;
  avgSize: number;
  avgPerception: number;
  population: number;
  births: number;
}

export interface HistorySnapshot {
  day: number;
  prey: DeathStats;
  predator: DeathStats;
  preyTraits: TraitStats;
  predatorTraits: TraitStats;
  foodCount: number;
}

export interface SimulationParams {
  width: number;
  height: number;
  initialPrey: number;
  initialPredators: number;
  initialFood: number;
  selectedTerrain: 'lake' | 'river' | 'multi-lake' | 'puddles';
  simulationSpeed: number;
  hungerTickBase: number;
  thirstTickBase: number;
  reproductionInc: number;
  perceptionMult: number;
  agingMult: number;
}

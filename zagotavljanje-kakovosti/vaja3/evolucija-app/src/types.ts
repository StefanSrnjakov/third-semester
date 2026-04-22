export type SimulationParams = {
  initialCount: number;
  initialFood: number;
  initialEnergy: number;
  initialSpeed: number;
  initialSize: number;
  initialPerception: number;
  foodScenario: 'constant' | 'decreasing';
  width: number;
  height: number;
  simulationSpeed: number;
};

export type Creature = {
  id: string;
  x: number;
  y: number;
  energy: number;
  speed: number;
  size: number;
  perception: number;
  foodEaten: number;
  isReturning: boolean;
  isDead: boolean;
};

export type Food = {
  id: string;
  x: number;
  y: number;
};

export type HistoryPoint = {
  generation: number;
  count: number;
  avgSpeed: number;
  avgSize: number;
  avgPerception: number;
};

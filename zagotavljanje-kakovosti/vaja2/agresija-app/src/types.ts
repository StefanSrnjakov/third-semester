export type SimulationParams = {
  initialPeaceful: number;
  initialAggressive: number;
  foodPairs: number;
  speed: number;
};

export type CreatureType = 'Peaceful' | 'Aggressive';

export type Creature = {
  id: string;
  type: CreatureType;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  hasEaten: number; // 0, 0.5, 1, or 1.5
};

export type FoodPair = {
  id: string;
  x: number;
  y: number;
  creaturesTargeting: number;
};

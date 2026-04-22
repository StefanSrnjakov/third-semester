export type SimulationParams = {
  st: number;
  r: number;
  s: number;
  k: number;
};

export type Creature = {
  id: string;
  typeId: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
};

import { Creature, Food, Tile, SimulationParams } from '../types';
import { SIM_CONSTANTS } from '../constants';

export const getDistance = (x1: number, y1: number, x2: number, y2: number) =>
    Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));

export const getVariant = (val: number) => {
    return val * (1 + (Math.random() * SIM_CONSTANTS.VARIATION * 2 - SIM_CONSTANTS.VARIATION));
};

export const createCreature = (type: 'predator' | 'prey', x: number, y: number, traits: Partial<Creature> = {}): Creature => ({
    id: Math.random().toString(36).substring(2, 2 + SIM_CONSTANTS.ID_COUNT),
    type, x, y,
    hunger: 0,
    thirst: 0,
    reproductionDesire: 0,
    age: 0,
    maxAge: traits.maxAge || getVariant(SIM_CONSTANTS.MAX_AGE_BASE),
    speed: traits.speed || getVariant(1),
    size: traits.size || getVariant(1),
    perception: traits.perception || getVariant(5),
    gender: traits.gender || (Math.random() > 0.5 ? 'male' : 'female'),
    isDead: false,
    currentAction: 'idle',
});

export const spawnCreature = (type: 'predator' | 'prey', params: SimulationParams, grid: Tile[][]) => {
    let x = 0, y = 0, tries = 0;
    do {
        x = Math.floor(Math.random() * params.width);
        y = Math.floor(Math.random() * params.height);
        tries++;
    } while (grid[y] && grid[y][x] && grid[y][x].type === 'water' && tries < 200);
    return createCreature(type, x, y);
};

export const spawnFood = (params: SimulationParams, grid: Tile[][]): Food => {
    let x = 0, y = 0, tries = 0;
    do {
        x = Math.floor(Math.random() * params.width);
        y = Math.floor(Math.random() * params.height);
        tries++;
    } while (grid[y] && grid[y][x] && grid[y][x].type === 'water' && tries < 200);
    return { id: Math.random().toString(36).substring(2, 8), x, y, isEaten: false };
};

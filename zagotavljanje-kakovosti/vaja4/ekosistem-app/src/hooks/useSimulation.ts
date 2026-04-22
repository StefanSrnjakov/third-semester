import { useState, useEffect, useRef } from 'react';
import { Creature, SimulationParams, Food, Tile, HistorySnapshot } from '../types';
import { SIM_CONSTANTS } from '../constants';
import { generateTerrain } from '../utils/terrain';
import { spawnCreature, spawnFood } from '../utils/simulationUtils';
import { updateCreature } from '../utils/creatureAI';

export function useSimulation(params: SimulationParams, isRunning: boolean) {
    const [, setTrigger] = useState(0);
    const stateRef = useRef({
        creatures: [] as Creature[],
        food: [] as Food[],
        grid: [] as Tile[][],
        ticks: 0,
        deathStatsPrey: { age: 0, thirst: 0, starvation: 0, eaten: 0 },
        deathStatsPredator: { age: 0, thirst: 0, starvation: 0, eaten: 0 },
        dailyPreyBirths: 0,
        dailyPredatorBirths: 0,
        deathHistory: [] as HistorySnapshot[]
    });

    const init = () => {
        const grid = generateTerrain(params);
        const creatures: Creature[] = [];
        const food: Food[] = [];

        for (let i = 0; i < params.initialPrey; i++) creatures.push(spawnCreature('prey', params, grid));
        for (let i = 0; i < params.initialPredators; i++) creatures.push(spawnCreature('predator', params, grid));

        const fCount = params.initialFood || 50;
        for (let i = 0; i < fCount; i++) food.push(spawnFood(params, grid));

        stateRef.current = {
            creatures, food, grid, ticks: 0,
            deathStatsPrey: { age: 0, thirst: 0, starvation: 0, eaten: 0 },
            deathStatsPredator: { age: 0, thirst: 0, starvation: 0, eaten: 0 },
            dailyPreyBirths: 0,
            dailyPredatorBirths: 0,
            deathHistory: []
        };
        setTrigger(t => t + 1);
    };

    const tick = () => {
        if (!isRunning) return;
        const state = stateRef.current;
        if (state.creatures.length === 0 && state.ticks > 0) return;

        // Cleanup dead and eaten
        state.creatures = state.creatures.filter(c => !c.isDead);
        state.food = state.food.filter(f => !f.isEaten);

        // Daily Events
        if (state.ticks % SIM_CONSTANTS.TICKS_PER_DAY === 0 && state.ticks > 0) {
            // Relocate 100% of food every day (forces movement)
            state.food = state.food.map(() => spawnFood(params, state.grid));

            // Calculate Trait Statistics
            const calculateTraits = (type: 'prey' | 'predator') => {
                const group = state.creatures.filter(c => c.type === type);
                if (group.length === 0) return { avgSpeed: 0, avgSize: 0, avgPerception: 0, population: 0, births: type === 'prey' ? state.dailyPreyBirths : state.dailyPredatorBirths };
                return {
                    avgSpeed: group.reduce((acc, c) => acc + c.speed, 0) / group.length,
                    avgSize: group.reduce((acc, c) => acc + c.size, 0) / group.length,
                    avgPerception: group.reduce((acc, c) => acc + c.perception, 0) / group.length,
                    population: group.length,
                    births: type === 'prey' ? state.dailyPreyBirths : state.dailyPredatorBirths
                };
            };

            // Record History Snapshot
            state.deathHistory.push({
                day: Math.floor(state.ticks / SIM_CONSTANTS.TICKS_PER_DAY),
                prey: { ...state.deathStatsPrey },
                predator: { ...state.deathStatsPredator },
                preyTraits: calculateTraits('prey'),
                predatorTraits: calculateTraits('predator'),
                foodCount: state.food.length
            });

            // Reset Daily Counters
            state.dailyPreyBirths = 0;
            state.dailyPredatorBirths = 0;
        }

        // Seasonal Food Replenishment
        if (state.food.length < (params.initialFood || 50) && Math.random() < 0.2) {
            state.food.push(spawnFood(params, state.grid));
        }

        // Update each creature
        // Pass a wrapper to track births
        const extendedState = {
            ...state,
            addBirth: (type: 'prey' | 'predator') => {
                if (type === 'prey') state.dailyPreyBirths++;
                else state.dailyPredatorBirths++;
            }
        };
        state.creatures.forEach(c => updateCreature(c, params, extendedState));

        state.ticks++;
    };

    useEffect(() => {
        if (!isRunning) return;
        const loop = setInterval(() => {
            const batchSize = Math.max(1, Math.floor(params.simulationSpeed / 10));
            for (let i = 0; i < batchSize; i++) tick();
            setTrigger(t => t + 1);
        }, SIM_CONSTANTS.TICK_MS);

        return () => clearInterval(loop);
    }, [isRunning, params]);

    return { state: stateRef.current, init };
}

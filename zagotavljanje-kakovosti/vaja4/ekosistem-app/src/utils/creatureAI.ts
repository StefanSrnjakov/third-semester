import { Creature, Food, Tile, SimulationParams } from '../types';
import { SIM_CONSTANTS } from '../constants';
import { getDistance, createCreature, getVariant } from './simulationUtils';

export const updateCreature = (
    c: Creature,
    params: SimulationParams,
    state: { creatures: Creature[], food: Food[], grid: Tile[][], deathStatsPrey: any, deathStatsPredator: any, addBirth: (type: 'prey' | 'predator') => void }
) => {
    const isWater = (x: number, y: number) => {
        const gX = Math.floor(x);
        const gY = Math.floor(y);
        if (gX < 0 || gX >= params.width || gY < 0 || gY >= params.height) return true;
        return state.grid[gY][gX].type === 'water';
    };

    // 1. Survival Cycle
    c.age += (1 / SIM_CONSTANTS.TICKS_PER_DAY) * params.agingMult;

    let hungerInc = params.hungerTickBase * (1 + (c.size - 1) * SIM_CONSTANTS.SIZE_TO_HUNGER);
    // Metabolism slows down: creatures get 50% hungrier as they approach max age
    hungerInc *= (1 + (c.age / c.maxAge) * 0.5);
    let thirstInc = params.thirstTickBase * (1 + (c.speed - 1) * SIM_CONSTANTS.SPEED_TO_THIRST);

    if (c.hunger > SIM_CONSTANTS.NEED_PRIORITY.HUNGER) {
        thirstInc *= SIM_CONSTANTS.HUNGER_THIRST_PENALTY;
    }

    c.hunger += hungerInc;
    c.thirst += thirstInc;
    c.reproductionDesire += params.reproductionInc;

    const stats = c.type === 'prey' ? state.deathStatsPrey : state.deathStatsPredator;
    if (c.age > c.maxAge) { c.isDead = true; stats.age++; return; }
    if (c.thirst >= SIM_CONSTANTS.THIRST_CAP) { c.isDead = true; stats.thirst++; return; }
    if (c.hunger >= SIM_CONSTANTS.HUNGER_CAP) { c.isDead = true; stats.starvation++; return; }

    // 2. AI Decision Making
    if (!c.wanderTarget || getDistance(c.x, c.y, c.wanderTarget.x, c.wanderTarget.y) < SIM_CONSTANTS.WANDER_ARRIVE_DIST) {
        let wx = 0, wy = 0, tries = 0;
        do {
            const dx = (Math.random() - 0.5);
            const dy = (Math.random() - 0.5);
            const mag = Math.sqrt(dx * dx + dy * dy) || 1;
            wx = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.width - SIM_CONSTANTS.MARGIN, c.x + (dx / mag) * SIM_CONSTANTS.WANDER_DIST));
            wy = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.height - SIM_CONSTANTS.MARGIN, c.y + (dy / mag) * SIM_CONSTANTS.WANDER_DIST));
            tries++;
        } while (isWater(wx, wy) && tries < 20);

        c.wanderTarget = { x: wx, y: wy };
    }

    let targetX = c.wanderTarget.x;
    let targetY = c.wanderTarget.y;
    let hasTarget = false;

    const viewDist = c.perception * params.perceptionMult;

    c.currentAction = 'idle';

    // Reproduction (P1)
    if (!hasTarget && c.reproductionDesire > SIM_CONSTANTS.NEED_PRIORITY.REPRODUCTION) {
        const mates = state.creatures.filter(m =>
            m.type === c.type && m.gender !== c.gender && !m.isDead &&
            m.reproductionDesire > SIM_CONSTANTS.NEED_PRIORITY.REPRODUCTION &&
            getDistance(c.x, c.y, m.x, m.y) <= viewDist
        );
        if (mates.length > 0) {
            c.currentAction = 'searchingMate';
            const closestMate = mates.reduce((prev, curr) => getDistance(c.x, c.y, curr.x, curr.y) < getDistance(c.x, c.y, prev.x, prev.y) ? curr : prev);
            targetX = closestMate.x; targetY = closestMate.y; hasTarget = true;

            if (getDistance(c.x, c.y, closestMate.x, closestMate.y) <= SIM_CONSTANTS.BREED_RANGE) {
                state.creatures.push(createCreature(c.type, c.x, c.y, {
                    speed: getVariant((c.speed + closestMate.speed) / 2),
                    size: getVariant((c.size + closestMate.size) / 2),
                    perception: getVariant((c.perception + closestMate.perception) / 2),
                }));
                state.addBirth(c.type);
                c.reproductionDesire = 0; closestMate.reproductionDesire = 0;
            }
        }
    }

    // Thirst (P2)
    if (!hasTarget && c.thirst > SIM_CONSTANTS.NEED_PRIORITY.THIRST) {
        const waterSources: { x: number, y: number, d: number }[] = [];
        const startX = Math.max(0, Math.floor(c.x - viewDist));
        const endX = Math.min(params.width - 1, Math.ceil(c.x + viewDist));
        const startY = Math.max(0, Math.floor(c.y - viewDist));
        const endY = Math.min(params.height - 1, Math.ceil(c.y + viewDist));

        for (let y = startY; y <= endY; y++) {
            for (let x = startX; x <= endX; x++) {
                if (state.grid[y][x].type === 'water') {
                    const d = getDistance(c.x, c.y, x, y);
                    if (d <= viewDist) waterSources.push({ x, y, d });
                }
            }
        }
        if (waterSources.length > 0) {
            c.currentAction = 'searchingWater';
            waterSources.sort((a, b) => a.d - b.d);
            const chosen = waterSources[Math.floor(Math.random() * Math.min(SIM_CONSTANTS.SEARCH_CANDIDATES, waterSources.length))];
            targetX = chosen.x; targetY = chosen.y; hasTarget = true;
            if (getDistance(c.x, c.y, targetX, targetY) <= SIM_CONSTANTS.WATER_DRINK_RANGE) c.thirst = 0;
        }
    }

    // Hunger (P3)
    if (!hasTarget && c.hunger > SIM_CONSTANTS.NEED_PRIORITY.HUNGER) {
        if (c.type === 'predator') {
            const targets = state.creatures.filter(p =>
                p.type === 'prey' && !p.isDead &&
                getDistance(c.x, c.y, p.x, p.y) <= viewDist
            );
            if (targets.length > 0) {
                c.currentAction = 'chasing';
                const closest = targets.reduce((prev, curr) => getDistance(c.x, c.y, curr.x, curr.y) < getDistance(c.x, c.y, prev.x, prev.y) ? curr : prev);
                targetX = closest.x; targetY = closest.y; hasTarget = true;
                if (getDistance(c.x, c.y, closest.x, closest.y) <= SIM_CONSTANTS.EAT_RANGE) {
                    if (!closest.isDead) { closest.isDead = true; state.deathStatsPrey.eaten++; }
                    c.hunger = 0;
                }
            }
        } else {
            const foodList = state.food.filter(f =>
                !f.isEaten &&
                getDistance(c.x, c.y, f.x, f.y) <= viewDist
            );
            if (foodList.length > 0) {
                c.currentAction = 'searchingFood';
                foodList.sort((a, b) => getDistance(c.x, c.y, a.x, a.y) - getDistance(c.x, c.y, b.x, b.y));
                const chosenFood = foodList[Math.floor(Math.random() * Math.min(SIM_CONSTANTS.SEARCH_CANDIDATES, foodList.length))];
                targetX = chosenFood.x; targetY = chosenFood.y; hasTarget = true;
                if (getDistance(c.x, c.y, chosenFood.x, chosenFood.y) <= SIM_CONSTANTS.EAT_RANGE) {
                    chosenFood.isEaten = true; c.hunger = 0;
                }
            }
        }
    }

    // Fleeing (P4)
    if (!hasTarget && c.type === 'prey') {
        const predatorsInSight = state.creatures.filter(p =>
            p.type === 'predator' && !p.isDead &&
            getDistance(c.x, c.y, p.x, p.y) < (viewDist * SIM_CONSTANTS.FLEE_PERCEPTION_MULT)
        );

        if (predatorsInSight.length > 0) {
            c.currentAction = 'fleeing';
            let escapeX = 0;
            let escapeY = 0;

            // 1. Repulsion from predators (Closer ones push significantly harder)
            predatorsInSight.forEach(p => {
                const dx = c.x - p.x;
                const dy = c.y - p.y;
                const distSq = dx * dx + dy * dy || 0.1;
                escapeX += dx / distSq;
                escapeY += dy / distSq;
            });

            // 2. Repulsion from walls (Forces prey away from corners and edges)
            const margin = SIM_CONSTANTS.EDGE_DIST;
            const power = SIM_CONSTANTS.EDGE_REPULSION;

            if (c.x < margin) escapeX += power / (c.x * c.x || 0.1);
            if (c.x > params.width - margin) escapeX -= power / ((params.width - c.x) * (params.width - c.x) || 0.1);
            if (c.y < margin) escapeY += power / (c.y * c.y || 0.1);
            if (c.y > params.height - margin) escapeY -= power / ((params.height - c.y) * (params.height - c.y) || 0.1);

            // 3. Repulsion from nearby water (Prevents getting pinned against a lake)
            const waterMargin = 3;
            const wXMin = Math.max(0, Math.floor(c.x - waterMargin));
            const wXMax = Math.min(params.width - 1, Math.ceil(c.x + waterMargin));
            const wYMin = Math.max(0, Math.floor(c.y - waterMargin));
            const wYMax = Math.min(params.height - 1, Math.ceil(c.y + waterMargin));

            for (let wy = wYMin; wy <= wYMax; wy++) {
                for (let wx = wXMin; wx <= wXMax; wx++) {
                    if (state.grid[wy][wx].type === 'water') {
                        const dx = c.x - (wx + 0.5);
                        const dy = c.y - (wy + 0.5);
                        const d2 = dx * dx + dy * dy || 0.1;
                        if (d2 < waterMargin * waterMargin) {
                            escapeX += (power * 0.5) / d2;
                            escapeY += (power * 0.5) / d2;
                        }
                    }
                }
            }

            // 4. Break symmetry (Avoid getting stuck in a perfect center-sandwich)
            if (Math.abs(escapeX) < 0.001 && Math.abs(escapeY) < 0.001) {
                escapeX = (Math.random() - 0.5);
                escapeY = (Math.random() - 0.5);
            }

            const mag = Math.sqrt(escapeX * escapeX + escapeY * escapeY) || 1;

            // Project a target far away in the best possible escape direction
            c.wanderTarget = {
                x: c.x + (escapeX / mag) * SIM_CONSTANTS.WANDER_DIST,
                y: c.y + (escapeY / mag) * SIM_CONSTANTS.WANDER_DIST
            };
            targetX = c.wanderTarget.x;
            targetY = c.wanderTarget.y;
            hasTarget = true;
        }
    }

    // 3. Separation (Prevention of stacking)
    let sepX = 0, sepY = 0;
    const SEPARATION_DIST = 1.3;
    for (const other of state.creatures) {
        if (other === c || other.isDead || other.type !== c.type) continue;
        const d = getDistance(c.x, c.y, other.x, other.y);
        if (d < SEPARATION_DIST) {
            if (d < 0.1) {
                sepX += (Math.random() - 0.5) * 5;
                sepY += (Math.random() - 0.5) * 5;
            } else {
                sepX += (c.x - other.x) / d;
                sepY += (c.y - other.y) / d;
            }
        }
    }

    // 4. Movement Execution
    let dx = (targetX - c.x) + sepX * 0.5;
    let dy = (targetY - c.y) + sepY * 0.5;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    let nextX = c.x + (dx / dist) * Math.min(dist, c.speed / SIM_CONSTANTS.SPEED_DIVISOR);
    let nextY = c.y + (dy / dist) * Math.min(dist, c.speed / SIM_CONSTANTS.SPEED_DIVISOR);

    nextX = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.width - SIM_CONSTANTS.MARGIN, nextX));
    nextY = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.height - SIM_CONSTANTS.MARGIN, nextY));

    if (isWater(nextX, nextY)) {
        if (!hasTarget) c.wanderTarget = undefined;
        if (!isWater(nextX, c.y)) nextY = c.y;
        else if (!isWater(c.x, nextY)) nextX = c.x;
        else { nextX = c.x; nextY = c.y; }
    }

    // Anti-Stuck (Enhanced)
    const actualStep = getDistance(c.x, c.y, nextX, nextY);
    if (dist > SIM_CONSTANTS.STUCK_DIST_THRESHOLD && actualStep < SIM_CONSTANTS.STUCK_ACTUAL_THRESHOLD) {
        c.wanderTarget = undefined; // Force choose new dir next tick
        // Small random nudge towards center as fallback
        const nudgeX = (params.width / 2 - c.x) * 0.1;
        const nudgeY = (params.height / 2 - c.y) * 0.1;
        nextX = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.width - SIM_CONSTANTS.MARGIN, c.x + nudgeX + (Math.random() - 0.5) * 2));
        nextY = Math.max(SIM_CONSTANTS.MARGIN, Math.min(params.height - SIM_CONSTANTS.MARGIN, c.y + nudgeY + (Math.random() - 0.5) * 2));
        if (isWater(nextX, nextY)) { nextX = c.x; nextY = c.y; }
    }

    c.x = nextX;
    c.y = nextY;
};

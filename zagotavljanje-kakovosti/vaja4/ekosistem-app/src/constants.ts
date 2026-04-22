export const SIM_CONSTANTS = {
  // Survival parameters (Age in days, Hunger, Thirst)
  TICKS_PER_DAY: 50,            // 50 simulation ticks equal 1 day
  MAX_AGE_BASE: 30,             // Base lifespan in days before variation
  HUNGER_CAP: 100,              // Death from starvation when hunger hits 100
  THIRST_CAP: 100,              // Death from dehydration when thirst hits 100

  // Rate modifiers
  HUNGER_TICK_BASE: 0.25,        // Base hunger increase per tick
  THIRST_TICK_BASE: 0.3,        // Base thirst increase per tick
  HUNGER_THIRST_PENALTY: 1.5,   // Hunger increases thirst penalty by 1.5x when food depleted

  // Trait impact on consumption (Higher speed = more thirst, Larger size = more hunger)
  SPEED_TO_THIRST: 0.05,
  SIZE_TO_HUNGER: 0.1,

  // Ranges and Interactions
  PERCEPTION_MULT: 3,           // Perception multiplier (e.g. 5 perception * 2 = 10 tiles visibility)
  WATER_DRINK_RANGE: 1.5,       // Max distance to drink water
  BREED_RANGE: 1.5,             // Max breeding distance
  EAT_RANGE: 1.5,               // Max attack/eating distance

  // Need Priorities - Reproduction > Thirst > Hunger
  NEED_PRIORITY: {
    REPRODUCTION: 70,
    THIRST: 50,
    HUNGER: 60,
  },

  // Variation during inheritance
  VARIATION: 0.1,               // Deviation from parent traits (+/- 10%)

  // System Simulation
  TICK_MS: 33,                  // ~30 FPS in normal simulation time
  ID_COUNT: 6,

  // AI & Movement Fine-tuning
  WANDER_DIST: 500,             // How far to project the wander vector
  WANDER_ARRIVE_DIST: 2.0,      // Distance to pick a new wander target
  FLEE_PERCEPTION_MULT: 0.9,    // Vision modifier for escaping
  FLEE_PROJECTION: 10,          // Move vector length when fleeing
  EDGE_DIST: 6,                 // Distance from edge to start avoiding
  EDGE_REPULSION: 2.5,          // Strength of edge avoidance
  SPEED_DIVISOR: 5,             // Base speed normalization factor
  MARGIN: 1,                    // Map boundary margin
  REPRODUCTION_INC: 0.2,        // Daily desire increase
  STUCK_DIST_THRESHOLD: 0.5,    // Min distance intended to move to check if stuck
  STUCK_ACTUAL_THRESHOLD: 0.05, // Max distance actually moved to be considered stuck
  STUCK_WIGGLE: 5,              // Random bounce distance when stuck
  SEARCH_CANDIDATES: 3,         // Randomly pick from top N closest targets (food/water)
};

export const COLORS = {
  water: '#3498db',    // Light blue
  land: '#2ecc71',     // Green
  prey: '#bdc3c7',     // Gray (Prey / Rabbits)
  predator: '#e74c3c', // Red (Predators / Foxes)
  food: '#27ae60',     // Dark green (Plant / Food)
};

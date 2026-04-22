import { Tile, TileType, SimulationParams } from '../types';

export const generateTerrain = (params: SimulationParams): Tile[][] => {
  const { width, height, selectedTerrain } = params;
  const grid: Tile[][] = [];

  for (let y = 0; y < height; y++) {
    const row: Tile[] = [];
    for (let x = 0; x < width; x++) {
      let type: TileType = 'land';

      const nx = x / width; // Normalized [0, 1]
      const ny = y / height;

      if (selectedTerrain === 'lake') {
        const dx = nx - 0.5;
        const dy = ny - 0.5;
        // Diameter of lake
        if (Math.sqrt(dx * dx + dy * dy) < 0.25) type = 'water';
      } else if (selectedTerrain === 'river') {
        // Curve river through the middle
        const riverCenter = 0.5 + Math.sin(nx * Math.PI * 2) * 0.15;
        if (Math.abs(ny - riverCenter) < 0.1) type = 'water';
      } else if (selectedTerrain === 'multi-lake') {
        // Three lakes at specific points
        const d1 = Math.sqrt(Math.pow(nx - 0.25, 2) + Math.pow(ny - 0.3, 2));
        const d2 = Math.sqrt(Math.pow(nx - 0.75, 2) + Math.pow(ny - 0.3, 2));
        const d3 = Math.sqrt(Math.pow(nx - 0.5, 2) + Math.pow(ny - 0.75, 2));
        if (d1 < 0.15 || d2 < 0.15 || d3 < 0.15) type = 'water';
      } else if (selectedTerrain === 'puddles') {
        // Multiple small ponds/puddles scattered
        const val = Math.sin(nx * Math.PI * 4) * Math.cos(ny * Math.PI * 4);
        if (val > 0.4) type = 'water';
      }

      row.push({ x, y, type });
    }
    grid.push(row);
  }

  return grid;
};

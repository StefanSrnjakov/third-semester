export function watershed(
  gradient: Uint8Array,
  width: number,
  height: number
): Int32Array {
  const n = gradient.length
  const regions = new Int32Array(n).fill(-1)
  let currentRegion = 0

  // Pre-bucket pixels by height for speed & determinism
  const buckets: number[][] = Array.from({ length: 256 }, () => [])
  for (let i = 0; i < n; i++) buckets[gradient[i]].push(i)

  // Helper for 4-neighborhood
  const nbrOffsets = [-1, 1, -width, width]

  // Visited map for plateau expansion (only used within a level)
  const visited = new Uint8Array(n)

  for (let h = 0; h <= 255; h++) {
    const pixels = buckets[h]

    // reset visited marks for this level's pixels only (cheap)
    for (const idx of pixels) visited[idx] = 0

    for (const start of pixels) {
      if (regions[start] !== -1) continue      // already assigned earlier
      if (visited[start]) continue             // already part of a processed plateau

      // 1) BFS plateau (all connected pixels with gradient == h)
      const queue: number[] = [start]
      visited[start] = 1
      const plateau: number[] = [start]

      // Collect neighboring region ids around the plateau
      const neighborRegions = new Set<number>()

      while (queue.length) {
        const i = queue.pop()!
        const x = i % width

        for (const off of nbrOffsets) {
          const j = i + off

          // bounds checks (handle left/right edges)
          if (j < 0 || j >= n) continue
          if (off === -1 && x === 0) continue
          if (off === 1 && x === width - 1) continue

          const r = regions[j]
          if (r >= 0) neighborRegions.add(r)

          if (!visited[j] && regions[j] === -1 && gradient[j] === h) {
            visited[j] = 1
            queue.push(j)
            plateau.push(j)
          }
        }
      }

      // 2) Assign whole plateau consistently
      if (neighborRegions.size === 0) {
        const id = currentRegion++
        for (const p of plateau) regions[p] = id
      } else if (neighborRegions.size === 1) {
        const id = neighborRegions.values().next().value!
        for (const p of plateau) regions[p] = id
      } else {
        // watershed boundary
        for (const p of plateau) regions[p] = -1
      }
    }
  }

  return regions
}

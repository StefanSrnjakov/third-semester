export function watershed(
  gradient: Uint8Array,
  width: number,
  _height: number
): Int32Array {
  const n = gradient.length
  const regions = new Int32Array(n).fill(-1)
  let currentRegion = 0

  const buckets: number[][] = Array.from({ length: 256 }, () => [])
  for (let i = 0; i < n; i++) {
    buckets[gradient[i]].push(i)
  }

  const nbrOffsets = [-1, 1, -width, width, -width-1, -width+1, width-1, width+1]
  const visited = new Uint8Array(n)

  for (let v = 0; v <= 255; v++) {
    const pixels = buckets[v]
    for (const idx of pixels) visited[idx] = 0

    // Expand existing regions
    for (const idx of pixels) {
      if (regions[idx] !== -1 || visited[idx]) continue
      
      const x = idx % width
      let neighborRegion = -1
      
      for (const off of nbrOffsets) {
        const j = idx + off
        if (j < 0 || j >= n) continue
        if ((off === -1 || off === -width-1 || off === width-1) && x === 0) continue
        if ((off === 1 || off === -width+1 || off === width+1) && x === width - 1) continue
        
        if (regions[j] >= 0) {
          neighborRegion = regions[j]
          break
        }
      }
      
      if (neighborRegion >= 0) {
        const queue: number[] = [idx]
        visited[idx] = 1
        regions[idx] = neighborRegion
        
        while (queue.length > 0) {
          const curr = queue.shift()!
          const cx = curr % width
          
          for (const off of nbrOffsets) {
            const j = curr + off
            if (j < 0 || j >= n) continue
            if ((off === -1 || off === -width-1 || off === width-1) && cx === 0) continue
            if ((off === 1 || off === -width+1 || off === width+1) && cx === width - 1) continue
            
            if (!visited[j] && regions[j] === -1 && gradient[j] === v) {
              visited[j] = 1
              regions[j] = neighborRegion
              queue.push(j)
            }
          }
        }
      }
    }

    // Create new regions
    for (const idx of pixels) {
      if (regions[idx] !== -1) continue
      
      const newRegion = currentRegion++
      const queue: number[] = [idx]
      visited[idx] = 1
      regions[idx] = newRegion
      
      while (queue.length > 0) {
        const curr = queue.shift()!
        const cx = curr % width
        
        for (const off of nbrOffsets) {
          const j = curr + off
          if (j < 0 || j >= n) continue
          if ((off === -1 || off === -width-1 || off === width-1) && cx === 0) continue
          if ((off === 1 || off === -width+1 || off === width+1) && cx === width - 1) continue
          
          if (!visited[j] && regions[j] === -1 && gradient[j] === v) {
            visited[j] = 1
            regions[j] = newRegion
            queue.push(j)
          }
        }
      }
    }
  }

  return regions
}

export function morphologicalGradient(
  image: Uint8Array,
  width: number,
  height: number
): Uint8Array {
  const dilated = dilate(image, width, height)
  const eroded = erode(image, width, height)
  const gradient = new Uint8Array(image.length)
  
  for (let i = 0; i < gradient.length; i++) {
    gradient[i] = dilated[i] - eroded[i]
  }
  
  return gradient
}

export function removeSmallEdges(
  edges: Uint8Array,
  _width: number,
  _height: number,
  threshold: number
): Uint8Array {
  const result = new Uint8Array(edges.length)
  
  for (let i = 0; i < edges.length; i++) {
    result[i] = edges[i] > threshold ? edges[i] : 0
  }
  
  return result
}

function dilate(image: Uint8Array, width: number, height: number): Uint8Array {
  const result = new Uint8Array(image.length)
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      let max = 0
      
      // 3x3 kernel
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            max = Math.max(max, image[ny * width + nx])
          }
        }
      }
      
      result[idx] = max
    }
  }
  
  return result
}

function erode(image: Uint8Array, width: number, height: number): Uint8Array {
  const result = new Uint8Array(image.length)
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      let min = 255
      
      // 3x3 kernel
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            min = Math.min(min, image[ny * width + nx])
          }
        }
      }
      
      result[idx] = min
    }
  }
  
  return result
}


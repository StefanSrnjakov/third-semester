import { morphologicalGradient, removeSmallEdges } from './MorphologicalOps'
import { watershed } from './Watershed'

export interface EdgeData {
  original: string
  edges: string
  cleanedEdges: string
  grayscale: Uint8Array
  cleaned: Uint8Array
  width: number
  height: number
}

export async function detectEdges(file: File, threshold: number): Promise<EdgeData> {
  const img = await loadImage(file)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  
  canvas.width = img.width
  canvas.height = img.height
  ctx.drawImage(img, 0, 0)
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const grayscale = toGrayscale(imageData)
  const edges = morphologicalGradient(grayscale, canvas.width, canvas.height)
  const cleaned = removeSmallEdges(edges, threshold)
  
  return {
    original: canvas.toDataURL(),
    edges: grayscaleToDataURL(edges, canvas.width, canvas.height),
    cleanedEdges: grayscaleToDataURL(cleaned, canvas.width, canvas.height),
    grayscale,
    cleaned,
    width: canvas.width,
    height: canvas.height
  }
}

export function applyWatershed(edgeData: EdgeData): string {
  const regions = watershed(edgeData.cleaned, edgeData.width, edgeData.height)
  return segmentedToDataURL(regions, edgeData.width, edgeData.height)
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

function toGrayscale(imageData: ImageData): Uint8Array {
  const gray = new Uint8Array(imageData.width * imageData.height)
  const data = imageData.data
  
  for (let i = 0; i < gray.length; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
  }
  
  return gray
}

function grayscaleToDataURL(data: Uint8Array, width: number, height: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  const imageData = ctx.createImageData(width, height)
  
  for (let i = 0; i < data.length; i++) {
    const val = data[i]
    imageData.data[i * 4] = val
    imageData.data[i * 4 + 1] = val
    imageData.data[i * 4 + 2] = val
    imageData.data[i * 4 + 3] = 255
  }
  
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL()
}

function segmentedToDataURL(regions: Int32Array, width: number, height: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  const imageData = ctx.createImageData(width, height)
  
  let maxRegion = 0
  for (let i = 0; i < regions.length; i++) {
    if (regions[i] > maxRegion) maxRegion = regions[i]
  }
  const colors = generateColors(maxRegion + 1)
  
  for (let i = 0; i < regions.length; i++) {
    const region = regions[i]
    const color = colors[region]
    imageData.data[i * 4] = color[0]
    imageData.data[i * 4 + 1] = color[1]
    imageData.data[i * 4 + 2] = color[2]
    imageData.data[i * 4 + 3] = 255
  }
  
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL()
}

function generateColors(count: number): number[][] {
  const colors: number[][] = []
  for (let i = 0; i < count; i++) {
    const hue = (i * 137.508) % 360
    colors.push(hslToRgb(hue, 70, 50))
  }
  return colors
}

function hslToRgb(h: number, s: number, l: number): number[] {
  s /= 100
  l /= 100
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))]
}

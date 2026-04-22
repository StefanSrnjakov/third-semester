import { useState } from 'react'
import { detectEdges, applyWatershed, EdgeData } from './ImageProcessor'
import './App.css'

function App() {
  const [edgeData, setEdgeData] = useState<EdgeData | null>(null)
  const [watershedResult, setWatershedResult] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(10)
  const [processing, setProcessing] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploadedFile(file)
    setWatershedResult(null)
    await processEdges(file)
  }

  const processEdges = async (file: File) => {
    setProcessing(true)
    try {
      const result = await detectEdges(file, threshold)
      setEdgeData(result)
    } catch (error) {
      console.error('Error processing image:', error)
      alert('Error processing image')
    }
    setProcessing(false)
  }

  const handleThresholdChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const newThreshold = Number(e.target.value)
    setThreshold(newThreshold)
    setWatershedResult(null)
    
    if (uploadedFile) {
      await processEdges(uploadedFile)
    }
  }

  const handleRunWatershed = () => {
    if (!edgeData) return
    
    setProcessing(true)
    try {
      const result = applyWatershed(edgeData)
      setWatershedResult(result)
    } catch (error) {
      console.error('Error running watershed:', error)
      alert('Error running watershed')
    }
    setProcessing(false)
  }

  return (
    <div className="app">
      <h1>Watershed Algorithm - Edge Detection & Segmentation</h1>
      
      <div className="controls">
        <div className="control-group">
          <label htmlFor="image-upload">Upload Image:</label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            disabled={processing}
          />
        </div>
        
        <div className="control-group">
          <label htmlFor="threshold">Edge Removal Threshold: {threshold}</label>
          <input
            id="threshold"
            type="range"
            min="1"
            max="150"
            value={threshold}
            onChange={handleThresholdChange}
            disabled={processing}
          />
        </div>
      </div>

      {processing && <div className="loading">Processing...</div>}

      {edgeData && (
        <>
          <div className="results">
            <div className="image-container">
              <h3>Original Image</h3>
              <img src={edgeData.original} alt="Original" />
            </div>
            
            <div className="image-container">
              <h3>Edge Detection (Morphological Gradient)</h3>
              <img src={edgeData.edges} alt="Edges" />
            </div>
            
            <div className="image-container">
              <h3>Cleaned Edges (After Threshold Removal)</h3>
              <img src={edgeData.cleanedEdges} alt="Cleaned Edges" />
            </div>
          </div>

          <div className="watershed-control">
            <button 
              onClick={handleRunWatershed} 
              disabled={processing}
              className="watershed-button"
            >
              Run Watershed Segmentation
            </button>
          </div>

          {watershedResult && (
            <div className="results">
              <div className="image-container full-width">
                <h3>Watershed Segmentation Result</h3>
                <img src={watershedResult} alt="Watershed" />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App

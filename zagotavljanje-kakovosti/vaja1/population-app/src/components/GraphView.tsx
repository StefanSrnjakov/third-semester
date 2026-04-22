import React, { useRef, useEffect } from 'react';

interface GraphViewProps {
  history: { [key: number]: number[] };
  width: number;
  height: number;
  colors: string[];
}

export const GraphView: React.FC<GraphViewProps> = ({ history, width, height, colors }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const pad = 40;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;

    let maxPop = 10;
    let maxCycles = 10;

    Object.values(history).forEach(h => {
      maxCycles = Math.max(maxCycles, h.length);
      h.forEach(val => maxPop = Math.max(maxPop, val));
    });

    maxPop = Math.ceil(maxPop * 1.1);

    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(width - pad, height - pad);
    ctx.stroke();

    ctx.fillStyle = '#000';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    const ySteps = [0, maxPop / 2, maxPop];
    ySteps.forEach(pop => {
      const y = (height - pad) - (pop / maxPop) * innerH;
      ctx.fillText(Math.round(pop).toString(), pad - 5, y);
    });

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    const xSteps = [0, maxCycles / 2, maxCycles];
    xSteps.forEach(cycle => {
      const x = pad + (cycle / maxCycles) * innerW;
      ctx.fillText(Math.round(cycle).toString(), x, height - pad + 5);
    });

    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    Object.keys(history).forEach(key => {
      const typeId = parseInt(key);
      const data = history[typeId];
      if (data.length === 0) return;

      ctx.strokeStyle = colors[typeId];
      ctx.beginPath();
      
      data.forEach((val, idx) => {
        const x = pad + (idx / Math.max(1, maxCycles - 1)) * innerW;
        const y = (height - pad) - (val / maxPop) * innerH;
        
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      ctx.stroke();
    });

  }, [history, width, height, colors]);

  return (
    <div>
      <h2>Population Graph</h2>
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  );
};

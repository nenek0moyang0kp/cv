'use client'

import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'

interface Detection {
  bbox: [number, number, number, number]
  confidence: number
  class_id: number
}

const labelMap: Record<number, string> = {
  0: 'Person',
  1: 'Bicycle',
  2: 'Car',
  3: 'Motorcycle',
  4: 'Bus',
  5: 'Truck',
}

function getSummary(detections: Detection[]): Record<string, number> {
  const summary: Record<string, number> = {}
  for (const d of detections) {
    const label = labelMap[d.class_id] ?? `Class ${d.class_id}`
    summary[label] = (summary[label] || 0) + 1
  }
  return summary
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Detection[] | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      setPreviewUrl(URL.createObjectURL(selected))
      setPredictions(null)
      setVideoUrl(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()

      if (data.output_path) {
        setVideoUrl(`http://localhost:3000${data.output_path}`)
        setPredictions(data.result || null)
        setPreviewUrl(null)
      } else if (data.result) {
        setPredictions(data.result)
      } else {
        alert('Tidak ada hasil deteksi.')
      }
    } catch (err) {
      console.error(err)
      alert('Gagal mengirim ke server.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (canvas && ctx && previewUrl && predictions) {
      const img = new Image()
      img.src = previewUrl
      img.onload = () => {
        canvas.width = img.width
        canvas.height = img.height
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0)

        ctx.lineWidth = 2
        ctx.font = '14px Arial'

        predictions.forEach(({ bbox, class_id }) => {
          const [x1, y1, x2, y2] = bbox
          const label = labelMap[class_id] ?? `Class ${class_id}`

          let color = 'red'
          switch (label.toLowerCase()) {
            case 'car': color = 'red'; break
            case 'bus': color = 'green'; break
            case 'person': color = 'orange'; break
            case 'motorcycle': color = 'blue'; break
            case 'bicycle': color = 'pink'; break
          }

          ctx.strokeStyle = color
          ctx.fillStyle = color
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          ctx.fillText(label, x1, y2 + 15)
        })
      }
    }
  }, [previewUrl, predictions])

  return (
    <main className="max-w-2xl mx-auto p-6 space-y-6">
      <Card>
        <CardContent className="p-4 space-y-4">
          <Input type="file" accept="image/*,video/*" onChange={handleFileChange} />
          <Button onClick={handleUpload} disabled={loading || !file}>
            {loading ? 'Memproses...' : 'Upload & Deteksi'}
          </Button>

          {predictions && (
            <>
              <h3 className="text-lg font-semibold mt-4">Ringkasan Deteksi:</h3>
              <ul className="text-sm pl-4 list-disc">
                {Object.entries(getSummary(predictions)).map(([label, count]) => (
                  <li key={label}>{label}: {count}</li>
                ))}
              </ul>
            </>
          )}

          {previewUrl && predictions && (
            <canvas ref={canvasRef} className="w-full rounded-xl shadow border" />
          )}

          {videoUrl && (
            <div className="mt-4 space-y-2">
              <h3 className="text-lg font-semibold">Hasil Deteksi Video:</h3>
              <video
                src={videoUrl}
                controls
                className="w-full rounded-xl shadow border"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  )
}

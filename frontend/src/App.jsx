import { useEffect, useRef, useState, useCallback } from 'react'
import { Camera, Upload, HardHat, ShieldAlert, ShieldCheck, User, Activity, CircleDot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Separator } from '@/components/ui/separator'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const FRAME_INTERVAL_MS = 1500

const VIOLATION_CLASSES = new Set(['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'])
const COMPLIANT_CLASSES = new Set(['Hardhat', 'Mask', 'Safety Vest'])

function classifyDetection(name) {
  if (VIOLATION_CLASSES.has(name)) return 'violation'
  if (COMPLIANT_CLASSES.has(name)) return 'compliant'
  return 'neutral'
}

function HealthBadge() {
  const [state, setState] = useState({ label: 'connecting…', kind: 'secondary' })
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/health`)
        const d = await r.json()
        if (!alive) return
        setState(
          d.model_loaded
            ? { label: 'Online', kind: 'success' }
            : { label: 'Model not loaded', kind: 'destructive' },
        )
      } catch {
        if (alive) setState({ label: 'Offline', kind: 'destructive' })
      }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])
  const className =
    state.kind === 'success'
      ? 'bg-success text-success-foreground hover:bg-success/90'
      : ''
  const variant = state.kind === 'success' ? 'default' : state.kind
  return (
    <Badge variant={variant} className={className}>
      <CircleDot className="mr-1 h-3 w-3" />
      {state.label}
    </Badge>
  )
}

function ComplianceSummary({ counts }) {
  const personCount = counts.person ?? 0
  const violations = (counts.no_hardhat ?? 0) + (counts.no_vest ?? 0) + (counts.no_mask ?? 0)
  if (personCount === 0) {
    return (
      <Alert>
        <Activity className="h-4 w-4" />
        <AlertTitle>No people detected</AlertTitle>
        <AlertDescription>The frame contains no detectable workers right now.</AlertDescription>
      </Alert>
    )
  }
  if (violations > 0) {
    return (
      <Alert variant="destructive">
        <ShieldAlert className="h-4 w-4" />
        <AlertTitle>PPE violations detected</AlertTitle>
        <AlertDescription>
          {violations} missing item{violations === 1 ? '' : 's'} across {personCount} {personCount === 1 ? 'person' : 'people'}.
        </AlertDescription>
      </Alert>
    )
  }
  return (
    <Alert className="border-success/50 [&>svg]:text-success">
      <ShieldCheck className="h-4 w-4" />
      <AlertTitle className="text-success">All compliant</AlertTitle>
      <AlertDescription>
        {personCount} {personCount === 1 ? 'person' : 'people'} detected with no missing PPE.
      </AlertDescription>
    </Alert>
  )
}

function MetricCard({ icon: Icon, label, value, tone = 'default' }) {
  const tones = {
    default: 'text-foreground',
    good: 'text-success',
    bad: 'text-destructive',
  }
  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4">
        <div className={`rounded-md bg-muted p-2 ${tones[tone]}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <div className="text-xs text-muted-foreground">{label}</div>
          <div className={`text-2xl font-semibold leading-none ${tones[tone]}`}>{value ?? 0}</div>
        </div>
      </CardContent>
    </Card>
  )
}

function MetricsGrid({ counts }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
      <MetricCard icon={User} label="People" value={counts.person ?? 0} />
      <MetricCard icon={HardHat} label="Hardhats" value={counts.hardhat ?? 0} tone="good" />
      <MetricCard icon={ShieldAlert} label="No Hardhat" value={counts.no_hardhat ?? 0} tone="bad" />
      <MetricCard icon={ShieldCheck} label="Vests" value={counts.vest ?? 0} tone="good" />
      <MetricCard icon={ShieldAlert} label="No Vest" value={counts.no_vest ?? 0} tone="bad" />
      <MetricCard icon={Activity} label="Total" value={counts.total ?? 0} />
    </div>
  )
}

function DetectionsTable({ detections }) {
  if (!detections || detections.length === 0) {
    return (
      <div className="px-4 py-6 text-center text-sm text-muted-foreground">No detections yet.</div>
    )
  }
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence)
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Class</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Confidence</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((d, i) => {
            const kind = classifyDetection(d.class_name)
            const variant =
              kind === 'violation' ? 'destructive' :
              kind === 'compliant' ? 'default' : 'secondary'
            const className =
              kind === 'compliant' ? 'bg-success text-success-foreground hover:bg-success/90' : ''
            const label = kind === 'violation' ? 'Violation' : kind === 'compliant' ? 'OK' : 'Info'
            return (
              <TableRow key={i}>
                <TableCell className="font-medium">{d.class_name}</TableCell>
                <TableCell>
                  <Badge variant={variant} className={className}>{label}</Badge>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {(d.confidence * 100).toFixed(1)}%
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

function drawDetections(canvas, data) {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.lineWidth = 3
  ctx.font = 'bold 14px sans-serif'
  ctx.textBaseline = 'top'
  for (const d of data.detections) {
    const kind = classifyDetection(d.class_name)
    const stroke =
      kind === 'violation' ? '#ef4444' :
      kind === 'compliant' ? '#22c55e' : '#3b82f6'
    ctx.strokeStyle = stroke
    ctx.fillStyle = stroke
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1)
    const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`
    const tw = ctx.measureText(label).width + 10
    ctx.fillRect(d.x1, Math.max(d.y1 - 22, 0), tw, 22)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, d.x1 + 5, Math.max(d.y1 - 20, 2))
  }
}

function WebcamTab() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const inFlightRef = useRef(false)
  const [running, setRunning] = useState(false)
  const [data, setData] = useState({ detections: [], counts: {} })
  const [latency, setLatency] = useState('—')
  const [error, setError] = useState('')

  const detectFrame = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || inFlightRef.current || !video.videoWidth) return
    inFlightRef.current = true
    const t0 = performance.now()
    try {
      const tmp = document.createElement('canvas')
      tmp.width = video.videoWidth
      tmp.height = video.videoHeight
      const tctx = tmp.getContext('2d')
      // Mirror horizontally so the frame sent to the backend matches
      // what the user sees (CSS-flipped video below).
      tctx.translate(tmp.width, 0)
      tctx.scale(-1, 1)
      tctx.drawImage(video, 0, 0, tmp.width, tmp.height)
      const blob = await new Promise(res => tmp.toBlob(res, 'image/jpeg', 0.7))
      const fd = new FormData()
      fd.append('file', blob, 'frame.jpg')
      const r = await fetch(`${API_BASE}/api/v1/detect-frame`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error('http ' + r.status)
      const json = await r.json()
      drawDetections(canvas, json)
      setData(json)
      setLatency(`${(performance.now() - t0).toFixed(0)} ms`)
    } catch {
      setLatency('error')
    } finally {
      inFlightRef.current = false
    }
  }, [])

  const stop = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setRunning(false)
    setData({ detections: [], counts: {} })
    setLatency('—')
  }, [])

  const start = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera API not available. Use http://localhost (not file://) and a modern browser.')
      return
    }
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }, audio: false,
      })
    } catch (e) {
      setError('Camera access denied or unavailable: ' + e.message)
      return
    }
    streamRef.current = stream
    const video = videoRef.current
    video.srcObject = stream
    await new Promise(res => { video.onloadedmetadata = res })
    canvasRef.current.width = video.videoWidth
    canvasRef.current.height = video.videoHeight
    setRunning(true)
    timerRef.current = setInterval(detectFrame, FRAME_INTERVAL_MS)
  }

  useEffect(() => stop, [stop])

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
          <div>
            <CardTitle className="text-base">Live Webcam</CardTitle>
            <CardDescription>Detections refresh every {FRAME_INTERVAL_MS} ms</CardDescription>
          </div>
          <div className="flex gap-2">
            {!running ? (
              <Button onClick={start} size="sm"><Camera className="mr-2 h-4 w-4" />Start</Button>
            ) : (
              <Button onClick={stop} size="sm" variant="destructive">Stop</Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative w-full overflow-hidden rounded-md border bg-black" style={{ aspectRatio: '4 / 3' }}>
            <video ref={videoRef} autoPlay playsInline muted
              className="absolute inset-0 h-full w-full object-contain"
              style={{ display: running ? 'block' : 'none', transform: 'scaleX(-1)' }} />
            <canvas ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              style={{ display: running ? 'block' : 'none' }} />
            {!running && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">
                {error || 'Click Start to enable camera'}
              </div>
            )}
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
            <span><span className="text-success">green = compliant</span>, <span className="text-destructive">red = violation</span></span>
            <span>{latency}</span>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4 lg:col-span-2">
        <ComplianceSummary counts={data.counts} />
        <MetricsGrid counts={data.counts} />
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Detections</CardTitle></CardHeader>
          <CardContent className="p-0"><DetectionsTable detections={data.detections} /></CardContent>
        </Card>
      </div>
    </div>
  )
}

function UploadTab() {
  const [status, setStatus] = useState('idle')
  const [resultUrl, setResultUrl] = useState('')
  const [details, setDetails] = useState({ detections: [], counts: {} })
  const [error, setError] = useState('')

  const onSubmit = async (e) => {
    e.preventDefault()
    const input = e.currentTarget.elements.file
    const f = input.files[0]
    if (!f) return
    setStatus('detecting')
    setError(''); setResultUrl(''); setDetails({ detections: [], counts: {} })
    try {
      const fd1 = new FormData(); fd1.append('file', f)
      const fd2 = new FormData(); fd2.append('file', f)
      const [imgRes, jsonRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/detect-image?format=jpeg`, { method: 'POST', body: fd1 }),
        fetch(`${API_BASE}/api/v1/detect-image?format=json`, { method: 'POST', body: fd2 }),
      ])
      if (!imgRes.ok || !jsonRes.ok) throw new Error('detection failed')
      setResultUrl(URL.createObjectURL(await imgRes.blob()))
      setDetails(await jsonRes.json())
      setStatus('done')
    } catch (err) {
      setStatus('error')
      setError(err.message)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader>
          <CardTitle className="text-base">Upload an Image</CardTitle>
          <CardDescription>Pick any photo to inspect PPE compliance.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={onSubmit} className="flex flex-wrap items-center gap-2">
            <input type="file" name="file" accept="image/*" required
              className="block w-full max-w-xs cursor-pointer rounded-md border bg-background px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-secondary file:px-3 file:py-1 file:text-sm" />
            <Button type="submit" disabled={status === 'detecting'}>
              <Upload className="mr-2 h-4 w-4" />
              {status === 'detecting' ? 'Detecting…' : 'Detect'}
            </Button>
          </form>

          {status === 'error' && (
            <Alert variant="destructive">
              <ShieldAlert className="h-4 w-4" />
              <AlertTitle>Detection failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {resultUrl && (
            <div className="overflow-hidden rounded-md border bg-black">
              <img src={resultUrl} alt="annotated" className="mx-auto max-h-[70vh] w-auto object-contain" />
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4 lg:col-span-2">
        {status === 'done' && <ComplianceSummary counts={details.counts} />}
        <MetricsGrid counts={details.counts} />
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Detections</CardTitle></CardHeader>
          <CardContent className="p-0"><DetectionsTable detections={details.detections} /></CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <div className="dark min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
        <div className="container flex h-14 items-center justify-between">
          <div className="flex items-center gap-2">
            <HardHat className="h-5 w-5" />
            <span className="font-semibold tracking-tight">PPE Detection</span>
          </div>
          <HealthBadge />
        </div>
      </header>

      <main className="container py-6">
        <Tabs defaultValue="webcam" className="space-y-4">
          <TabsList>
            <TabsTrigger value="webcam"><Camera className="mr-2 h-4 w-4" />Webcam</TabsTrigger>
            <TabsTrigger value="upload"><Upload className="mr-2 h-4 w-4" />Upload Image</TabsTrigger>
          </TabsList>
          <Separator />
          <TabsContent value="webcam"><WebcamTab /></TabsContent>
          <TabsContent value="upload"><UploadTab /></TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

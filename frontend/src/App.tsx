import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Html, useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import {
  Sparkles,
  Image as ImageIcon,
  Box,
  FolderOpen,
  Download,
  RefreshCw,
  Trash2,
  Eye,
  Sliders,
  Cpu,
  Layers,
  Palette,
  Ruler,
  RotateCw,
  Upload,
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8765'

interface Preset {
  key: string
  label: string
  target_height_m: number
  target_faces: number
  palette_colors: number
  prompt_hint: string
}

interface AssetMeta {
  id: string
  name: string
  slug: string
  preset: string
  created: string
  has_dense?: boolean
  options: {
    target_faces: number
    palette_colors: number
    target_height_m: number
    saturation_boost: number
    front_yaw_deg: number
  }
  stats: {
    dense_faces: number
    final_faces: number
    palette_colors: number
    size_m: [number, number, number]
  }
  timings: Record<string, number>
  vram: string
  dir: string
  urls: {
    glb: string
    obj: string
    mtl: string
    texture: string
    thumbnail: string
    turnaround: string
    source: string
    prepared: string
  }
}

function GLBModel({
  url,
  wireframe,
  autoRotate,
}: {
  url: string
  wireframe: boolean
  autoRotate: boolean
}) {
  const { scene } = useGLTF(url)
  const cloned = useMemo(() => {
    const c = scene.clone(true)
    c.traverse((obj: any) => {
      if (obj.isMesh && obj.material) {
        obj.material = obj.material.clone()
        obj.material.wireframe = wireframe
        if (obj.material.map) {
          obj.material.map.magFilter = THREE.NearestFilter
          obj.material.map.minFilter = THREE.NearestFilter
          obj.material.map.needsUpdate = true
        }
      }
    })
    return c
  }, [scene, wireframe])

  return <primitive object={cloned} />
}

const DEFAULT_PRESETS: Preset[] = [
  { key: 'crop', label: 'Cây trồng / rau củ', target_height_m: 0.4, target_faces: 800, palette_colors: 12, prompt_hint: 'single stylized vegetable crop, plump clean shape' },
  { key: 'fruit', label: 'Trái cây / nông sản', target_height_m: 0.25, target_faces: 800, palette_colors: 12, prompt_hint: 'single fresh stylized fruit, plump rounded shape' },
  { key: 'tree', label: 'Cây lớn', target_height_m: 4.0, target_faces: 2000, palette_colors: 14, prompt_hint: 'single stylized low-poly farm tree, thick trunk, chunky foliage' },
  { key: 'animal', label: 'Con vật', target_height_m: 1.2, target_faces: 3000, palette_colors: 16, prompt_hint: 'single cute stylized farm animal, full body, standing, four legs visible' },
  { key: 'building', label: 'Nhà / chuồng trại', target_height_m: 4.0, target_faces: 3000, palette_colors: 16, prompt_hint: 'single cute miniature farm building, isometric front-three-quarter view' },
  { key: 'tool', label: 'Dụng cụ', target_height_m: 1.0, target_faces: 1200, palette_colors: 12, prompt_hint: 'single stylized farm tool prop, thick sturdy proportions' },
  { key: 'prop', label: 'Đồ vật trang trí', target_height_m: 1.0, target_faces: 1500, palette_colors: 14, prompt_hint: 'single stylized farm prop object, clean solid silhouette' },
  { key: 'character', label: 'Nhân vật', target_height_m: 1.7, target_faces: 4000, palette_colors: 16, prompt_hint: 'single cute chibi farm character, full body, standing, A-pose' },
]

export default function App() {
  const [presets, setPresets] = useState<Preset[]>(DEFAULT_PRESETS)
  const [selectedPreset, setSelectedPreset] = useState<string>('animal')
  const [mode, setMode] = useState<'text' | 'image'>('text')

  // Text to 2D state
  const [subject, setSubject] = useState<string>('a cute white and black dairy cow')
  const [numImages, setNumImages] = useState<number>(2)
  const [seed, setSeed] = useState<string>('42')
  const [candidateImages, setCandidateImages] = useState<string[]>([])
  const [selectedImageIdx, setSelectedImageIdx] = useState<number>(0)

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadPreview, setUploadPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Post-process options
  const [targetFaces, setTargetFaces] = useState<number>(3000)
  const [paletteColors, setPaletteColors] = useState<number>(16)
  const [targetHeight, setTargetHeight] = useState<number>(1.2)
  const [frontYaw, setFrontYaw] = useState<number>(0)


  // Viewer & Library state
  const [library, setLibrary] = useState<AssetMeta[]>([])
  const [activeAsset, setActiveAsset] = useState<AssetMeta | null>(null)
  const [cacheBuster, setCacheBuster] = useState<number>(Date.now())
  const [wireframe, setWireframe] = useState<boolean>(false)
  const [autoRotate, setAutoRotate] = useState<boolean>(true)
  const [showTurnaround, setShowTurnaround] = useState<boolean>(false)

  // Status polling
  const [status, setStatus] = useState<{
    busy: boolean
    stage: string
    percent: number
    error: string | null
    vram: string
  }>({
    busy: false,
    stage: 'Sẵn sàng',
    percent: 0,
    error: null,
    vram: '',
  })

  useEffect(() => {
    fetch(`${API_BASE}/api/presets`)
      .then((r) => r.json())
      .then((data) => {
        if (data.presets) {
          setPresets(data.presets)
        }
      })
      .catch(() => {})
    loadLibrary(true)
  }, [])

  useEffect(() => {
    const timer = setInterval(() => {
      fetch(`${API_BASE}/api/status`)
        .then((r) => r.json())
        .then((s) => setStatus(s))
        .catch(() => {})
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  function loadLibrary(selectFirst = false) {
    fetch(`${API_BASE}/api/library`)
      .then((r) => r.json())
      .then((data) => {
        const items: AssetMeta[] = data.items || []
        setLibrary(items)
        if (selectFirst && items.length > 0 && !activeAsset) {
          selectAsset(items[0])
        }
      })
      .catch(() => {})
  }

  function selectAsset(item: AssetMeta) {
    setActiveAsset(item)
    setCacheBuster(Date.now())
    if (item.options) {
      setTargetFaces(item.options.target_faces)
      setPaletteColors(item.options.palette_colors)
      setTargetHeight(item.options.target_height_m)
      setFrontYaw(item.options.front_yaw_deg || 0)
    }
  }

  function handlePresetChange(key: string) {
    setSelectedPreset(key)
    const p = presets.find((x) => x.key === key)
    if (p) {
      setTargetFaces(p.target_faces)
      setPaletteColors(p.palette_colors)
      setTargetHeight(p.target_height_m)
    }
  }

  async function handleGenerateImages() {
    if (!subject.trim() || status.busy) return
    try {
      const res = await fetch(`${API_BASE}/api/generate/images`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject,
          preset: selectedPreset,
          num_images: numImages,
          seed: seed.trim() ? parseInt(seed.trim(), 10) : null,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Lỗi tạo ảnh')
      setCandidateImages(data.images || [])
      setSelectedImageIdx(0)
    } catch (e: any) {
      alert(e.message)
    }
  }

  async function handleBuild3DFromCandidate() {
    const imgUrl = candidateImages[selectedImageIdx]
    if (!imgUrl || status.busy) return
    try {
      const res = await fetch(`${API_BASE}/api/generate/asset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data_url: imgUrl,
          name: subject || selectedPreset,
          preset: selectedPreset,
          target_faces: targetFaces,
          palette_colors: paletteColors,
          target_height_m: targetHeight,
          front_yaw_deg: frontYaw,
        }),
      })
      const meta = await res.json()
      if (!res.ok) throw new Error(meta.detail || 'Lỗi dựng 3D')
      selectAsset(meta)
      loadLibrary(false)
    } catch (e: any) {
      alert(e.message)
    }
  }

  async function handleBuild3DFromUpload() {
    if (!uploadFile || status.busy) return
    setStatus((s) => ({ ...s, busy: true, stage: 'Chuẩn bị dựng 3D từ ảnh tải lên...', percent: 5 }))
    try {
      const fd = new FormData()
      fd.append('file', uploadFile)
      fd.append('name', uploadFile.name.replace(/\.[^.]+$/, ''))
      fd.append('preset', selectedPreset)
      fd.append('target_faces', String(targetFaces))
      fd.append('palette_colors', String(paletteColors))
      fd.append('target_height_m', String(targetHeight))

      const res = await fetch(`${API_BASE}/api/generate/upload`, {
        method: 'POST',
        body: fd,
      })
      const meta = await res.json()
      if (!res.ok) throw new Error(meta.detail || 'Lỗi dựng 3D từ ảnh')
      selectAsset(meta)
      loadLibrary(false)
    } catch (e: any) {
      alert(e.message)
    }
  }


  async function handleReprocess() {
    if (!activeAsset || status.busy) return
    try {
      const res = await fetch(`${API_BASE}/api/library/${activeAsset.id}/reprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_faces: targetFaces,
          palette_colors: paletteColors,
          target_height_m: targetHeight,
          saturation_boost: 1.15,
          front_yaw_deg: frontYaw,
        }),
      })
      const meta = await res.json()
      if (!res.ok) throw new Error(meta.detail || 'Lỗi tối ưu lại')
      selectAsset(meta)
      loadLibrary(false)
    } catch (e: any) {
      alert(e.message)
    }
  }

  async function handleOpenFolder(id: string) {
    await fetch(`${API_BASE}/api/library/${id}/open-folder`, { method: 'POST' })
  }

  async function handleDeleteAsset(id: string) {
    if (!confirm('Xoá model này khỏi thư viện?')) return
    await fetch(`${API_BASE}/api/library/${id}`, { method: 'DELETE' })
    if (activeAsset?.id === id) setActiveAsset(null)
    loadLibrary(true)
  }

  const quickExamples = [
    { label: 'Bò sữa', text: 'a cute white and black dairy cow', preset: 'animal' },
    { label: 'Gà mái', text: 'a cute fat brown hen chicken', preset: 'animal' },
    { label: 'Cừu trắng', text: 'a fluffy cute white sheep', preset: 'animal' },
    { label: 'Quả cà chua', text: 'a plump red tomato with green leaves on top', preset: 'fruit' },
    { label: 'Cà rốt', text: 'a bright orange carrot with green leaves', preset: 'crop' },
    { label: 'Bí ngô', text: 'a plump orange pumpkin', preset: 'crop' },
    { label: 'Cây táo', text: 'a stylized low poly apple tree with red apples', preset: 'tree' },
    { label: 'Nhà kho đỏ', text: 'a cute red wooden farm barn with white roof trim', preset: 'building' },
    { label: 'Bình tưới cây', text: 'a green metal watering can', preset: 'tool' },
  ]

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 select-none">
      {/* Top Header */}
      <header className="h-12 border-b border-slate-800 bg-slate-900/80 px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
            <Box className="w-5 h-5" />
          </div>
          <div>
            <span className="font-bold tracking-tight text-white">MESH STUDIO</span>
            <span className="ml-2 text-xs px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
              Hunyuan3D-2mini Turbo • Unity Farm • 4GB VRAM
            </span>
          </div>
        </div>

        {/* Live Status & VRAM bar */}
        <div className="flex items-center gap-4 text-xs">
          {status.busy && (
            <div className="flex items-center gap-2 bg-amber-500/15 border border-amber-500/40 text-amber-300 px-3 py-1 rounded-full">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>{status.stage}</span>
              <span className="font-mono font-semibold">{status.percent}%</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 text-slate-400 bg-slate-800/70 px-2.5 py-1 rounded border border-slate-700/60">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-mono">{status.vram || 'RTX 3050 4GB'}</span>
          </div>
        </div>
      </header>

      {/* Main 3-Column Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT PANEL: Generator Controls */}
        <aside className="w-96 border-r border-slate-800 bg-slate-900/50 flex flex-col overflow-y-auto p-4 gap-4 shrink-0">
          {/* Mode Switcher */}
          <div className="grid grid-cols-2 gap-1 p-1 bg-slate-950 rounded-lg border border-slate-800">
            <button
              onClick={() => setMode('text')}
              className={`flex items-center justify-center gap-2 py-2 rounded-md text-xs font-semibold transition ${
                mode === 'text'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              Tạo từ mô tả (Text)
            </button>
            <button
              onClick={() => setMode('image')}
              className={`flex items-center justify-center gap-2 py-2 rounded-md text-xs font-semibold transition ${
                mode === 'image'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              Tạo từ ảnh có sẵn
            </button>
          </div>

          {/* Preset Selector: Interactive 2-Column Visual Grid */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Loại Asset Nông Trại (tự chỉnh kích thước Unity & số đa giác)
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {presets.map((p) => {
                const active = selectedPreset === p.key
                return (
                  <button
                    key={p.key}
                    type="button"
                    onClick={() => handlePresetChange(p.key)}
                    className={`text-left px-2.5 py-1.5 rounded-lg border transition flex flex-col ${
                      active
                        ? 'border-emerald-500 bg-emerald-500/15 text-white shadow-sm'
                        : 'border-slate-800 bg-slate-950/70 text-slate-300 hover:border-slate-700 hover:bg-slate-900'
                    }`}
                  >
                    <span className="text-xs font-semibold truncate flex items-center justify-between">
                      <span>{p.label}</span>
                      {active && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      cao {p.target_height_m}m • {p.target_faces} tris • {p.palette_colors} màu
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {mode === 'text' ? (
            <>
              {/* Prompt Input */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Mô tả vật thể (tiếng Anh ngắn gọn)
                </label>
                <textarea
                  rows={2}
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="Ví dụ: a cute white and black dairy cow"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
                />
                {/* Quick Chips */}
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {quickExamples.map((ex) => (
                    <button
                      key={ex.label}
                      type="button"
                      onClick={() => {
                        setSubject(ex.text)
                        handlePresetChange(ex.preset)
                      }}
                      className={`text-[11px] px-2 py-0.5 rounded border transition ${
                        subject === ex.text && selectedPreset === ex.preset
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                      }`}
                    >
                      + {ex.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Số ảnh nháp 2D</label>
                  <select
                    value={numImages}
                    onChange={(e) => setNumImages(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                  >
                    <option className="bg-slate-900 text-white" value={1}>1 ảnh (nhanh ~12s)</option>
                    <option className="bg-slate-900 text-white" value={2}>2 ảnh (~22s)</option>
                    <option className="bg-slate-900 text-white" value={4}>4 ảnh (~45s)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Seed (để trống = ngẫu nhiên)</label>
                  <input
                    type="text"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    placeholder="Ngẫu nhiên"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs"
                  />
                </div>
              </div>

              <button
                onClick={handleGenerateImages}
                disabled={status.busy}
                className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 font-semibold text-sm flex items-center justify-center gap-2 shadow-lg transition"
              >
                <Sparkles className="w-4 h-4" />
                Bước 1: Vẽ ảnh 2D (SD 1.5)
              </button>

              {/* Candidate 2D Images Grid */}
              {candidateImages.length > 0 && (
                <div className="border border-slate-800 rounded-xl p-3 bg-slate-950/60 flex flex-col gap-3">
                  <div className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                    <span>Chọn mẫu 2D ưng ý nhất:</span>
                    <span className="text-emerald-400">Mẫu #{selectedImageIdx + 1}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {candidateImages.map((src, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedImageIdx(idx)}
                        className={`relative rounded-lg overflow-hidden border-2 transition ${
                          selectedImageIdx === idx
                            ? 'border-emerald-500 ring-2 ring-emerald-500/30'
                            : 'border-slate-800 opacity-75 hover:opacity-100'
                        }`}
                      >
                        <img src={src} alt={`candidate-${idx}`} className="w-full aspect-square object-cover" />
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={handleBuild3DFromCandidate}
                    disabled={status.busy}
                    className="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 font-semibold text-sm flex items-center justify-center gap-2 shadow-lg transition"
                  >
                    <Box className="w-4 h-4" />
                    Bước 2: Dựng Model 3D từ mẫu #{selectedImageIdx + 1}
                  </button>
                </div>
              )}
            </>
          ) : (
            /* Upload Mode */
            <div className="flex flex-col gap-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) {
                    setUploadFile(f)
                    setUploadPreview(URL.createObjectURL(f))
                  }
                }}
              />
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700 hover:border-emerald-500 rounded-xl p-4 flex flex-col items-center justify-center gap-2 cursor-pointer bg-slate-950/50 transition min-h-[180px]"
              >
                {uploadPreview ? (
                  <img src={uploadPreview} alt="preview" className="max-h-44 rounded-lg object-contain" />
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-slate-500" />
                    <span className="text-xs text-slate-400 text-center">
                      Bấm để chọn ảnh PNG/JPG của vật thể (nền tự động được xóa bằng rembg)
                    </span>
                  </>
                )}
              </div>
              <button
                onClick={handleBuild3DFromUpload}
                disabled={!uploadFile || status.busy}
                className="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 font-semibold text-sm flex items-center justify-center gap-2 shadow-lg transition"
              >
                {status.busy ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>{status.stage} ({status.percent}%)</span>
                  </>
                ) : (
                  <>
                    <Box className="w-4 h-4" />
                    <span>Dựng Model 3D từ ảnh này</span>
                  </>
                )}
              </button>

            </div>
          )}

          {/* Low-Poly & Unity Post-process Sliders */}
          <div className="border-t border-slate-800 pt-4 mt-1 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                Thông số Low-Poly & Unity
              </span>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-slate-400" /> Số mặt tam giác (Triangles)
                </span>
                <span className="font-mono text-emerald-400 font-semibold">{targetFaces} tris</span>
              </div>
              <input
                type="range"
                min={200}
                max={8000}
                step={100}
                value={targetFaces}
                onChange={(e) => setTargetFaces(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 flex items-center gap-1">
                  <Palette className="w-3.5 h-3.5 text-slate-400" /> Số màu bảng Palette
                </span>
                <span className="font-mono text-emerald-400 font-semibold">{paletteColors} màu</span>
              </div>
              <input
                type="range"
                min={4}
                max={64}
                step={1}
                value={paletteColors}
                onChange={(e) => setPaletteColors(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1 flex items-center gap-1">
                  <Ruler className="w-3 h-3" /> Chiều cao Unity (m)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="20"
                  value={targetHeight}
                  onChange={(e) => setTargetHeight(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs font-mono"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-400 mb-1 flex items-center gap-1">
                  <RotateCw className="w-3 h-3" /> Xoay trục Y (°)
                </label>
                <input
                  type="number"
                  step="45"
                  value={frontYaw}
                  onChange={(e) => setFrontYaw(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs font-mono"
                />
              </div>
            </div>

            {activeAsset && activeAsset.has_dense && (
              <button
                onClick={handleReprocess}
                disabled={status.busy}
                className="w-full py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-emerald-300 text-xs font-semibold flex items-center justify-center gap-1.5 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Áp dụng ngay cho "{activeAsset.name}" (~1 giây, không chạy lại AI)
              </button>
            )}
          </div>
        </aside>

        {/* CENTER PANEL: Interactive 3D Viewport */}
        <main className="flex-1 flex flex-col bg-slate-950 relative overflow-hidden">
          {/* Viewport Toolbar */}
          <div className="absolute top-3 left-3 right-3 z-20 flex items-center justify-between pointer-events-none">
            <div className="flex items-center gap-2 pointer-events-auto bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-1.5 text-xs">
              <button
                onClick={() => setWireframe(!wireframe)}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  wireframe ? 'bg-emerald-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                Lưới (Wireframe)
              </button>
              <button
                onClick={() => setAutoRotate(!autoRotate)}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  autoRotate ? 'bg-emerald-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                Tự xoay
              </button>
              <button
                onClick={() => setShowTurnaround(!showTurnaround)}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  showTurnaround ? 'bg-emerald-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                Ảnh 4 góc & Palette
              </button>
            </div>

            {activeAsset && (
              <div className="pointer-events-auto flex items-center gap-2 bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
                <span className="text-slate-400">
                  Tam giác: <strong className="text-emerald-400">{activeAsset.stats.final_faces}</strong>
                </span>
                <span className="text-slate-600">•</span>
                <span className="text-slate-400">
                  Màu: <strong className="text-emerald-400">{activeAsset.stats.palette_colors}</strong>
                </span>
                <span className="text-slate-600">•</span>
                <span className="text-slate-400">
                  Kích thước (m):{' '}
                  <strong className="text-slate-200 font-mono">
                    {activeAsset.stats.size_m.join(' × ')}
                  </strong>
                </span>
              </div>
            )}
          </div>

          {/* 3D Canvas */}
          <div className="flex-1 min-h-0 relative">
            {activeAsset ? (
              <Canvas camera={{ position: [1.8, 1.4, 2.2], fov: 45 }}>
                <color attach="background" args={['#090d16']} />
                <ambientLight intensity={0.85} />
                <directionalLight position={[4, 8, 5]} intensity={1.3} />
                <directionalLight position={[-4, 3, -4]} intensity={0.4} />
                <Grid
                  infiniteGrid
                  fadeDistance={12}
                  sectionColor="#1e293b"
                  cellColor="#0f172a"
                  position={[0, 0, 0]}
                />
                <Suspense
                  fallback={
                    <Html center>
                      <div className="px-4 py-2 rounded bg-slate-900 border border-slate-700 text-xs text-emerald-400 whitespace-nowrap">
                        Đang nạp model 3D...
                      </div>
                    </Html>
                  }
                >
                  <GLBModel
                    url={`${API_BASE}${activeAsset.urls.glb}?t=${cacheBuster}`}
                    wireframe={wireframe}
                    autoRotate={autoRotate}
                  />
                </Suspense>
                <OrbitControls
                  makeDefault
                  target={[0, (activeAsset.options?.target_height_m || 1) * 0.45, 0]}
                  autoRotate={autoRotate}
                  autoRotateSpeed={1.5}
                />
              </Canvas>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-2">
                <Box className="w-12 h-12 stroke-1" />
                <p className="text-sm">Chưa có model nào được chọn</p>
              </div>
            )}

            {/* Floating Scrollable Turnaround & Palette Overlay Card */}
            {activeAsset && showTurnaround && (
              <div className="absolute bottom-3 left-3 right-3 z-20 max-h-[60%] overflow-y-auto bg-slate-900/95 backdrop-blur-md border border-slate-700 rounded-xl p-3 shadow-2xl">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5" />
                    Xem trước Ảnh gốc 2D • Bảng màu Palette • Ảnh 4 góc (Trước / Phải / Sau / Trái)
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowTurnaround(false)}
                    className="text-xs px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                  >
                    ✕ Đóng
                  </button>
                </div>
                <div className="flex items-start gap-4 overflow-x-auto pb-1">
                  <div className="shrink-0">
                    <div className="text-[11px] text-slate-400 mb-1">Ảnh gốc 2D</div>
                    <img
                      src={`${API_BASE}${activeAsset.urls.source}?t=${cacheBuster}`}
                      className="h-32 w-32 rounded-lg border border-slate-700 object-cover bg-slate-950"
                    />
                  </div>
                  <div className="shrink-0">
                    <div className="text-[11px] text-slate-400 mb-1">Đã tách nền (RGBA)</div>
                    <img
                      src={`${API_BASE}${activeAsset.urls.prepared}?t=${cacheBuster}`}
                      className="h-32 w-32 rounded-lg border border-slate-700 object-contain bg-slate-950"
                    />
                  </div>
                  <div className="shrink-0">
                    <div className="text-[11px] text-slate-400 mb-1">Bảng màu Palette (64×64)</div>
                    <img
                      src={`${API_BASE}${activeAsset.urls.texture}?t=${cacheBuster}`}
                      style={{ imageRendering: 'pixelated' }}
                      className="h-32 w-32 rounded-lg border border-slate-700"
                    />
                  </div>
                  <div className="shrink-0">
                    <div className="text-[11px] text-slate-400 mb-1">4 góc (Trước / Phải / Sau / Trái)</div>
                    <img
                      src={`${API_BASE}${activeAsset.urls.turnaround}?t=${cacheBuster}`}
                      className="h-32 rounded-lg border border-slate-700"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>


          {/* Bottom Export Bar for Unity */}
          {activeAsset && (
            <div className="h-14 border-t border-slate-800 bg-slate-900 px-4 flex items-center justify-between shrink-0">
              <div className="text-xs text-slate-400 truncate max-w-md">
                Thư mục: <span className="font-mono text-slate-300">{activeAsset.dir}</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleOpenFolder(activeAsset.id)}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  <FolderOpen className="w-3.5 h-3.5 text-amber-400" />
                  Mở thư mục chứa file (kéo vào Unity)
                </button>
                <a
                  href={`${API_BASE}${activeAsset.urls.glb}?t=${cacheBuster}`}
                  download={`${activeAsset.slug}.glb`}
                  className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  Tải .GLB
                </a>
                <a
                  href={`${API_BASE}${activeAsset.urls.obj}?t=${cacheBuster}`}
                  download={`${activeAsset.slug}.obj`}
                  className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  Tải .OBJ
                </a>
              </div>
            </div>
          )}
        </main>

        {/* RIGHT PANEL: Asset Library */}
        <aside className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
          <div className="p-3 border-b border-slate-800 flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Thư viện đã tạo ({library.length})
            </span>
            <button
              onClick={() => loadLibrary(false)}
              className="text-slate-400 hover:text-white p-1 rounded"
              title="Làm mới"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2.5">
            {library.map((item) => {
              const isSelected = activeAsset?.id === item.id
              return (
                <div
                  key={item.id}
                  onClick={() => selectAsset(item)}
                  className={`group rounded-xl border p-2 flex gap-2.5 cursor-pointer transition ${
                    isSelected
                      ? 'border-emerald-500 bg-emerald-500/10'
                      : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
                  }`}
                >
                  <img
                    src={`${API_BASE}${item.urls.thumbnail}?t=${cacheBuster}`}
                    alt={item.name}
                    className="w-16 h-16 rounded-lg object-cover bg-slate-900 shrink-0 border border-slate-800"
                  />
                  <div className="flex-1 min-w-0 flex flex-col justify-between">
                    <div>
                      <div className="text-xs font-semibold text-white truncate">{item.name}</div>
                      <div className="text-[11px] text-slate-400">
                        {item.stats?.final_faces} tris • {item.stats?.palette_colors} màu
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {item.preset}
                      </span>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleOpenFolder(item.id)
                          }}
                          className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-amber-400"
                          title="Mở thư mục"
                        >
                          <FolderOpen className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteAsset(item.id)
                          }}
                          className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-rose-400"
                          title="Xoá"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </aside>
      </div>
    </div>
  )
}

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
  Search,
  ChevronDown,
  Check,
  Plus,
  X,
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8765'

interface Preset {
  key: string
  label: string
  target_height_m: number
  target_faces: number
  palette_colors: number
  prompt_hint: string
  custom?: boolean
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

interface EngineOption {
  key: string
  label: string
  badge: string
  desc: string
}

const DEFAULT_ENGINES: EngineOption[] = [
  {
    key: 'hunyuan3d_char',
    label: 'Hunyuan3D-2 Full (1.1B)',
    badge: 'High • 4.5–6GB VRAM',
    desc: '3,072 Latents, Lưới 384³, 16 bước (~40s). Chi tiết cao nhất.',
  },
  {
    key: 'hunyuan3d_mini_hq',
    label: 'Hunyuan3D-2 Mini Ultra (0.6B)',
    badge: 'Medium • 3.2GB VRAM',
    desc: 'Bản 0.6B mở khóa 3,072 Latents, Lưới 384³ (~25s). Cân bằng chi tiết & tốc độ.',
  },
  {
    key: 'hunyuan3d_mini',
    label: 'Hunyuan3D-2 Mini (0.6B)',
    badge: 'Low • 2.4GB VRAM',
    desc: 'Bản 0.6B (512 Latents), Lưới 256³ (~14s). Nhanh, nhẹ máy.',
  },
  {
    key: 'triposr',
    label: 'TripoSR (Siêu tốc)',
    badge: 'Flash • 1.8GB VRAM',
    desc: 'Model NeRF nhẹ (~5s). Phù hợp phác thảo nhanh.',
  },
]

const DEFAULT_PRESETS: Preset[] = [
  { key: 'default', label: 'Mặc định', target_height_m: 1.0, target_faces: 200000, palette_colors: 1024, prompt_hint: 'single stylized 3d game model, clean well-defined proportions' },
]

export default function App() {
  const [presets, setPresets] = useState<Preset[]>(DEFAULT_PRESETS)
  const [engines, setEngines] = useState<EngineOption[]>(DEFAULT_ENGINES)
  const [selectedEngine, setSelectedEngine] = useState<string>('hunyuan3d_mini_hq')
  const [selectedPreset, setSelectedPreset] = useState<string>('default')
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
  const [targetFaces, setTargetFaces] = useState<number>(200000)
  const [paletteColors, setPaletteColors] = useState<number>(1024)
  const [targetHeight, setTargetHeight] = useState<number>(1.0)
  const [frontYaw, setFrontYaw] = useState<number>(0)
  const [showAddPreset, setShowAddPreset] = useState<boolean>(false)
  const [isEngineOpen, setIsEngineOpen] = useState<boolean>(false)
  const [isPresetOpen, setIsPresetOpen] = useState<boolean>(false)
  const [isExamplesOpen, setIsExamplesOpen] = useState<boolean>(false)
  const [isParamsOpen, setIsParamsOpen] = useState<boolean>(true)

  const engineDropdownRef = useRef<HTMLDivElement>(null)
  const presetDropdownRef = useRef<HTMLDivElement>(null)
  const examplesDropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node
      if (engineDropdownRef.current && !engineDropdownRef.current.contains(target)) {
        setIsEngineOpen(false)
      }
      if (presetDropdownRef.current && !presetDropdownRef.current.contains(target)) {
        setIsPresetOpen(false)
      }
      if (examplesDropdownRef.current && !examplesDropdownRef.current.contains(target)) {
        setIsExamplesOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])
  const [newPreset, setNewPreset] = useState({ label: '', target_height_m: 1.0, target_faces: 200000, palette_colors: 1024 })


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
        if (data.engines) {
          setEngines(data.engines)
        }
        if (data.default_engine) {
          setSelectedEngine(data.default_engine)
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

  async function handleAddPreset() {
    if (!newPreset.label.trim()) return
    try {
      const r = await fetch(`${API_BASE}/api/presets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPreset),
      })
      const data = await r.json()
      if (data.presets) {
        setPresets(data.presets)
        const added = data.presets[data.presets.length - 1]
        if (added) {
          setSelectedPreset(added.key)
          setTargetFaces(added.target_faces)
          setPaletteColors(added.palette_colors)
          setTargetHeight(added.target_height_m)
        }
      }
      setShowAddPreset(false)
      setNewPreset({ label: '', target_height_m: 1.0, target_faces: 200000, palette_colors: 1024 })
    } catch {
      alert('Không thêm được loại asset (backend chưa chạy?)')
    }
  }

  async function handleDeletePreset(key: string) {
    if (!confirm('Xoá loại asset này?')) return
    const r = await fetch(`${API_BASE}/api/presets/${key}`, { method: 'DELETE' })
    const data = await r.json()
    if (data.presets) setPresets(data.presets)
    if (selectedPreset === key) handlePresetChange('default')
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
    setStatus((s) => ({ ...s, busy: true, stage: 'Chuẩn bị dựng 3D...', percent: 5 }))
    try {
      const res = await fetch(`${API_BASE}/api/generate/asset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data_url: imgUrl,
          name: subject || selectedPreset,
          preset: selectedPreset,
          engine: selectedEngine,
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
      fd.append('engine', selectedEngine)
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

  const [exampleSearch, setExampleSearch] = useState<string>('')

  const quickExamples = [
    // Nhân vật & Dân làng
    { label: 'Nông dân nam', category: 'Nhân vật', text: 'a cute stylized male farm worker in overalls, straw hat, standing in A-pose' },
    { label: 'Nữ nông dân', category: 'Nhân vật', text: 'a cute stylized female farmer with apron and bandana, standing in A-pose' },
    { label: 'Cậu bé nông trại', category: 'Nhân vật', text: 'a cute stylized chibi farm boy with dungarees, standing in A-pose' },
    { label: 'Ông lão làm vườn', category: 'Nhân vật', text: 'a cute stylized old gardener with white beard and boots, standing in A-pose' },
    { label: 'Hiệp sĩ tí hon', category: 'Nhân vật', text: 'a cute chibi stylized knight with small wooden sword and shield, standing in A-pose' },

    // Vật nuôi & Gia súc
    { label: 'Bò sữa', category: 'Gia súc', text: 'a cute white and black dairy cow, four legs visible' },
    { label: 'Bò vàng', category: 'Gia súc', text: 'a cute brown farm cattle cow, rounded stylized body' },
    { label: 'Gà mái', category: 'Gia súc', text: 'a cute fat brown hen chicken, stylized feathers' },
    { label: 'Gà trống', category: 'Gia súc', text: 'a colorful stylized farm rooster with red comb, full body' },
    { label: 'Cừu trắng', category: 'Gia súc', text: 'a fluffy cute white sheep with black face, chunky wool' },
    { label: 'Heo con hồng', category: 'Gia súc', text: 'a plump cute pink piglet, rounded clean shape' },
    { label: 'Ngựa nâu', category: 'Gia súc', text: 'a stylized brown farm horse standing, simple mane and tail' },
    { label: 'Chó chăn cừu', category: 'Gia súc', text: 'a cute stylized border collie farm dog, sitting attentively' },
    { label: 'Mèo mướp', category: 'Gia súc', text: 'a cute chunky tabby farm cat sitting, stylized paws' },
    { label: 'Vịt cỏ', category: 'Gia súc', text: 'a cute stylized white duck with orange beak, standing' },
    { label: 'Thỏ trắng', category: 'Gia súc', text: 'a cute chubby white rabbit with long ears' },
    { label: 'Dê núi', category: 'Gia súc', text: 'a cute stylized farm goat with small horns and goatee' },

    // Rau củ & Nông sản
    { label: 'Quả cà chua', category: 'Rau củ', text: 'a plump red tomato with green leaves on top' },
    { label: 'Cà rốt', category: 'Rau củ', text: 'a bright orange carrot with fresh green leaves' },
    { label: 'Bí ngô', category: 'Rau củ', text: 'a plump rounded orange pumpkin with curved green stem' },
    { label: 'Bắp ngô vàng', category: 'Rau củ', text: 'a yellow sweet corn cob with green husks peeled back' },
    { label: 'Quả dưa hấu', category: 'Rau củ', text: 'a large striped green round watermelon with stem' },
    { label: 'Quả dâu tây', category: 'Rau củ', text: 'a plump red strawberry with green sepals and yellow seeds' },
    { label: 'Củ khoai tây', category: 'Rau củ', text: 'a lumpy stylized brown potato with clean surface' },
    { label: 'Bắp cải xanh', category: 'Rau củ', text: 'a layered stylized green head of cabbage' },
    { label: 'Quả ớt đỏ', category: 'Rau củ', text: 'a bright red spicy chili pepper with green calyx' },
    { label: 'Củ tỏi', category: 'Rau củ', text: 'a plump white garlic bulb with paper skin texture' },
    { label: 'Nấm rơm', category: 'Rau củ', text: 'a cute stylized red mushroom with white spots' },

    // Cây cối & Trái cây
    { label: 'Cây táo', category: 'Cây cối', text: 'a stylized low poly apple tree with red apples, chunky canopy' },
    { label: 'Cây cam', category: 'Cây cối', text: 'a cute stylized small orange tree with round orange fruits' },
    { label: 'Cây lúa vàng', category: 'Cây cối', text: 'a golden sheaf of ripe wheat stalks, tied with twine' },
    { label: 'Cây chuối', category: 'Cây cối', text: 'a small tropical banana plant with broad green leaves and yellow bananas' },
    { label: 'Quả xoài chín', category: 'Cây cối', text: 'a plump golden yellow ripe mango fruit' },
    { label: 'Quả thanh long', category: 'Cây cối', text: 'a vibrant pink dragonfruit with green tipped scales' },
    { label: 'Bụi hoa hướng dương', category: 'Cây cối', text: 'a vibrant yellow blooming sunflower with brown center and green stem' },

    // Nhà cửa & Công trình
    { label: 'Nhà kho đỏ', category: 'Công trình', text: 'a cute red wooden farm barn with white roof trim and sliding door' },
    { label: 'Cối xay gió', category: 'Công trình', text: 'a cute rustic wooden farm windmill with four sails' },
    { label: 'Chuồng gà', category: 'Công trình', text: 'a small cute wooden chicken coop with ramp and roost' },
    { label: 'Giếng nước đá', category: 'Công trình', text: 'a round cobblestone water well with wooden roof and bucket' },
    { label: 'Hàng rào gỗ', category: 'Công trình', text: 'a short segment of white wooden picket fence' },

    // Dụng cụ & Đồ vật
    { label: 'Bình tưới cây', category: 'Dụng cụ', text: 'a green vintage metal watering can with nozzle' },
    { label: 'Xe cút kít', category: 'Dụng cụ', text: 'a wooden farm wheelbarrow with metal wheel' },
    { label: 'Thùng gỗ đựng táo', category: 'Dụng cụ', text: 'a rustic wooden crate filled with red apples' },
    { label: 'Xẻng làm vườn', category: 'Dụng cụ', text: 'a garden shovel tool with wooden handle and steel blade' },
    { label: 'Bao tải hạt giống', category: 'Dụng cụ', text: 'a burlap sack full of farm seeds tied with string' },
    { label: 'Đèn bão cổ', category: 'Dụng cụ', text: 'a classic brass kerosene lantern with glass globe' },
  ]

  const filteredExamples = useMemo(() => {
    const q = exampleSearch.trim().toLowerCase()
    if (!q) return quickExamples
    return quickExamples.filter(
      (ex) => ex.label.toLowerCase().includes(q) || ex.text.toLowerCase().includes(q) || (ex.category && ex.category.toLowerCase().includes(q))
    )
  }, [exampleSearch])

  const currentEngine = engines.find((e) => e.key === selectedEngine) || engines[0]
  const currentPreset = presets.find((p) => p.key === selectedPreset) || presets[0]

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
              Hunyuan3D-2 • Unity Farm
            </span>
          </div>
        </div>

        {/* Live Status & Export & VRAM bar */}
        <div className="flex items-center gap-3 text-xs">
          {status.busy && (
            <div className="flex items-center gap-2 bg-amber-500/15 border border-amber-500/40 text-amber-300 px-3 py-1 rounded-full">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>{status.stage}</span>
              <span className="font-mono font-semibold">{status.percent}%</span>
            </div>
          )}

          {/* Cụm nút Tải file đặt bên trái VRAM - chỉ hiện khi có model */}
          {activeAsset && (
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => handleOpenFolder(activeAsset.id)}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition"
                title={`Mở thư mục: ${activeAsset.dir}`}
              >
                <FolderOpen className="w-3.5 h-3.5 text-amber-400" />
                <span className="hidden sm:inline">Thư mục</span>
              </button>
              <a
                href={`${API_BASE}${activeAsset.urls.glb}?t=${cacheBuster}`}
                download={`${activeAsset.slug}.glb`}
                className="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-sm transition"
                title="Tải model .GLB"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Tải .GLB</span>
              </a>
              <a
                href={`${API_BASE}${activeAsset.urls.obj}?t=${cacheBuster}`}
                download={`${activeAsset.slug}.obj`}
                className="px-2.5 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-sm transition"
                title="Tải model .OBJ"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Tải .OBJ</span>
              </a>
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

          {/* 3D AI Model Selector - Dropdown */}
          <div className="relative" ref={engineDropdownRef}>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Model AI Dựng 3D</span>
              <span className="text-[10px] text-emerald-400 font-mono">Tự động VRAM</span>
            </label>
            <button
              type="button"
              onClick={() => {
                setIsEngineOpen(!isEngineOpen)
                setIsPresetOpen(false)
                setIsExamplesOpen(false)
              }}
              className="w-full text-left px-3 py-2 rounded-lg border border-slate-800 bg-slate-950/90 hover:border-slate-700 hover:bg-slate-900/90 transition flex items-center justify-between gap-2 shadow-sm"
            >
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-xs font-semibold text-white truncate">
                  {currentEngine?.label || selectedEngine}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shrink-0">
                  {currentEngine?.badge || ''}
                </span>
              </div>
              <ChevronDown className={`w-4 h-4 text-slate-400 shrink-0 transition-transform duration-200 ${isEngineOpen ? 'rotate-180 text-emerald-400' : ''}`} />
            </button>

            {isEngineOpen && (
              <div className="absolute top-full left-0 right-0 mt-1.5 z-40 bg-slate-900/95 border border-slate-700 rounded-xl shadow-2xl p-1.5 flex flex-col gap-1 backdrop-blur-md">
                {engines.map((eng) => {
                  const active = selectedEngine === eng.key
                  return (
                    <button
                      key={eng.key}
                      type="button"
                      onClick={() => {
                        setSelectedEngine(eng.key)
                        setIsEngineOpen(false)
                      }}
                      className={`text-left px-2.5 py-2 rounded-lg transition flex items-start justify-between gap-2 ${
                        active
                          ? 'bg-emerald-500/15 border border-emerald-500/40 text-white'
                          : 'hover:bg-slate-800 text-slate-300 border border-transparent'
                      }`}
                    >
                      <div className="flex flex-col min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-semibold text-white truncate">{eng.label}</span>
                          <span className="text-[10px] px-1.5 py-0.2 rounded font-mono bg-slate-800 text-slate-300 border border-slate-700 shrink-0">
                            {eng.badge}
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-400 mt-0.5 leading-snug">{eng.desc}</span>
                      </div>
                      {active && <Check className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />}
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Loại Asset - Dropdown + Thêm */}
          <div className="relative" ref={presetDropdownRef}>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Loại Asset</span>
              <span className="text-[10px] text-slate-400">Tự chỉnh kích thước & đa giác</span>
            </label>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  setIsPresetOpen(!isPresetOpen)
                  setIsEngineOpen(false)
                  setIsExamplesOpen(false)
                }}
                className="flex-1 text-left px-3 py-2 rounded-lg border border-slate-800 bg-slate-950/90 hover:border-slate-700 hover:bg-slate-900/90 transition flex items-center justify-between gap-2 shadow-sm min-w-0"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs font-semibold text-white truncate">
                    {currentPreset?.label || 'Mặc định'}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono truncate">
                    {currentPreset?.target_height_m}m • {currentPreset?.target_faces} tris • {currentPreset?.palette_colors} màu
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 shrink-0 transition-transform duration-200 ${isPresetOpen ? 'rotate-180 text-emerald-400' : ''}`} />
              </button>

              <button
                type="button"
                onClick={() => {
                  setShowAddPreset((v) => !v)
                  setIsPresetOpen(false)
                }}
                title="Thêm loại asset mới"
                className={`px-2.5 py-2 rounded-lg border text-xs font-semibold flex items-center gap-1 transition shrink-0 ${
                  showAddPreset
                    ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300'
                    : 'border-slate-800 bg-slate-950/90 hover:bg-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Thêm</span>
              </button>
            </div>

            {isPresetOpen && (
              <div className="absolute top-full left-0 right-0 mt-1.5 z-40 bg-slate-900/95 border border-slate-700 rounded-xl shadow-2xl p-1.5 flex flex-col gap-1 backdrop-blur-md max-h-56 overflow-y-auto">
                {presets.map((p) => {
                  const active = selectedPreset === p.key
                  return (
                    <div
                      key={p.key}
                      className={`flex items-center justify-between px-2.5 py-2 rounded-lg transition ${
                        active
                          ? 'bg-emerald-500/15 border border-emerald-500/40 text-white'
                          : 'hover:bg-slate-800 text-slate-300 border border-transparent'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          handlePresetChange(p.key)
                          setIsPresetOpen(false)
                        }}
                        className="flex-1 text-left flex items-center justify-between pr-2 min-w-0"
                      >
                        <span className="text-xs font-semibold text-white">{p.label}</span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          {p.target_height_m}m • {p.target_faces} tris • {p.palette_colors} màu
                        </span>
                      </button>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {active && <Check className="w-4 h-4 text-emerald-400" />}
                        {p.custom && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeletePreset(p.key)
                            }}
                            className="p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-400"
                            title="Xoá loại này"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {showAddPreset && (
              <div className="mt-2 p-2.5 rounded-xl border border-slate-700 bg-slate-950/95 grid grid-cols-2 gap-2 text-[11px] shadow-xl">
                <div className="col-span-2 flex items-center justify-between text-xs font-semibold text-emerald-400">
                  <span>Thêm loại asset mới</span>
                  <button
                    type="button"
                    onClick={() => setShowAddPreset(false)}
                    className="text-slate-400 hover:text-white"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                <input
                  className="col-span-2 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-emerald-500"
                  placeholder="Tên loại (vd: Cây trồng, Đồ vật...)"
                  value={newPreset.label}
                  onChange={(e) => setNewPreset({ ...newPreset, label: e.target.value })}
                />
                <label className="text-slate-400">Cao (m)
                  <input type="number" step="0.1" min="0.05" className="w-full mt-0.5 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                    value={newPreset.target_height_m}
                    onChange={(e) => setNewPreset({ ...newPreset, target_height_m: Number(e.target.value) })} />
                </label>
                <label className="text-slate-400">Số tris
                  <input type="number" step="500" min="200" max="200000" className="w-full mt-0.5 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                    value={newPreset.target_faces}
                    onChange={(e) => setNewPreset({ ...newPreset, target_faces: Number(e.target.value) })} />
                </label>
                <label className="text-slate-400">Số màu
                  <input type="number" step="8" min="4" max="1024" className="w-full mt-0.5 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                    value={newPreset.palette_colors}
                    onChange={(e) => setNewPreset({ ...newPreset, palette_colors: Number(e.target.value) })} />
                </label>
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={handleAddPreset}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg px-2 py-1.5 font-semibold transition text-xs shadow"
                  >
                    Lưu
                  </button>
                </div>
              </div>
            )}
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
                {/* Quick Examples Dropdown with Search */}
                <div className="relative mt-2" ref={examplesDropdownRef}>
                  <button
                    type="button"
                    onClick={() => {
                      setIsExamplesOpen(!isExamplesOpen)
                      setIsEngineOpen(false)
                      setIsPresetOpen(false)
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-950/70 hover:bg-slate-900 hover:border-slate-700 text-xs text-slate-300 flex items-center justify-between gap-1.5 transition"
                  >
                    <span className="flex items-center gap-1.5 text-slate-400 truncate">
                      <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                      <span>Chọn mẫu mô tả ({quickExamples.length} mẫu)...</span>
                    </span>
                    <ChevronDown className={`w-3.5 h-3.5 text-slate-400 shrink-0 transition-transform duration-200 ${isExamplesOpen ? 'rotate-180 text-emerald-400' : ''}`} />
                  </button>

                  {isExamplesOpen && (
                    <div className="absolute top-full left-0 right-0 mt-1 z-30 bg-slate-900/95 border border-slate-700 rounded-xl shadow-2xl overflow-hidden flex flex-col backdrop-blur-md">
                      {/* Search Bar */}
                      <div className="p-2 border-b border-slate-800 flex items-center gap-1.5 bg-slate-950/60">
                        <Search className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        <input
                          type="text"
                          value={exampleSearch}
                          onChange={(e) => setExampleSearch(e.target.value)}
                          placeholder="Tìm mẫu (bò, dưa hấu, nông dân...)..."
                          className="w-full bg-transparent text-xs text-white placeholder-slate-500 focus:outline-none"
                          autoFocus
                        />
                        {exampleSearch && (
                          <button
                            type="button"
                            onClick={() => setExampleSearch('')}
                            className="text-slate-400 hover:text-white p-0.5"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        )}
                      </div>

                      {/* List */}
                      <div className="p-1 flex flex-col gap-0.5 max-h-60 overflow-y-auto">
                        {filteredExamples.length === 0 ? (
                          <div className="py-4 text-center text-xs text-slate-500">
                            Không tìm thấy mẫu phù hợp
                          </div>
                        ) : (
                          filteredExamples.map((ex) => {
                            const active = subject === ex.text
                            return (
                              <button
                                key={ex.label}
                                type="button"
                                onClick={() => {
                                  setSubject(ex.text)
                                  setIsExamplesOpen(false)
                                  setExampleSearch('')
                                }}
                                className={`text-left px-2.5 py-1.5 rounded-lg transition flex items-center justify-between gap-2 text-xs ${
                                  active
                                    ? 'bg-emerald-500/15 text-emerald-300 font-semibold'
                                    : 'hover:bg-slate-800 text-slate-300'
                                }`}
                              >
                                <span className="font-medium shrink-0 flex items-center gap-1.5">
                                  <span>+ {ex.label}</span>
                                  {ex.category && (
                                    <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-400 font-normal">
                                      {ex.category}
                                    </span>
                                  )}
                                </span>
                                <span className="text-[10px] text-slate-500 italic truncate max-w-[150px]">{ex.text}</span>
                              </button>
                            )
                          })
                        )}
                      </div>
                    </div>
                  )}
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

          {/* Low-Poly & Unity Post-process Sliders - Collapsible */}
          <div className="border-t border-slate-800 pt-3 mt-1 flex flex-col gap-2.5">
            <button
              type="button"
              onClick={() => setIsParamsOpen(!isParamsOpen)}
              className="w-full flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-400 hover:text-slate-200 transition py-0.5"
            >
              <span className="flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                Thông số Low-Poly & Unity
              </span>
              <div className="flex items-center gap-2">
                {!isParamsOpen && (
                  <span className="text-[10px] font-mono font-normal normal-case text-slate-500">
                    {targetFaces.toLocaleString('en-US')} tris • {paletteColors} màu
                  </span>
                )}
                <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${isParamsOpen ? 'rotate-180 text-emerald-400' : ''}`} />
              </div>
            </button>
            {isParamsOpen && (
              <>
                <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-slate-400" /> Số mặt tam giác (Triangles)
                </span>
                <span className="font-mono text-emerald-400 font-semibold">
                  {targetFaces.toLocaleString('en-US')} tris
                </span>
              </div>
              <input
                type="range"
                min={200}
                max={200000}
                step={500}
                value={targetFaces}
                onChange={(e) => setTargetFaces(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-0.5">
                <span>200 (Low-Poly)</span>
                <span>50,000 (Mịn cao)</span>
                <span>200,000 (Tối đa)</span>
              </div>
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
                max={1024}
                step={4}
                value={paletteColors}
                onChange={(e) => setPaletteColors(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-0.5">
                <span>8 màu (Flat)</span>
                <span>256 màu (Mịn)</span>
                <span>1,024 màu (Tối đa)</span>
              </div>
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
                title={`Nạp lại mesh "${activeAsset.name}" với số tris (${targetFaces}) & màu (${paletteColors}) mới (~1s)`}
                className="w-full py-2 px-3 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 text-emerald-300 hover:text-white text-xs font-semibold flex items-center justify-center gap-2 shadow-sm transition group"
              >
                <RefreshCw className={`w-3.5 h-3.5 text-emerald-400 group-hover:rotate-180 transition-transform duration-500 ${status.busy ? 'animate-spin' : ''}`} />
                <span>Nạp lại cấu hình</span>
                <span className="text-[10px] text-emerald-400/80 font-mono font-normal">
                  (~1s • {activeAsset.name})
                </span>
              </button>
            )}
              </>
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

import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { ContactShadows, Environment, Grid, Html, Lightformer, OrbitControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import {
  Camera,
  Eye,
  Grid3x3,
  Lightbulb,
  Maximize,
  Minimize,
  Palette,
  RotateCcw,
  RotateCw,
  Box,
  Layers,
} from 'lucide-react'

export interface ShowcaseProps {
  url: string
  name: string
  stats: { final_faces: number; palette_colors: number; size_m: [number, number, number] }
  wireframe: boolean
  setWireframe: (v: boolean) => void
  autoRotate: boolean
  setAutoRotate: (v: boolean) => void
  showTurnaround: boolean
  setShowTurnaround: (v: boolean) => void
}

const BACKGROUNDS = [
  { key: 'studio', label: 'Studio tối', color: '#0b1020', floor: '#000000' },
  { key: 'graphite', label: 'Xám than', color: '#242933', floor: '#000000' },
  { key: 'light', label: 'Sáng', color: '#e9edf4', floor: '#6b7280' },
  { key: 'warm', label: 'Kem ấm', color: '#f1e8dc', floor: '#7c6a55' },
  { key: 'green', label: 'Green screen', color: '#00b140', floor: '#003d17' },
]

const LIGHTINGS = [
  { key: 'studio', label: 'Studio', env: 1.0, key_i: 2.2, rim_i: 2.0 },
  { key: 'soft', label: 'Dịu', env: 1.35, key_i: 1.2, rim_i: 0.8 },
  { key: 'dramatic', label: 'Kịch tính', env: 0.45, key_i: 4.0, rim_i: 4.0 },
]

type FitInfo = { size: THREE.Vector3 }

function Model({
  url,
  wireframe,
  envIntensity,
  onFit,
}: {
  url: string
  wireframe: boolean
  envIntensity: number
  onFit: (f: FitInfo) => void
}) {
  const { scene } = useGLTF(url)

  const { obj, size } = useMemo(() => {
    const c = scene.clone(true)
    c.traverse((o: any) => {
      if (o.isMesh && o.material) {
        o.material = o.material.clone()
        if (o.material.map) {
          // palette cells must stay crisp (no colour bleeding between swatches)
          o.material.map.magFilter = THREE.NearestFilter
          o.material.map.minFilter = THREE.NearestFilter
          o.material.map.generateMipmaps = false
          o.material.map.needsUpdate = true
        }
      }
    })
    const box = new THREE.Box3().setFromObject(c)
    const center = box.getCenter(new THREE.Vector3())
    const sz = box.getSize(new THREE.Vector3())
    // centre on X/Z and sit exactly on the floor
    c.position.set(-center.x, -box.min.y, -center.z)
    return { obj: c, size: sz }
  }, [scene])

  useEffect(() => {
    obj.traverse((o: any) => {
      if (o.isMesh && o.material) {
        o.material.wireframe = wireframe
        o.material.envMapIntensity = envIntensity
        o.material.needsUpdate = true
      }
    })
  }, [obj, wireframe, envIntensity])

  useEffect(() => {
    onFit({ size })
  }, [size])

  return <primitive object={obj} />
}

function CameraRig({ fit, resetKey, controlsRef }: { fit: FitInfo | null; resetKey: number; controlsRef: React.MutableRefObject<any> }) {
  const { camera } = useThree()
  useEffect(() => {
    if (!fit) return
    const maxDim = Math.max(fit.size.x, fit.size.y, fit.size.z, 0.01)
    const fov = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180
    const dist = (maxDim / 2 / Math.tan(fov / 2)) * 1.9
    camera.position.set(dist * 0.62, fit.size.y * 0.62 + dist * 0.18, dist * 0.78)
    camera.near = Math.max(dist / 200, 0.001)
    camera.far = dist * 80
    camera.updateProjectionMatrix()
    const ctl = controlsRef.current
    if (ctl) {
      ctl.target.set(0, fit.size.y * 0.5, 0)
      ctl.minDistance = dist * 0.25
      ctl.maxDistance = dist * 4
      ctl.update()
    }
  }, [fit, resetKey])
  return null
}

function GLBridge({ bridge }: { bridge: React.MutableRefObject<any> }) {
  const { gl, scene, camera } = useThree()
  useEffect(() => {
    bridge.current = { gl, scene, camera }
  }, [gl, scene, camera])
  return null
}

export default function Showcase(p: ShowcaseProps) {
  const [bgKey, setBgKey] = useState('studio')
  const [lightKey, setLightKey] = useState('studio')
  const [showGrid, setShowGrid] = useState(true)
  const [showHud, setShowHud] = useState(true)
  const [fit, setFit] = useState<FitInfo | null>(null)
  const [resetKey, setResetKey] = useState(0)
  const [isFull, setIsFull] = useState(false)
  const [bgOpen, setBgOpen] = useState(false)
  const [lightOpen, setLightOpen] = useState(false)

  const wrapRef = useRef<HTMLDivElement>(null)
  const controlsRef = useRef<any>(null)
  const bridge = useRef<any>(null)

  const bg = BACKGROUNDS.find((b) => b.key === bgKey) || BACKGROUNDS[0]
  const lt = LIGHTINGS.find((l) => l.key === lightKey) || LIGHTINGS[0]
  const darkBg = bg.key === 'studio' || bg.key === 'graphite'

  useEffect(() => {
    const h = () => setIsFull(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', h)
    return () => document.removeEventListener('fullscreenchange', h)
  }, [])

  function toggleFull() {
    if (document.fullscreenElement) document.exitFullscreen()
    else wrapRef.current?.requestFullscreen?.()
  }

  function screenshot() {
    const b = bridge.current
    if (!b) return
    b.gl.render(b.scene, b.camera)
    const a = document.createElement('a')
    a.href = b.gl.domElement.toDataURL('image/png')
    a.download = `${p.name || 'model'}_preview.png`
    a.click()
  }

  const btn = (active: boolean) =>
    `flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition ${
      active ? 'bg-emerald-500 text-white shadow shadow-emerald-500/30' : 'text-slate-300 hover:bg-white/10'
    }`

  return (
    <div ref={wrapRef} className="absolute inset-0" style={{ background: bg.color }}>
      <Canvas
        camera={{ position: [2, 1.6, 2.4], fov: 38 }}
        dpr={[1, 2]}
        gl={{ antialias: true, preserveDrawingBuffer: true, toneMapping: THREE.NeutralToneMapping }}
      >
        <color attach="background" args={[bg.color]} />
        <GLBridge bridge={bridge} />

        {/* Procedural studio lighting (works offline - no HDRI download) */}
        <Environment resolution={256} frames={1}>
          <group rotation={[0, 0.6, 0]}>
            <Lightformer form="rect" intensity={lt.key_i} position={[4, 5, 3]} scale={[6, 4, 1]} target={[0, 0, 0]} color="#ffffff" />
            <Lightformer form="rect" intensity={lt.key_i * 0.45} position={[-5, 2, 2]} scale={[5, 5, 1]} target={[0, 0, 0]} color="#dbe8ff" />
            <Lightformer form="rect" intensity={lt.rim_i} position={[0, 3, -6]} scale={[8, 3, 1]} target={[0, 0, 0]} color="#ffe9d2" />
            <Lightformer form="ring" intensity={lt.env} position={[0, 7, 0]} scale={6} rotation-x={Math.PI / 2} color="#ffffff" />
          </group>
        </Environment>
        <ambientLight intensity={lt.env * 0.35} />
        <directionalLight position={[3, 6, 4]} intensity={lt.key_i * 0.35} />

        {showGrid && (
          <Grid
            infiniteGrid
            fadeDistance={14}
            fadeStrength={2}
            cellSize={0.25}
            sectionSize={1}
            cellThickness={0.6}
            sectionThickness={1.1}
            cellColor={darkBg ? '#1c2540' : '#c4cad6'}
            sectionColor={darkBg ? '#2c3a63' : '#9aa3b5'}
            position={[0, -0.001, 0]}
          />
        )}

        <Suspense
          fallback={
            <Html center>
              <div className="px-4 py-2 rounded-lg bg-slate-900/90 border border-slate-700 text-xs text-emerald-400 whitespace-nowrap">
                Đang nạp model 3D...
              </div>
            </Html>
          }
        >
          <Model url={p.url} wireframe={p.wireframe} envIntensity={lt.env} onFit={setFit} />
          <ContactShadows
            position={[0, 0, 0]}
            opacity={darkBg ? 0.7 : 0.45}
            scale={Math.max(6, (fit ? Math.max(fit.size.x, fit.size.z) : 1) * 5)}
            blur={2.6}
            far={Math.max(1.5, fit ? fit.size.y * 1.4 : 1.5)}
            color={bg.floor}
          />
        </Suspense>

        <CameraRig fit={fit} resetKey={resetKey} controlsRef={controlsRef} />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          autoRotate={p.autoRotate}
          autoRotateSpeed={1.6}
          maxPolarAngle={Math.PI * 0.495}
        />
      </Canvas>

      {/* Toolbar (left) */}
      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-1 bg-slate-950/70 backdrop-blur-md border border-white/10 rounded-xl p-1">
        <button onClick={() => p.setAutoRotate(!p.autoRotate)} className={btn(p.autoRotate)} title="Tự xoay">
          <RotateCw className="w-3.5 h-3.5" /> Tự xoay
        </button>
        <button onClick={() => p.setWireframe(!p.wireframe)} className={btn(p.wireframe)} title="Hiển thị lưới tam giác">
          <Layers className="w-3.5 h-3.5" /> Lưới
        </button>
        <button onClick={() => setShowGrid(!showGrid)} className={btn(showGrid)} title="Lưới nền">
          <Grid3x3 className="w-3.5 h-3.5" /> Sàn
        </button>
        <button onClick={() => p.setShowTurnaround(!p.showTurnaround)} className={btn(p.showTurnaround)} title="Ảnh 4 góc & Palette">
          <Eye className="w-3.5 h-3.5" /> 4 góc
        </button>

        <span className="w-px h-5 bg-white/10 mx-0.5" />

        {/* Lighting */}
        <div className="relative">
          <button
            onClick={() => {
              setLightOpen(!lightOpen)
              setBgOpen(false)
            }}
            className={btn(lightOpen)}
            title="Ánh sáng"
          >
            <Lightbulb className="w-3.5 h-3.5 text-amber-300" /> {lt.label}
          </button>
          {lightOpen && (
            <div className="absolute top-full left-0 mt-1.5 w-36 bg-slate-900/95 border border-slate-700 rounded-xl p-1 shadow-2xl z-30">
              {LIGHTINGS.map((l) => (
                <button
                  key={l.key}
                  onClick={() => {
                    setLightKey(l.key)
                    setLightOpen(false)
                  }}
                  className={`w-full px-2.5 py-1.5 rounded-lg text-xs text-left transition ${
                    lightKey === l.key ? 'bg-emerald-500/15 text-emerald-300' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Background */}
        <div className="relative">
          <button onClick={() => { setBgOpen(!bgOpen); setLightOpen(false) }} className={btn(bgOpen)} title="Màu nền">
            <Palette className="w-3.5 h-3.5" />
            <span
              className="w-3 h-3 rounded-full border border-white/40"
              style={{ background: bg.color }}
            />
            Nền
          </button>
          {bgOpen && (
            <div className="absolute top-full left-0 mt-1.5 w-40 bg-slate-900/95 border border-slate-700 rounded-xl p-1 shadow-2xl z-30">
              {BACKGROUNDS.map((b) => (
                <button
                  key={b.key}
                  onClick={() => {
                    setBgKey(b.key)
                    setBgOpen(false)
                  }}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition ${
                    bgKey === b.key ? 'bg-emerald-500/15 text-emerald-300' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <span className="w-3.5 h-3.5 rounded-full border border-white/30" style={{ background: b.color }} />
                  {b.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Actions (right) */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-slate-950/70 backdrop-blur-md border border-white/10 rounded-xl p-1">
        <button onClick={() => setResetKey((k) => k + 1)} className={btn(false)} title="Về góc nhìn mặc định">
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <button onClick={screenshot} className={btn(false)} title="Chụp ảnh PNG">
          <Camera className="w-3.5 h-3.5" /> Chụp ảnh
        </button>
        <button onClick={() => setShowHud(!showHud)} className={btn(showHud)} title="Ẩn/hiện thông tin (chế độ quay video)">
          <Box className="w-3.5 h-3.5" />
        </button>
        <button onClick={toggleFull} className={btn(isFull)} title="Toàn màn hình">
          {isFull ? <Minimize className="w-3.5 h-3.5" /> : <Maximize className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* HUD: model info */}
      {showHud && (
        <div className="absolute bottom-3 left-3 z-10 bg-slate-950/70 backdrop-blur-md border border-white/10 rounded-xl px-4 py-2.5 min-w-[220px]">
          <div className="text-sm font-bold text-white truncate max-w-[300px]">{p.name}</div>
          <div className="mt-1 grid grid-cols-3 gap-4 text-[11px]">
            <div>
              <div className="text-slate-400">Tam giác</div>
              <div className="font-mono font-semibold text-emerald-400">{p.stats.final_faces.toLocaleString('en-US')}</div>
            </div>
            <div>
              <div className="text-slate-400">Màu</div>
              <div className="font-mono font-semibold text-emerald-400">{p.stats.palette_colors}</div>
            </div>
            <div>
              <div className="text-slate-400">Kích thước (m)</div>
              <div className="font-mono font-semibold text-slate-200">{p.stats.size_m.join(' × ')}</div>
            </div>
          </div>
        </div>
      )}

      {/* Brand watermark */}
      <div className="absolute bottom-3 right-3 z-10 pointer-events-none text-right select-none">
        <div className={`text-sm font-extrabold tracking-[0.18em] ${darkBg ? 'text-white/80' : 'text-slate-800/80'}`}>
          AURASOFT
        </div>
        <div className={`text-[9px] tracking-[0.35em] -mt-0.5 ${darkBg ? 'text-emerald-300/80' : 'text-emerald-700/80'}`}>
          SYSTEMS
        </div>
      </div>
    </div>
  )
}

const { app, BrowserWindow } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const http = require('http')

let backendProc = null

function checkServerReady(url) {
  return new Promise((resolve) => {
    http
      .get(url, (res) => {
        resolve(res.statusCode === 200)
      })
      .on('error', () => resolve(false))
  })
}

async function startBackend() {
  const ready = await checkServerReady('http://127.0.0.1:8765/api/presets')
  if (ready) return

  const rootDir = path.resolve(__dirname, '..', '..')
  const pyExe = path.join(rootDir, '.venv', 'Scripts', 'python.exe')
  const serverScript = path.join(rootDir, 'backend', 'server.py')

  backendProc = spawn(pyExe, [serverScript], {
    cwd: rootDir,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    stdio: 'ignore',
    windowsHide: true,
  })

  for (let i = 0; i < 40; i++) {
    const ok = await checkServerReady('http://127.0.0.1:8765/api/presets')
    if (ok) break
    await new Promise((r) => setTimeout(r, 500))
  }
}

async function createWindow() {
  await startBackend()

  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    title: 'Mesh Studio – Tạo Model 3D Game Nông Trại (Unity)',
    backgroundColor: '#020617',
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  win.webContents.on('console-message', (event, level, message, line, sourceId) => {
    if (level >= 2) {
      console.error(`[Renderer Error] ${message} (${sourceId}:${line})`)
    }
  })

  const devUrl = process.env.VITE_DEV_SERVER_URL
  if (devUrl) {
    win.loadURL(devUrl)
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  }
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (backendProc) {
    try {
      backendProc.kill()
    } catch {}
  }
  app.quit()
})

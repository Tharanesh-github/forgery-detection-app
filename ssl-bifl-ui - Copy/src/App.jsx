import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL

export default function App() {
  const [file, setFile]         = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [activeView, setView]   = useState('heatmap')
  const [condition, setCondition] = useState('clean')
  const [threshold, setThreshold] = useState(0.5)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) handleFile(f)
  }

  async function analyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('threshold', threshold)
      form.append('condition', condition)
      const { data } = await axios.post(`${API}/analyze`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000
      })
      setResult(data)
      setView('heatmap')
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The Space may be waking up — wait 60 seconds and try again.')
      } else {
        setError('Analysis failed. Check that the backend is running at ' + API)
      }
    } finally {
      setLoading(false)
    }
  }

  const viewMap = {
    original: result?.original_b64,
    heatmap:  result?.heatmap_b64,
    mask:     result?.mask_b64,
    overlay:  result?.overlay_b64,
  }

  return (
    <div className="app">
      {/* Scanline overlay */}
      <div className="scanlines" />

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <div>
              <div className="logo-title">SSL-BIFL</div>
              <div className="logo-sub">Forgery Localization System</div>
            </div>
          </div>
        </div>
        <div className="header-right">
          <span className="badge">ResNet-18 + U-Net</span>
          <span className="badge badge-green">
            <span className="pulse" /> ONLINE
          </span>
        </div>
      </header>

      <main className="main">
        {/* LEFT PANEL */}
        <section className="panel left-panel">

          {/* Section label */}
          <div className="section-label">01 — INPUT IMAGE</div>

          {/* Drop zone */}
          <div
            className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !preview && inputRef.current.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={e => handleFile(e.target.files[0])}
            />
            {preview ? (
              <img src={preview} alt="Preview" className="preview-img" />
            ) : (
              <div className="dropzone-placeholder">
                <div className="drop-icon">⬡</div>
                <div className="drop-title">Drop image here</div>
                <div className="drop-sub">or <span onClick={() => inputRef.current.click()}>browse files</span></div>
                <div className="drop-formats">JPG · PNG · TIF</div>
              </div>
            )}
          </div>

          {preview && (
            <button className="btn-ghost btn-sm" onClick={() => { setFile(null); setPreview(null); setResult(null) }}>
              ✕ Remove image
            </button>
          )}

          {/* Stress test */}
          <div className="section-label" style={{ marginTop: '1.5rem' }}>02 — STRESS TEST</div>
          <div className="toggle-row">
            {['clean', 'noise', 'jpeg'].map(c => (
              <button
                key={c}
                className={`toggle-btn ${condition === c ? 'active' : ''}`}
                onClick={() => setCondition(c)}
              >
                {c === 'clean' ? 'Clean' : c === 'noise' ? '+ Noise' : 'JPEG Q50'}
              </button>
            ))}
          </div>

          {/* Threshold */}
          <div className="section-label" style={{ marginTop: '1.5rem' }}>03 — THRESHOLD</div>
          <div className="slider-row">
            <span className="slider-label">0.20</span>
            <input
              type="range" min="20" max="70" step="1"
              value={Math.round(threshold * 100)}
              onChange={e => setThreshold(e.target.value / 100)}
              className="slider"
            />
            <span className="slider-label">0.70</span>
            <span className="slider-val">{threshold.toFixed(2)}</span>
          </div>
          <div className="hint">Dynamic scanning enabled — override operating point</div>

          {/* Analyze button */}
          <button
            className={`btn-analyze ${loading ? 'loading' : ''}`}
            disabled={!file || loading}
            onClick={analyze}
          >
            {loading ? (
              <><span className="spinner" /> SCANNING PIXELS...</>
            ) : (
              '⬡ ANALYZE FOR FORGERY'
            )}
          </button>

          {error && <div className="error-box">{error}</div>}

          {/* Model info */}
          <div className="section-label" style={{ marginTop: '1.5rem' }}>04 — MODEL INFO</div>
          <div className="info-grid">
            <div className="info-card"><div className="info-label">Architecture</div><div className="info-val">ResNet-18 + U-Net</div></div>
            <div className="info-card"><div className="info-label">Training Data</div><div className="info-val">DIV2K 800 imgs</div></div>
            <div className="info-card"><div className="info-label">Loss Function</div><div className="info-val">BCE + Dice</div></div>
            <div className="info-card"><div className="info-label">Specificity</div><div className="info-val spec">&gt; 95%</div></div>
          </div>
        </section>

        {/* RIGHT PANEL */}
        <section className="panel right-panel">

          {/* Output view */}
          <div className="output-header">
            <div className="section-label">05 — DETECTION OUTPUT</div>
            {result && (
              <div className="view-tabs">
                {['heatmap', 'mask', 'overlay', 'original'].map(v => (
                  <button
                    key={v}
                    className={`view-tab ${activeView === v ? 'active' : ''}`}
                    onClick={() => setView(v)}
                  >
                    {v}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className={`output-box ${loading ? 'scanning' : ''}`}>
            {loading && (
              <div className="scan-overlay">
                <div className="scan-bar" />
                <div className="scan-text">ANALYZING PIXELS...</div>
              </div>
            )}
            {result ? (
              <img
                src={`data:image/png;base64,${viewMap[activeView]}`}
                alt={activeView}
                className="result-img"
              />
            ) : (
              <div className="output-placeholder">
                <div className="placeholder-icon">⬡</div>
                <div className="placeholder-text">Awaiting analysis</div>
                <div className="placeholder-sub">Upload an image to begin</div>
              </div>
            )}
          </div>

          {/* Metrics */}
          <div className="section-label" style={{ marginTop: '1.5rem' }}>06 — METRICS</div>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className={`metric-val ${result ? (result.forged_percentage > 5 ? 'danger' : 'safe') : 'neutral'}`}>
                {result ? `${result.forged_percentage.toFixed(1)}%` : '—'}
              </div>
              <div className="metric-label">Forged Area</div>
            </div>
            <div className="metric-card">
              <div className={`metric-val ${result ? 'accent' : 'neutral'}`}>
                {result ? result.best_threshold.toFixed(2) : '—'}
              </div>
              <div className="metric-label">Best Threshold</div>
            </div>
            <div className="metric-card">
              <div className={`metric-val ${result ? (result.confidence > 60 ? 'safe' : 'warn') : 'neutral'}`}>
                {result ? `${result.confidence}%` : '—'}
              </div>
              <div className="metric-label">Confidence</div>
            </div>
          </div>

          {/* Verdict */}
          <div className="section-label" style={{ marginTop: '1.5rem' }}>07 — VERDICT</div>
          <div className={`verdict ${
  !result ? 'pending'
  : result.verdict_level === 'FORGED' ? 'forged'
  : result.verdict_level === 'SUSPICIOUS' ? 'forged'
  : result.verdict_level === 'INCONCLUSIVE' ? 'pending'
  : 'authentic'
}`}>
            <div className={`verdict-dot ${!result ? 'pending' : result.is_forged ? 'forged' : 'authentic'}`} />
            <div className="verdict-content">
              <div className="verdict-title">
  {!result
    ? 'Upload an image to begin analysis'
    : result.verdict_level === 'FORGED'
    ? '⚠ FORGERY DETECTED'
    : result.verdict_level === 'SUSPICIOUS'
    ? '⚠ SUSPICIOUS — Possible Forgery'
    : result.verdict_level === 'INCONCLUSIVE'
    ? '◈ INCONCLUSIVE — Weak Signal'
    : '✓ AUTHENTIC'}
</div>
{result && (
  <div className="verdict-sub">
    {result.verdict_message}
  </div>
)}
            </div>
          </div>

          {/* Student attribution */}
          <div className="attribution">
            V. Tharanesh · Final Year Project · SSL-BIFL · CASIA 2.0 Tested
          </div>
        </section>
      </main>
    </div>
  )
}
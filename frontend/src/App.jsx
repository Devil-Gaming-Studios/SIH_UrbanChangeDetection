import { useEffect, useState } from 'react'

function ImageCard({ src, caption, onOpen }) {
  if (!src) return null
  return (
    <figure className="image-card" onClick={() => onOpen(src, caption)}>
      <img src={src} alt={caption} loading="lazy" />
      <figcaption>{caption}</figcaption>
    </figure>
  )
}

function Section({ title, meta, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="card collapsible">
      <div className="card-header" onClick={() => setOpen((o) => !o)}>
        <div className="card-header-left">
          <span className={`chevron ${open ? 'open' : ''}`}>▶</span>
          <span className="card-header-title">{title}</span>
        </div>
        {meta && <span className="card-header-meta">{meta}</span>}
      </div>
      {open && <div className="card-body">{children}</div>}
    </div>
  )
}

function Lightbox({ image, onClose }) {
  if (!image) return null

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        <img src={image.src} alt={image.caption} />
        <div className="lightbox-caption">{image.caption}</div>
      </div>
    </div>
  )
}

function HistoryPanel({ onSelect, refreshKey }) {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    fetch('/api/history')
      .then((res) => {
        if (!res.ok) throw new Error('Could not load history.')
        return res.json()
      })
      .then((data) => {
        if (!cancelled) setRuns(data.runs || [])
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  async function handleDelete(e, runId) {
    e.stopPropagation()
    await fetch(`/api/history/${runId}`, { method: 'DELETE' }).catch(() => {})
    setRuns((prev) => prev.filter((r) => r.run_id !== runId))
  }

  if (loading) {
    return <p className="muted">Loading history…</p>
  }

  if (error) {
    return <p className="error">{error}</p>
  }

  if (runs.length === 0) {
    return <p className="muted">No past runs yet.</p>
  }

  return (
    <div className="image-grid">
      {runs.map((r) => (
        <figure
          key={r.run_id}
          className="image-card history-card"
          onClick={() => onSelect(r.run_id)}
        >
          {r.thumbnail_url ? (
            <img src={r.thumbnail_url} alt={r.run_id} loading="lazy" />
          ) : (
            <div className="history-placeholder" />
          )}

          <figcaption>
            <div className="history-row">
              <span>
                {r.year_earlier} → {r.year_later}
              </span>
              <button
                type="button"
                className="delete-button"
                onClick={(e) => handleDelete(e, r.run_id)}
                title="Delete run"
              >
                ✕
              </button>
            </div>

            {r.changed_percentage !== undefined && (
              <span className="badge">
                {Number(r.changed_percentage).toFixed(1)}% changed
              </span>
            )}

            <span className="badge">
              {r.mode ? r.mode.toUpperCase() : 'ANALYSIS'}
            </span>

            {r.created_at && (
              <span className="history-date">
                {new Date(r.created_at).toLocaleString()}
              </span>
            )}
          </figcaption>
        </figure>
      ))}
    </div>
  )
}

function FilePicker({ label, file, accept, onChange, hint }) {
  return (
    <div className="file-field">
      <label>{label}</label>
      <input type="file" accept={accept} onChange={onChange} />
      {file ? (
        <div className="selected-file">
          <span className="file-check">✓</span>
          <span title={file.name}>{file.name}</span>
        </div>
      ) : (
        <p className="field-hint">{hint}</p>
      )}
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('theme') || 'light'
  )

  const [activeTab, setActiveTab] = useState('analyze')

  // The REST API uses this value to select the trained checkpoint:
  // rgb -> checkpoint_final.pth
  // msi -> checkpoint_final (1).pth
  const [mode, setMode] = useState('rgb')

  const [earlierFile, setEarlierFile] = useState(null)
  const [laterFile, setLaterFile] = useState(null)
  const [msiZip, setMsiZip] = useState(null)

  const [yearEarlier, setYearEarlier] = useState(2018)
  const [yearLater, setYearLater] = useState(2024)
  const [threshold, setThreshold] = useState(0.5)
  const [pixelResolution, setPixelResolution] = useState('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [lightboxImage, setLightboxImage] = useState(null)
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  function changeMode(nextMode) {
    setMode(nextMode)
    setError(null)
    setResult(null)

    // Prevent accidentally submitting files from the other model mode.
    if (nextMode === 'rgb') {
      setMsiZip(null)
    } else {
      setEarlierFile(null)
      setLaterFile(null)
    }
  }

  function openLightbox(src, caption) {
    setLightboxImage({ src, caption })
  }

  function getFileUrl(path) {
    if (!path) return null
    if (path.startsWith('http://') || path.startsWith('https://')) return path
    return path
  }

  async function loadRunFromHistory(runId) {
    setError(null)

    try {
      const res = await fetch(`/api/history/${runId}`)
      if (!res.ok) throw new Error('Could not load that run.')

      const data = await res.json()

      setResult(data)

      if (data.year_earlier !== undefined) {
        setYearEarlier(data.year_earlier)
      }

      if (data.year_later !== undefined) {
        setYearLater(data.year_later)
      }

      if (data.mode === 'rgb' || data.mode === 'msi') {
        setMode(data.mode)
      }

      setActiveTab('analyze')
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setResult(null)

    if (yearLater <= yearEarlier) {
      setError('Later year must be greater than earlier year.')
      return
    }

    if (mode === 'rgb' && (!earlierFile || !laterFile)) {
      setError('Please upload both the earlier and later RGB images.')
      return
    }

    if (mode === 'msi' && !msiZip) {
      setError('Please upload the MSI ZIP containing both dates.')
      return
    }

    const formData = new FormData()

    // This field controls checkpoint selection on the FastAPI backend.
    formData.append('mode', mode)
    formData.append('year_earlier', yearEarlier)
    formData.append('year_later', yearLater)
    formData.append('threshold', threshold)

    if (pixelResolution) {
      formData.append('pixel_resolution_m', pixelResolution)
    }

    if (mode === 'rgb') {
      formData.append('image_earlier', earlierFile)
      formData.append('image_later', laterFile)
    } else {
      formData.append('msi_zip', msiZip)
    }

    setLoading(true)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(
          detail.detail || `Request failed (${res.status})`
        )
      }

      const data = await res.json()

      setResult(data)
      setHistoryRefreshKey((k) => k + 1)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const activeCheckpoint =
    mode === 'rgb'
      ? 'checkpoint_final.pth'
      : 'checkpoint_final (1).pth'

  return (
    <div className="container">
      <div className="topbar">
        <div>
          <h1>Urban Change Detection</h1>
          <p className="subtitle">
            Compare the same area at two different times using either
            the RGB model or the full Sentinel-2 MSI model.
          </p>
        </div>

        <button
          type="button"
          className="theme-toggle"
          onClick={() =>
            setTheme((t) => (t === 'light' ? 'dark' : 'light'))
          }
        >
          <span>{theme === 'light' ? 'Dark mode' : 'Light mode'}</span>
          <span className="icon">
            {theme === 'light' ? '🌙' : '☀️'}
          </span>
        </button>
      </div>

      <div className="tabs">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'analyze' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyze')}
        >
          Analyze
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          History
        </button>
      </div>

      {activeTab === 'history' && (
        <Section title="Past runs">
          <HistoryPanel
            onSelect={loadRunFromHistory}
            refreshKey={historyRefreshKey}
          />
        </Section>
      )}

      {activeTab === 'analyze' && (
        <>
          <form className="card" onSubmit={handleSubmit}>
            <div className="model-selector">
              <div>
                <div className="section-title">Input / model</div>
                <p className="muted model-description">
                  Select the type of imagery you are providing. The
                  backend automatically loads the matching trained
                  checkpoint.
                </p>
              </div>

              <div className="mode-buttons">
                <button
                  type="button"
                  className={`mode-button ${mode === 'rgb' ? 'selected' : ''}`}
                  onClick={() => changeMode('rgb')}
                >
                  <span className="mode-title">RGB images</span>
                  <span className="mode-subtitle">
                    2 JPG / PNG images
                  </span>
                </button>

                <button
                  type="button"
                  className={`mode-button ${mode === 'msi' ? 'selected' : ''}`}
                  onClick={() => changeMode('msi')}
                >
                  <span className="mode-title">Sentinel-2 MSI</span>
                  <span className="mode-subtitle">
                    1 ZIP · 13 bands × 2 dates
                  </span>
                </button>
              </div>

              <div className="checkpoint-badge">
                <span>Selected model</span>
                <strong>{activeCheckpoint}</strong>
              </div>
            </div>

            {mode === 'rgb' ? (
              <div className="grid-2">
                <FilePicker
                  label="Earlier image"
                  file={earlierFile}
                  accept="image/png,image/jpeg,.jpg,.jpeg,.png"
                  onChange={(e) =>
                    setEarlierFile(e.target.files?.[0] || null)
                  }
                  hint="JPG or PNG — the earlier date"
                />

                <FilePicker
                  label="Later image"
                  file={laterFile}
                  accept="image/png,image/jpeg,.jpg,.jpeg,.png"
                  onChange={(e) =>
                    setLaterFile(e.target.files?.[0] || null)
                  }
                  hint="JPG or PNG — the later date"
                />
              </div>
            ) : (
              <div className="msi-upload-box">
                <FilePicker
                  label="Sentinel-2 MSI ZIP"
                  file={msiZip}
                  accept=".zip,application/zip"
                  onChange={(e) =>
                    setMsiZip(e.target.files?.[0] || null)
                  }
                  hint="One ZIP containing an earlier and later folder, each with B01–B12 + B8A"
                />

                <div className="msi-structure">
                  <div className="structure-title">
                    Expected ZIP structure
                  </div>
                  <code>
                    earlier/
                    <br />
                    &nbsp;&nbsp;B01.tif … B12.tif + B8A.tif
                    <br />
                    later/
                    <br />
                    &nbsp;&nbsp;B01.tif … B12.tif + B8A.tif
                  </code>
                </div>
              </div>
            )}

            <div className="grid-2 metadata-grid">
              <div>
                <label>Earlier year</label>
                <input
                  type="number"
                  value={yearEarlier}
                  onChange={(e) =>
                    setYearEarlier(Number(e.target.value))
                  }
                />
              </div>

              <div>
                <label>Later year</label>
                <input
                  type="number"
                  value={yearLater}
                  onChange={(e) =>
                    setYearLater(Number(e.target.value))
                  }
                />
              </div>

              <div>
                <label>Change threshold (0–1)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={threshold}
                  onChange={(e) =>
                    setThreshold(Number(e.target.value))
                  }
                />
              </div>

              <div>
                <label>Pixel resolution (m/pixel, optional)</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 10 for Sentinel-2"
                  value={pixelResolution}
                  onChange={(e) =>
                    setPixelResolution(e.target.value)
                  }
                />
              </div>
            </div>

            <div className="submit-row">
              <button type="submit" disabled={loading}>
                {loading
                  ? 'Analyzing…'
                  : mode === 'rgb'
                    ? 'Analyze RGB images'
                    : 'Analyze MSI ZIP'}
              </button>

              {mode === 'rgb' ? (
                <span className="submit-note">
                  Uses the RGB-trained checkpoint
                </span>
              ) : (
                <span className="submit-note">
                  Uses the 13-channel Sentinel-2 checkpoint
                </span>
              )}
            </div>

            {error && <div className="error">{error}</div>}
          </form>
        </>
      )}

      {activeTab === 'analyze' && result && (
        <>
          <Section
            title="Executive summary"
            meta={`${result.year_earlier ?? yearEarlier} → ${result.year_later ?? yearLater}`}
          >
            <div className="result-model">
              <span className="badge">
                {(result.mode || mode).toUpperCase()}
              </span>
              {result.checkpoint && (
                <span className="badge">{result.checkpoint}</span>
              )}
            </div>

            <p className="exec-summary">
              {result.stats?.executive_summary}
            </p>
          </Section>

          {result.stats && (
            <Section title="Growth summary">
              <div className="stats-grid">
                {result.stats.changed_percentage !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {Number(result.stats.changed_percentage).toFixed(2)}%
                    </div>
                    <div className="label">New / changed area</div>
                  </div>
                )}

                {result.stats.annual_growth_rate_percentage !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {Number(
                        result.stats.annual_growth_rate_percentage
                      ).toFixed(2)}
                      %/yr
                    </div>
                    <div className="label">
                      Estimated growth rate
                    </div>
                  </div>
                )}

                {result.stats.changed_pixels !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {Number(
                        result.stats.changed_pixels
                      ).toLocaleString()}
                    </div>
                    <div className="label">Changed pixels</div>
                  </div>
                )}

                {result.stats.changed_area_km2 !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {Number(result.stats.changed_area_km2).toFixed(3)} km²
                    </div>
                    <div className="label">
                      Changed area (
                      {Number(
                        result.stats.changed_area_hectares
                      ).toFixed(1)}{' '}
                      ha)
                    </div>
                  </div>
                )}

                {result.stats.mean_confidence_percentage !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {Number(
                        result.stats.mean_confidence_percentage
                      ).toFixed(1)}
                      %
                    </div>
                    <div className="label">
                      Mean data confidence
                    </div>
                  </div>
                )}

                {result.stats.year_gap !== undefined && (
                  <div className="stat">
                    <div className="value">
                      {result.stats.year_gap}yr
                    </div>
                    <div className="label">Time span analyzed</div>
                  </div>
                )}
              </div>

              <ImageCard
                src={getFileUrl(result.growth_chart)}
                caption="Projected growth"
                onOpen={openLightbox}
              />

              <div>
                {result.report_pdf && (
                  <a
                    className="download-btn"
                    href={getFileUrl(result.report_pdf)}
                    download
                  >
                    <button type="button">
                      Download full PDF report
                    </button>
                  </a>
                )}
              </div>
            </Section>
          )}

          {result.stats?.landuse_breakdown && (
            <Section title="Land-use change breakdown">
              <div className="stats-grid">
                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.landuse_breakdown
                        .built_up_percentage
                    ).toFixed(2)}
                    %
                  </div>
                  <div className="label">
                    Built-up / other change
                  </div>
                </div>

                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.landuse_breakdown
                        .vegetation_loss_percentage
                    ).toFixed(2)}
                    %
                  </div>
                  <div className="label">Vegetation loss</div>
                </div>

                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.landuse_breakdown
                        .water_change_percentage
                    ).toFixed(2)}
                    %
                  </div>
                  <div className="label">Water-body change</div>
                </div>
              </div>

              {result.stats.direction_summary && (
                <p className="direction-note">
                  {result.stats.direction_summary}
                </p>
              )}
            </Section>
          )}

          {result.stats?.classical_metrics && (
            <Section
              title="Classical image-comparison metrics"
              defaultOpen={false}
            >
              <div className="stats-grid">
                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.classical_metrics.mse
                    ).toFixed(5)}
                  </div>
                  <div className="label">MSE</div>
                </div>

                <div className="stat">
                  <div className="value">
                    {result.stats.classical_metrics.psnr_db !== null
                      ? `${Number(
                          result.stats.classical_metrics.psnr_db
                        ).toFixed(2)} dB`
                      : '∞'}
                  </div>
                  <div className="label">PSNR</div>
                </div>

                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.classical_metrics.ssim_score
                    ).toFixed(4)}
                  </div>
                  <div className="label">SSIM</div>
                </div>

                <div className="stat">
                  <div className="value">
                    {Number(
                      result.stats.classical_metrics.histogram_distance
                    ).toFixed(4)}
                  </div>
                  <div className="label">Histogram distance</div>
                </div>
              </div>

              {result.stats.classical_metrics
                .illumination_mismatch_warning && (
                <p className="direction-note warning">
                  ⚠ The two images have noticeably different
                  color/illumination profiles. Interpret change
                  results with this in mind.
                </p>
              )}

              {result.images?.ssim_map && (
                <div className="image-grid image-grid-single">
                  <ImageCard
                    src={getFileUrl(result.images.ssim_map)}
                    caption="Per-pixel SSIM map"
                    onOpen={openLightbox}
                  />
                </div>
              )}
            </Section>
          )}

          <Section title="Before / after">
            <div className="image-grid">
              <ImageCard
                src={getFileUrl(result.images?.before_after)}
                caption={`${result.year_earlier ?? yearEarlier} (left) vs ${result.year_later ?? yearLater} (right)`}
                onOpen={openLightbox}
              />
            </div>
          </Section>

          <Section title="Change hotspots">
            <div className="image-grid">
              <ImageCard
                src={getFileUrl(result.images?.hotspot)}
                caption="Grid-based change density"
                onOpen={openLightbox}
              />
            </div>
          </Section>

          <Section title="Change detection outputs">
            <div className="image-grid">
              <ImageCard
                src={getFileUrl(result.images?.change_mask)}
                caption="Binary change bitmap"
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.change_prob)}
                caption="Change probability heatmap"
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.overlay_on_newer)}
                caption="Change overlaid on newer image"
                onOpen={openLightbox}
              />
              {result.images?.confidence && (
                <ImageCard
                  src={getFileUrl(result.images.confidence)}
                  caption="Data confidence (QA validity)"
                  onOpen={openLightbox}
                />
              )}
            </div>
          </Section>

          <Section title="NDVI (vegetation index)" defaultOpen={false}>
            <div className="image-grid">
              <ImageCard
                src={getFileUrl(result.images?.ndvi_earlier)}
                caption={`NDVI — ${result.year_earlier ?? yearEarlier}`}
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndvi_later)}
                caption={`NDVI — ${result.year_later ?? yearLater}`}
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndvi_change)}
                caption="NDVI change magnitude"
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndvi_overlay_on_newer)}
                caption="NDVI change overlaid on newer image"
                onOpen={openLightbox}
              />
            </div>
          </Section>

          <Section title="NDWI (water index)" defaultOpen={false}>
            <div className="image-grid">
              <ImageCard
                src={getFileUrl(result.images?.ndwi_earlier)}
                caption={`NDWI — ${result.year_earlier ?? yearEarlier}`}
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndwi_later)}
                caption={`NDWI — ${result.year_later ?? yearLater}`}
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndwi_change)}
                caption="NDWI change magnitude"
                onOpen={openLightbox}
              />
              <ImageCard
                src={getFileUrl(result.images?.ndwi_overlay_on_newer)}
                caption="NDWI change overlaid on newer image"
                onOpen={openLightbox}
              />
            </div>
          </Section>
        </>
      )}

      <Lightbox
        image={lightboxImage}
        onClose={() => setLightboxImage(null)}
      />
    </div>
  )
}

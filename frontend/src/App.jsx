import { useState, useEffect } from 'react'

function ImageCard({ src, caption, onOpen }) {
  if (!src) return null;
  return (
    <figure className="image-card" onClick={() => onOpen(src, caption)}>
      <img src={src} alt={caption} loading="lazy" />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function Section({ title, meta, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);
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
  );
}

function Lightbox({ image, onClose }) {
  if (!image) return null;
  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        <img src={image.src} alt={image.caption} />
        <div className="lightbox-caption">{image.caption}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const [earlierFile, setEarlierFile] = useState(null);
  const [laterFile, setLaterFile] = useState(null);
  const [yearEarlier, setYearEarlier] = useState(2018);
  const [yearLater, setYearLater] = useState(2024);
  const [threshold, setThreshold] = useState(0.5);
  const [pixelResolution, setPixelResolution] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [lightboxImage, setLightboxImage] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  function openLightbox(src, caption) {
    setLightboxImage({ src, caption });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!earlierFile || !laterFile) {
      setError('Please upload both images.');
      return;
    }

    const formData = new FormData();
    formData.append('image_earlier', earlierFile);
    formData.append('image_later', laterFile);
    formData.append('year_earlier', yearEarlier);
    formData.append('year_later', yearLater);
    formData.append('threshold', threshold);
    if (pixelResolution) formData.append('pixel_resolution_m', pixelResolution);

    setLoading(true);
    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="topbar">
        <div>
          <h1>Urban Change Detection</h1>
          <p className="subtitle">
            Upload two satellite images of the same area to detect built-up change,
            vegetation (NDVI) and water (NDWI) shifts, and download a full report.
          </p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
        >
          <span>{theme === 'light' ? 'Dark mode' : 'Light mode'}</span>
          <span className="icon">{theme === 'light' ? '🌙' : '☀️'}</span>
        </button>
      </div>

      <form className="card" onSubmit={handleSubmit}>
        <div className="grid-2">
          <div>
            <label>Earlier image</label>
            <input
              type="file"
              accept="image/*,.tif,.tiff,.npy"
              onChange={(e) => setEarlierFile(e.target.files[0])}
            />
          </div>
          <div>
            <label>Later image</label>
            <input
              type="file"
              accept="image/*,.tif,.tiff,.npy"
              onChange={(e) => setLaterFile(e.target.files[0])}
            />
          </div>
          <div>
            <label>Earlier year</label>
            <input
              type="number"
              value={yearEarlier}
              onChange={(e) => setYearEarlier(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Later year</label>
            <input
              type="number"
              value={yearLater}
              onChange={(e) => setYearLater(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Change threshold (0-1)</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Pixel resolution (m/pixel, optional)</label>
            <input
              type="number"
              step="0.1"
              placeholder="e.g. 10 for Sentinel-2"
              value={pixelResolution}
              onChange={(e) => setPixelResolution(e.target.value)}
            />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing…' : 'Run analysis'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}
      </form>

      {result && (
        <>
          <Section title="Executive summary" meta={`${yearEarlier} → ${yearLater}`}>
            <p className="exec-summary">{result.stats.executive_summary}</p>
          </Section>

          <Section title="Growth summary">
            <div className="stats-grid">
              <div className="stat">
                <div className="value">{result.stats.changed_percentage.toFixed(2)}%</div>
                <div className="label">New / changed area</div>
              </div>
              <div className="stat">
                <div className="value">
                  {result.stats.annual_growth_rate_percentage.toFixed(2)}%/yr
                </div>
                <div className="label">Estimated growth rate</div>
              </div>
              <div className="stat">
                <div className="value">{result.stats.changed_pixels.toLocaleString()}</div>
                <div className="label">Changed pixels</div>
              </div>
              {result.stats.changed_area_km2 !== undefined && (
                <div className="stat">
                  <div className="value">{result.stats.changed_area_km2.toFixed(3)} km²</div>
                  <div className="label">Changed area ({result.stats.changed_area_hectares.toFixed(1)} ha)</div>
                </div>
              )}
              {result.stats.mean_confidence_percentage !== undefined && (
                <div className="stat">
                  <div className="value">{result.stats.mean_confidence_percentage.toFixed(1)}%</div>
                  <div className="label">Mean data confidence</div>
                </div>
              )}
              <div className="stat">
                <div className="value">{result.stats.year_gap}yr</div>
                <div className="label">Time span analyzed</div>
              </div>
            </div>
            <ImageCard src={result.growth_chart} caption="Projected growth" onOpen={openLightbox} />
            <div>
              <a className="download-btn" href={result.report_pdf} download>
                <button type="button">Download full PDF report</button>
              </a>
            </div>
          </Section>

          {result.stats.landuse_breakdown && (
            <Section title="Land-use change breakdown">
              <div className="stats-grid">
                <div className="stat">
                  <div className="value">{result.stats.landuse_breakdown.built_up_percentage.toFixed(2)}%</div>
                  <div className="label">Built-up / other change</div>
                </div>
                <div className="stat">
                  <div className="value">{result.stats.landuse_breakdown.vegetation_loss_percentage.toFixed(2)}%</div>
                  <div className="label">Vegetation loss</div>
                </div>
                <div className="stat">
                  <div className="value">{result.stats.landuse_breakdown.water_change_percentage.toFixed(2)}%</div>
                  <div className="label">Water-body change</div>
                </div>
              </div>
              <p className="direction-note">{result.stats.direction_summary}</p>
            </Section>
          )}

          <Section title="Before / after">
            <div className="image-grid">
              <ImageCard
                src={result.images.before_after}
                caption={`${yearEarlier} (left) vs ${yearLater} (right)`}
                onOpen={openLightbox}
              />
            </div>
          </Section>

          <Section title="Change hotspots">
            <div className="image-grid">
              <ImageCard src={result.images.hotspot} caption="Grid-based change density" onOpen={openLightbox} />
            </div>
          </Section>

          <Section title="Change detection outputs">
            <div className="image-grid">
              <ImageCard src={result.images.change_mask} caption="Binary change bitmap" onOpen={openLightbox} />
              <ImageCard src={result.images.change_prob} caption="Change probability heatmap" onOpen={openLightbox} />
              <ImageCard src={result.images.overlay_on_newer} caption="Change overlaid on newer image" onOpen={openLightbox} />
              {result.images.confidence && (
                <ImageCard src={result.images.confidence} caption="Data confidence (QA validity)" onOpen={openLightbox} />
              )}
            </div>
          </Section>

          <Section title="NDVI (vegetation index)" defaultOpen={false}>
            <div className="image-grid">
              <ImageCard src={result.images.ndvi_earlier} caption={`NDVI — ${yearEarlier}`} onOpen={openLightbox} />
              <ImageCard src={result.images.ndvi_later} caption={`NDVI — ${yearLater}`} onOpen={openLightbox} />
              <ImageCard src={result.images.ndvi_change} caption="NDVI change magnitude" onOpen={openLightbox} />
              <ImageCard src={result.images.ndvi_overlay_on_newer} caption="NDVI change overlaid on newer image" onOpen={openLightbox} />
            </div>
          </Section>

          <Section title="NDWI (water index)" defaultOpen={false}>
            <div className="image-grid">
              <ImageCard src={result.images.ndwi_earlier} caption={`NDWI — ${yearEarlier}`} onOpen={openLightbox} />
              <ImageCard src={result.images.ndwi_later} caption={`NDWI — ${yearLater}`} onOpen={openLightbox} />
              <ImageCard src={result.images.ndwi_change} caption="NDWI change magnitude" onOpen={openLightbox} />
              <ImageCard src={result.images.ndwi_overlay_on_newer} caption="NDWI change overlaid on newer image" onOpen={openLightbox} />
            </div>
          </Section>
        </>
      )}

      <Lightbox image={lightboxImage} onClose={() => setLightboxImage(null)} />
    </div>
  );
}

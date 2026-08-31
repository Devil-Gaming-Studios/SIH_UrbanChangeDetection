import { useState } from 'react'

const API_BASE = ''; // uses Vite proxy: /api -> backend, /files -> backend

function ImageCard({ src, caption }) {
  if (!src) return null;
  return (
    <figure>
      <img src={src} alt={caption} />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export default function App() {
  const [earlierFile, setEarlierFile] = useState(null);
  const [laterFile, setLaterFile] = useState(null);
  const [yearEarlier, setYearEarlier] = useState(2018);
  const [yearLater, setYearLater] = useState(2024);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

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
      <h1>Urban Change Detection</h1>
      <p className="subtitle">
        Upload two satellite images of the same area to detect built-up change,
        vegetation (NDVI) and water (NDWI) shifts, and download a full report.
      </p>

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
          <div className="card">
            <div className="section-title">Growth summary</div>
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
            </div>
            <ImageCard src={result.growth_chart} caption="Projected growth" />
            <a className="download-btn" href={result.report_pdf} download>
              <button type="button">Download full PDF report</button>
            </a>
          </div>

          <div className="card">
            <div className="section-title">Change detection</div>
            <div className="image-grid">
              <ImageCard src={result.images.change_mask} caption="Binary change bitmap" />
              <ImageCard src={result.images.change_prob} caption="Change probability heatmap" />
              <ImageCard src={result.images.overlay_on_newer} caption="Change overlaid on newer image" />
            </div>
          </div>

          <div className="card">
            <div className="section-title">NDVI (vegetation)</div>
            <div className="image-grid">
              <ImageCard src={result.images.ndvi_earlier} caption={`NDVI — ${yearEarlier}`} />
              <ImageCard src={result.images.ndvi_later} caption={`NDVI — ${yearLater}`} />
              <ImageCard src={result.images.ndvi_change} caption="NDVI change magnitude" />
              <ImageCard src={result.images.ndvi_overlay_on_newer} caption="NDVI change overlaid on newer image" />
            </div>
          </div>

          <div className="card">
            <div className="section-title">NDWI (water)</div>
            <div className="image-grid">
              <ImageCard src={result.images.ndwi_earlier} caption={`NDWI — ${yearEarlier}`} />
              <ImageCard src={result.images.ndwi_later} caption={`NDWI — ${yearLater}`} />
              <ImageCard src={result.images.ndwi_change} caption="NDWI change magnitude" />
              <ImageCard src={result.images.ndwi_overlay_on_newer} caption="NDWI change overlaid on newer image" />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

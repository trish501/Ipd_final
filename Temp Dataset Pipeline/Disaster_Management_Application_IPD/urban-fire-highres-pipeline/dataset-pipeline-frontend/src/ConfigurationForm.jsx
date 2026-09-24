import React, { useState, useEffect } from 'react';

function ConfigurationForm({ onStart, isRunning, activeConfig }) {
  const [config, setConfig] = useState({
    source: "VIIRS",
    loc_type: "World",
    loc_val: "World",
    bbox: "",
    sat: "Sentinel-2",
    mode: "B8A_AUXILIARY",
    start_date: "01-01-2025",
    end_date: "28-02-2025",
    target_images: 1500
  });

  useEffect(() => {
    if (activeConfig) {
      setConfig(prev => ({ ...prev, ...activeConfig }));
    }
  }, [activeConfig]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onStart(config);
  };

  return (
    <div className="glass-card">
      <h2 style={{ marginBottom: '1.5rem' }}>Pipeline Configuration</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label>Data Source</label>
            <select name="source" value={config.source} onChange={handleChange} className="form-control" disabled={isRunning}>
              <option value="VIIRS">VIIRS</option>
              <option value="MODIS">MODIS</option>
            </select>
          </div>

          <div className="form-group">
            <label>Location Type</label>
            <select name="loc_type" value={config.loc_type} onChange={handleChange} className="form-control" disabled={isRunning}>
              <option value="World">World</option>
              <option value="Country">Country</option>
              <option value="Custom">Custom Bounding Box</option>
            </select>
          </div>

          {config.loc_type === 'Country' && (
            <div className="form-group">
              <label>Country Name</label>
              <input type="text" name="loc_val" value={config.loc_val} onChange={handleChange} className="form-control" placeholder="e.g. India" disabled={isRunning} />
            </div>
          )}

          {config.loc_type === 'Custom' && (
            <div className="form-group">
              <label>Bounding Box (min_lat, max_lat, min_lon, max_lon)</label>
              <input type="text" name="bbox" value={config.bbox} onChange={handleChange} className="form-control" placeholder="e.g. 10.0, 20.0, 70.0, 80.0" disabled={isRunning} />
            </div>
          )}

          <div className="form-group">
            <label>Target Images</label>
            <input type="number" name="target_images" value={config.target_images} onChange={handleChange} className="form-control" min="1" disabled={isRunning} />
          </div>

          <div className="form-group">
            <label>Start Date (DD-MM-YYYY)</label>
            <input type="text" name="start_date" value={config.start_date} onChange={handleChange} className="form-control" disabled={isRunning} />
          </div>

          <div className="form-group">
            <label>End Date (DD-MM-YYYY)</label>
            <input type="text" name="end_date" value={config.end_date} onChange={handleChange} className="form-control" disabled={isRunning} />
          </div>

          <div className="form-group">
            <label>Pipeline Mode</label>
            <select name="mode" value={config.mode} onChange={handleChange} className="form-control" disabled={isRunning}>
              <option value="B8A_AUXILIARY">B8A Auxiliary (False-positive suppression)</option>
            </select>
          </div>
        </div>
        
        <button type="submit" className="btn-primary" disabled={isRunning}>
          {isRunning ? 'Pipeline is Running...' : 'Launch Pipeline'}
        </button>
      </form>
    </div>
  );
}

export default ConfigurationForm;

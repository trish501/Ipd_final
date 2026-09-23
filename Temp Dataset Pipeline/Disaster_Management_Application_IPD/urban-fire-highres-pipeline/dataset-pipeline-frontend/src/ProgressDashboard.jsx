import React, { useEffect, useState } from 'react';

function ProgressDashboard({ status, onStop }) {
  if (!status) return null;

  const {
    is_running,
    current_phase,
    elapsed_seconds,
    eta_seconds,
    overall_progress,
    phase1,
    phase2
  } = status;

  const formatTime = (seconds) => {
    if (!seconds || seconds <= 0) return '00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
  };

  const getStatusBadge = () => {
    if (current_phase === 'COMPLETED') return <div className="status-badge completed">Pipeline Completed</div>;
    if (is_running) return <div className="status-badge running">Pipeline Running</div>;
    return <div className="status-badge idle">Idle</div>;
  };

  const getPhaseText = () => {
    if (current_phase === 'PHASE_1_DOWNLOADING_CSVS') return 'Phase 1: Downloading FIRMS Data...';
    if (current_phase === 'PHASE_2_GENERATING_IMAGES') return 'Phase 2: Generating Satellite Images...';
    if (current_phase === 'COMPLETED') return 'Finished';
    return 'Waiting to start...';
  };

  return (
    <div className="glass-card" style={{ marginTop: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Live Dashboard</h2>
        {getStatusBadge()}
      </div>
      
      <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>{getPhaseText()}</p>

      <div className="dashboard-metrics">
        <div className="metric-card">
          <h3>Elapsed Time</h3>
          <div className="value">{formatTime(elapsed_seconds)}</div>
        </div>
        <div className="metric-card">
          <h3>Est. Remaining</h3>
          <div className="value">{is_running ? formatTime(eta_seconds) : 'N/A'}</div>
        </div>
        <div className="metric-card success">
          <h3>Images Generated</h3>
          <div className="value">{phase2.generated} <span style={{fontSize: '1rem', color: 'var(--text-muted)'}}>/ {phase2.target}</span></div>
        </div>
        <div className="metric-card error">
          <h3>Failed Events</h3>
          <div className="value">{phase2.failed}</div>
        </div>
        <div className="metric-card warning">
          <h3>Events Checked</h3>
          <div className="value">{phase2.total_events_checked}</div>
        </div>
      </div>

      <div className="progress-container">
        <div className="progress-header">
          <span>Overall Progress</span>
          <span>{overall_progress.toFixed(1)}%</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${overall_progress}%` }}></div>
        </div>
      </div>

      {current_phase === 'PHASE_1_DOWNLOADING_CSVS' && (
        <div className="progress-container">
          <div className="progress-header">
            <span>Downloading CSVs ({phase1.downloaded} / {phase1.expected})</span>
            <span>{phase1.progress.toFixed(1)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${phase1.progress}%`, background: 'var(--primary)' }}></div>
          </div>
        </div>
      )}

      {current_phase === 'PHASE_2_GENERATING_IMAGES' && (
        <div className="progress-container">
          <div className="progress-header">
            <span>Generating Images ({phase2.generated} / {phase2.target})</span>
            <span>{phase2.progress.toFixed(1)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${phase2.progress}%`, background: 'var(--accent-success)' }}></div>
          </div>
        </div>
      )}

      {is_running && (
        <button onClick={onStop} className="btn-danger">
          Stop Pipeline
        </button>
      )}
    </div>
  );
}

export default ProgressDashboard;

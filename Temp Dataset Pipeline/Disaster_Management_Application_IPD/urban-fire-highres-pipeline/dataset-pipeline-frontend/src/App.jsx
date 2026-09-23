import React, { useState, useEffect } from 'react';
import './App.css';
import ConfigurationForm from './ConfigurationForm';
import ProgressDashboard from './ProgressDashboard';

function App() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async (config) => {
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const data = await res.json();
        setError(data.detail || 'Failed to start pipeline');
      } else {
        fetchStatus();
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStop = async () => {
    try {
      await fetch('http://localhost:8000/api/stop', { method: 'POST' });
      fetchStatus();
    } catch (err) {
      console.error("Error stopping pipeline:", err);
    }
  };

  const isRunning = status?.is_running || false;

  return (
    <div className="app-container">
      <header className="header">
        <h1>Urban Fire Dataset Pipeline</h1>
        <p>Dynamic High-Resolution Satellite Generation</p>
      </header>

      {error && (
        <div style={{ background: 'var(--accent-error)', color: 'white', padding: '1rem', borderRadius: '8px', marginBottom: '1rem' }}>
          {error}
        </div>
      )}

      <ConfigurationForm onStart={handleStart} isRunning={isRunning} activeConfig={status?.active_config} />
      
      <ProgressDashboard status={status} onStop={handleStop} />
    </div>
  );
}

export default App;

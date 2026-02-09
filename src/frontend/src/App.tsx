import { useState } from 'react';
import { CameraView } from './components/CameraView';
import './App.css';

function App() {
  const [cameraError, setCameraError] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🩺 AI Guardian</h1>
        <p className="app-subtitle">Caring for your child's health</p>
      </header>

      <main className="app-main">
        <CameraView onError={setCameraError} />
      </main>

      {cameraError && (
        <div className="app-footer">
          <p className="error-status">Camera: {cameraError}</p>
        </div>
      )}
    </div>
  );
}

export default App;

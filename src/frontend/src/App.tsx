import { useState } from 'react';
import { VideoUploadView } from './components/VideoUploadView';
import './App.css';

function App() {
  const [uploadError, setUploadError] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-content">
          <img src="/ai-guardian-logo.svg" alt="AI Guardian" className="app-logo" />
          <div>
            <h1>AI Guardian</h1>
            <p className="app-subtitle">Infant Health Monitor</p>
          </div>
        </div>
      </header>

      <main className="app-main">
        <VideoUploadView onError={setUploadError} />
      </main>

      {uploadError && (
        <div className="app-footer">
          <p className="error-status">Error: {uploadError}</p>
        </div>
      )}

      <footer className="app-disclaimer">
        <p>This tool is for informational purposes only and does not replace professional medical advice.</p>
      </footer>
    </div>
  );
}

export default App;

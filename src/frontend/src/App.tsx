import { useState } from 'react';
import { VideoUploadView } from './components/VideoUploadView';
import './App.css';

function App() {
  const [uploadError, setUploadError] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🩺 AI Guardian</h1>
        <p className="app-subtitle">Multi-modal health analysis</p>
      </header>

      <main className="app-main">
        <VideoUploadView onError={setUploadError} />
      </main>

      {uploadError && (
        <div className="app-footer">
          <p className="error-status">Error: {uploadError}</p>
        </div>
      )}
    </div>
  );
}

export default App;

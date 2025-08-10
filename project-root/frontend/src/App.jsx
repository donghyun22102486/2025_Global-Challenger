import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import PredictionPage from "./pages/PredictionPage";

function App() {
  return (
    <Router>
      <div className="min-h-screen w-full bg-gray-900 text-white px-4 py-8">
        <div className="max-w-5xl mx-auto">
          <header className="text-center mb-8">
            <h1 className="text-4xl font-extrabold">
              📊 Data Insight Platform
            </h1>
            <p className="text-gray-400 mt-2 text-sm">
              LLM 기반 자동 전처리 & 모델 학습 파이프라인
            </p>
          </header>

          <Routes>
            <Route path="/" element={<Navigate to="/upload" />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/predict" element={<PredictionPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import EdaPage from "./pages/EdaPage";
import PredictPage from "./pages/PredictPage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/eda/:id" element={<EdaPage />} />
        <Route path="/predict" element={<PredictPage />} />
      </Routes>
    </Router>
  );
}

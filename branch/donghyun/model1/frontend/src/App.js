import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    X_Minimum: 800,
    X_Maximum: 1000,
    Y_Minimum: 900,
    Y_Maximum: 1100,
    Pixels_Areas: 2000,
    X_Perimeter: 100,
    Y_Perimeter: 100,
    Sum_of_Luminosity: 250000,
    Minimum_of_Luminosity: 30,
    Maximum_of_Luminosity: 250,
    Length_of_Conveyer: 2000,
    TypeOfSteel_A300: 1,
    TypeOfSteel_A400: 0,
    Steel_Plate_Thickness: 80,
    Edges_Index: 0.3,
    Empty_Index: 0.1,
    Square_Index: 0.2,
    Outside_X_Index: 0.05,
    Edges_X_Index: 0.25,
    Edges_Y_Index: 0.25,
    Outside_Global_Index: 0.02,
    LogOfAreas: 7.6,
    Log_X_Index: 9.9,
    Log_Y_Index: 9.8,
    Orientation_Index: 0.0,
    Luminosity_Index: 0.4,
    SigmoidOfAreas: 0.98,
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value === "" ? "" : parseFloat(value),
    }));
  };

  const handlePredict = async () => {
    const data = Object.fromEntries(
      Object.entries(formData).map(([k, v]) => [k, parseFloat(v)])
    );

    try {
      const response = await axios.post(
        "http://localhost:8000/api/predict",
        data
      );
      const { predicted_fault, expert_report, root_cause_analysis } =
        response.data;

      setResult({
        predicted_fault,
        expert_report,
        root_cause_analysis,
      });
    } catch (error) {
      console.error("❌ 예측 실패:", error);
      alert("예측 요청 실패");
    }
  };

  return (
    <div className="App">
      <h1>불량 예측 시스템</h1>

      <div className="form">
        {Object.keys(formData).map((key) => (
          <label key={key}>
            {key}:
            <input
              type="number"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              step="any"
            />
          </label>
        ))}
        <button onClick={handlePredict}>예측 실행</button>
      </div>

      {result && (
        <div className="result">
          <h2>🔧 예측된 불량 유형: {result.predicted_fault}</h2>

          <h3>📋 전문가 보고서:</h3>
          <pre>{result.expert_report}</pre>

          <h3>🔍 주요 원인 변수</h3>
          <ul>
            {result.root_cause_analysis.map((f, idx) => (
              <li key={idx}>
                {f.feature} (중요도: {(f.importance * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;

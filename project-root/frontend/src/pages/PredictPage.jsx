import React, { useState, useEffect } from "react";
import api from "../api";

function PredictPage() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [nlInput, setNlInput] = useState("");
  const [jsonInput, setJsonInput] = useState("{}");
  const [result, setResult] = useState(null);

  useEffect(() => {
    api
      .get("/list-models")
      .then((res) => {
        if (res.data.models) {
          setModels(res.data.models);
          setSelectedModel(res.data.models[0] || "");
        }
      })
      .catch((err) => console.error(err));
  }, []);

  const handleNlPredict = async () => {
    const formData = new FormData();
    formData.append("user_request", nlInput);
    formData.append("model_file", selectedModel);

    const res = await api.post("/predict-nl", formData);
    setResult(res.data);
  };

  const handleJsonPredict = async () => {
    const formData = new FormData();
    formData.append("input_data", jsonInput);
    formData.append("model_file", selectedModel);

    const res = await api.post("/predict", formData);
    setResult(res.data);
  };

  return (
    <div>
      <h2>예측 페이지</h2>
      <label>모델 선택</label>
      <select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>

      <h3>자연어 예측</h3>
      <textarea
        value={nlInput}
        onChange={(e) => setNlInput(e.target.value)}
        placeholder="예: 2013년에 평균 매출이 5000원일 때 예측"
      />
      <button onClick={handleNlPredict}>예측 실행</button>

      <h3>JSON 예측</h3>
      <textarea
        value={jsonInput}
        onChange={(e) => setJsonInput(e.target.value)}
        placeholder='{"feature1": 10, "feature2": 5}'
      />
      <button onClick={handleJsonPredict}>예측 실행</button>

      {result && (
        <div>
          <h4>결과</h4>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default PredictPage;

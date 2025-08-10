import React, { useState, useEffect } from "react";
import axios from "axios";

function PredictionPage() {
  const [userRequest, setUserRequest] = useState("");
  const [modelList, setModelList] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("result");

  useEffect(() => {
    axios.get("http://localhost:8000/list-models").then((res) => {
      if (res.data.models) {
        setModelList(res.data.models);
        setSelectedModel(res.data.models[0] || "");
      }
    });
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userRequest.trim()) return;

    const formData = new FormData();
    formData.append("user_request", userRequest);
    if (selectedModel) {
      formData.append("model_file", selectedModel);
    }

    try {
      setLoading(true);
      setResult(null);
      setExplanation("");
      setActiveTab("result");

      const res = await axios.post(
        "http://localhost:8000/predict-nl",
        formData
      );
      setResult(res.data);

      const explanationForm = new FormData();
      explanationForm.append("prediction_result", JSON.stringify(res.data));

      const expRes = await axios.post(
        "http://localhost:8000/explain-prediction",
        explanationForm
      );
      setExplanation(expRes.data.explanation || "⚠️ 설명 없음");
    } catch (err) {
      alert("❌ 예측 실패: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block mb-1 font-medium">🧾 예측 요청 문장</label>
          <textarea
            value={userRequest}
            onChange={(e) => setUserRequest(e.target.value)}
            placeholder="예: 2022년에 매출이 5000이면 얼마일까?"
            rows={3}
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          />
        </div>

        <div>
          <label className="block mb-1 font-medium">📦 모델 선택</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          >
            {modelList.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`px-6 py-2 rounded text-sm font-semibold ${
            loading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          } text-white w-full`}
        >
          {loading ? "⏳ 예측 중..." : "🚀 예측 실행"}
        </button>
      </form>

      {result && (
        <>
          {/* 탭 버튼 */}
          <div className="flex space-x-4 mt-6">
            <button
              className={`px-4 py-2 rounded-t ${
                activeTab === "result"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300"
              }`}
              onClick={() => setActiveTab("result")}
            >
              📊 예측 결과
            </button>

            <button
              className={`px-4 py-2 rounded-t ${
                activeTab === "explanation"
                  ? "bg-pink-600 text-white"
                  : "bg-gray-700 text-gray-300"
              }`}
              onClick={() => setActiveTab("explanation")}
              disabled={!explanation || explanation === "⚠️ 설명 없음"}
            >
              🧠 예측 해설
            </button>
          </div>

          {/* 탭 콘텐츠 */}
          {activeTab === "result" && (
            <div className="bg-gray-800 border border-gray-700 rounded-b p-4 text-sm space-y-4">
              <h3 className="text-lg font-bold text-white mb-1">
                📊 예측 결과
              </h3>
              <p className="text-blue-400 text-xl">{result.prediction}</p>
              <div>
                <h3 className="font-semibold text-white">
                  📥 입력값 (파싱 결과)
                </h3>
                <ul className="list-disc pl-6 text-gray-300">
                  {Object.entries(result.parsed_input).map(([key, val]) => (
                    <li key={key}>
                      <span className="text-white">{key}:</span> {val}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {activeTab === "explanation" && explanation && (
            <div className="bg-gray-900 border border-pink-500 rounded-b p-4 text-sm">
              <h3 className="text-lg font-bold text-pink-400 mb-2">
                🧠 예측 해설
              </h3>
              <p className="text-gray-300 whitespace-pre-wrap">{explanation}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default PredictionPage;

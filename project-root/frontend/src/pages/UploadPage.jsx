import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function UploadPage() {
  const [file, setFile] = useState(null);
  const [userRequest, setUserRequest] = useState("");
  const [processOption, setProcessOption] = useState("full");
  const [modelType, setModelType] = useState("rf");
  const [targetColumn, setTargetColumn] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("파일을 업로드해주세요.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("process_option", processOption);
    formData.append("model_type", modelType);
    formData.append("target_col_override", targetColumn);
    formData.append("user_request_text", userRequest);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:8000/process", formData);
      setResult(res.data);
    } catch (err) {
      alert(`에러 발생: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="grid grid-cols-2 gap-6 text-sm max-w-3xl mx-auto"
    >
      {/* 왼쪽 열 */}
      <div className="space-y-5">
        <div>
          <label className="block mb-1 font-medium">📁 파일 업로드</label>
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2 file:bg-blue-700 file:text-white file:border-0 file:px-4 file:py-1 hover:file:bg-blue-600"
          />
        </div>

        <div>
          <label className="block mb-1 font-medium">⚙️ 처리 옵션</label>
          <select
            value={processOption}
            onChange={(e) => setProcessOption(e.target.value)}
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          >
            <option value="full">전처리 + 학습 (full)</option>
            <option value="preprocess_only">전처리만</option>
            <option value="train_only">학습만</option>
          </select>
        </div>

        <div>
          <label className="block mb-1 font-medium">🧠 모델 선택</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          >
            <option value="rf">Random Forest</option>
            <option value="gbr">Gradient Boosting</option>
          </select>
        </div>
      </div>

      {/* 오른쪽 열 */}
      <div className="space-y-5">
        <div>
          <label className="block mb-1 font-medium">📝 사용자 요청</label>
          <textarea
            value={userRequest}
            onChange={(e) => setUserRequest(e.target.value)}
            rows={5}
            placeholder="예: 총수익 = 매출 - 세금"
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          />
        </div>

        <div>
          <label className="block mb-1 font-medium">🎯 타겟 컬럼</label>
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="예: heat_demand"
            className="w-full bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
          />
        </div>
      </div>

      {/* 버튼 */}
      <div className="md:col-span-2 flex justify-center mt-4">
        <button
          type="submit"
          disabled={loading}
          className={`px-6 py-2 rounded text-sm font-semibold ${
            loading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          } text-white`}
        >
          {loading ? "⏳ 처리 중..." : "🚀 분석 시작"}
        </button>
      </div>

      {/* 예측 페이지로 이동 버튼 */}
      <div className="md:col-span-2 flex justify-center mt-2">
        <button
          type="button"
          onClick={() => navigate("/predict")}
          className="text-blue-400 underline text-sm hover:text-blue-300"
        >
          👉 예측 페이지로 이동
        </button>
      </div>

      {/* 결과 */}
      {result && (
        <div className="md:col-span-2 mt-4 bg-gray-800 border border-gray-700 p-4 rounded text-white text-sm space-y-2">
          <p>
            ✅ 처리 완료: <strong>{result.timestamp}</strong>
          </p>
          {result.metrics && (
            <div className="grid grid-cols-2 gap-2">
              <p>📉 RMSE: {result.metrics.rmse}</p>
              <p>📈 R²: {result.metrics.r2}</p>
              <p>📊 MAE: {result.metrics.mae}</p>
              <p>🔁 CV R²: {result.metrics.cv_r2}</p>
            </div>
          )}
          <div className="mt-2 space-x-4">
            <a
              href={`http://localhost:8000${result.csv_url}`}
              className="underline text-blue-400"
              target="_blank"
            >
              📄 CSV
            </a>
            <a
              href={`http://localhost:8000${result.report_url}`}
              className="underline text-blue-400"
              target="_blank"
            >
              📝 리포트
            </a>
            <a
              href={`http://localhost:8000${result.eda_url}`}
              className="underline text-blue-400"
              target="_blank"
            >
              📊 EDA
            </a>
          </div>
        </div>
      )}
    </form>
  );
}

export default UploadPage;

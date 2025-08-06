import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

function EdaPage() {
  const { id } = useParams(); // timestamp
  const navigate = useNavigate();
  const [report, setReport] = useState("");
  const [edaUrl, setEdaUrl] = useState("");
  const [csvUrl, setCsvUrl] = useState("");

  useEffect(() => {
    setEdaUrl(`${import.meta.env.VITE_API_BASE_URL}/download/${id}/eda`);
    setCsvUrl(`${import.meta.env.VITE_API_BASE_URL}/download/${id}/csv`);

    fetch(`${import.meta.env.VITE_API_BASE_URL}/download/${id}/report`)
      .then((res) => res.text())
      .then((text) => setReport(text))
      .catch((err) => console.error(err));
  }, [id]);

  return (
    <div>
      <h2>EDA 결과</h2>
      <img
        src={edaUrl}
        alt="EDA Heatmap"
        style={{ maxWidth: "600px", border: "1px solid #ccc" }}
      />
      <h3>📄 보고서</h3>
      <pre>{report}</pre>
      <a href={csvUrl} download>
        📥 전처리 CSV 다운로드
      </a>
      <br />
      <br />
      <button onClick={() => navigate("/predict")}>➡ 예측 페이지로 이동</button>
    </div>
  );
}

export default EdaPage;

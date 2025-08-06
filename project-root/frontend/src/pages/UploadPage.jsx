import React, { useState } from "react";
import api from "../api";
import { useNavigate } from "react-router-dom";

function UploadPage() {
  const [file, setFile] = useState(null);
  const [processOption, setProcessOption] = useState("full");
  const [modelType, setModelType] = useState("rf");
  const [targetCol, setTargetCol] = useState("");
  const [userRequest, setUserRequest] = useState("");
  const [result, setResult] = useState(null);

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("process_option", processOption);
    formData.append("model_type", modelType);
    formData.append("target_col_override", targetCol);
    formData.append("user_request_text", userRequest);

    try {
      const res = await api.post("/process", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);

      if (res.data.timestamp) {
        navigate(`/eda/${res.data.timestamp}`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <h2>파일 업로드 & 전처리</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <select
          value={processOption}
          onChange={(e) => setProcessOption(e.target.value)}
        >
          <option value="full">Full</option>
          <option value="preprocess_only">Preprocess Only</option>
          <option value="train_only">Train Only</option>
        </select>
        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
        >
          <option value="rf">Random Forest</option>
          <option value="gbr">Gradient Boosting</option>
        </select>
        <input
          type="text"
          placeholder="Target Column"
          value={targetCol}
          onChange={(e) => setTargetCol(e.target.value)}
        />
        <textarea
          placeholder="User Request"
          value={userRequest}
          onChange={(e) => setUserRequest(e.target.value)}
        />
        <button type="submit">전송</button>
      </form>

      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

export default UploadPage;

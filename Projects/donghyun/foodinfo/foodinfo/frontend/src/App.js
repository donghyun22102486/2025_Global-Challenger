import React, { useEffect, useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
} from "chart.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, Legend, Tooltip);

function App() {
  const [categories, setCategories] = useState([]);
  const [nutrients, setNutrients] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedNutrients, setSelectedNutrients] = useState([]);
  const [resultData, setResultData] = useState([]);

  // FastAPI로부터 카테고리, 영양성분 목록 가져오기
  useEffect(() => {
    axios
      .get("http://localhost:8000/categories")
      .then((res) => setCategories(res.data));
    axios
      .get("http://localhost:8000/nutrients")
      .then((res) => setNutrients(res.data));
  }, []);

  // 분석 요청
  const analyze = () => {
    if (!selectedCategory || selectedNutrients.length === 0) {
      alert("대분류와 영양성분을 모두 선택해주세요.");
      return;
    }

    axios
      .post("http://localhost:8000/analyze", {
        category: selectedCategory,
        nutrients: selectedNutrients,
      })
      .then((res) => setResultData(res.data))
      .catch((err) => {
        console.error("분석 실패:", err);
        alert("분석 중 오류가 발생했습니다.");
      });
  };

  // 그래프 데이터 구성
  const chartData = {
    labels: selectedNutrients,
    datasets: ["가정식", "외식", "급식"].map((type) => ({
      label: type,
      data: selectedNutrients.map((nutrient) => {
        const row = resultData.find((d) => d["식사형태"] === type);
        return row ? row[nutrient] : 0;
      }),
      backgroundColor:
        type === "가정식" ? "#4caf50" : type === "외식" ? "#2196f3" : "#ff9800",
    })),
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial, sans-serif" }}>
      <h2>🍱 식사형태별 영양성분 비교</h2>

      <div style={{ marginBottom: "20px" }}>
        <label>대분류 선택:&nbsp;</label>
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          <option value="">-- 대분류 선택 --</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>비교할 영양성분 선택:</label>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "10px",
            marginTop: "10px",
          }}
        >
          {nutrients.map((n) => (
            <label key={n}>
              <input
                type="checkbox"
                value={n}
                checked={selectedNutrients.includes(n)}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setSelectedNutrients((prev) =>
                    checked ? [...prev, n] : prev.filter((x) => x !== n)
                  );
                }}
              />
              {n}
            </label>
          ))}
        </div>
      </div>

      <button
        onClick={analyze}
        style={{ padding: "10px 20px", fontWeight: "bold" }}
      >
        분석하기
      </button>

      {resultData.length > 0 && (
        <div style={{ marginTop: "40px" }}>
          <Bar data={chartData} />
        </div>
      )}
    </div>
  );
}

export default App;

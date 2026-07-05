import { useState } from "react";

function App() {
  // 1. Add state to track if the camera is running
  const [isStreaming, setIsStreaming] = useState(false);

  const detect = async () => {
    console.log("Sending start detect request...");
    const request = {
      source: 0,
      confidence: 0.5
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      const data = await response.json();
      console.log(data);

      // 2. If backend successfully started, turn the UI stream on
      if (response.ok) {
        setIsStreaming(true);
      }
    } catch (error) {
      console.error("Error starting detection:", error);
    }
  };

  const stopDetect = async () => {
    console.log("Sending stop detect request");
    try {
      const response = await fetch("http://127.0.0.1:8000/api/stopDetect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      const data = await response.json();
      console.log(data);

      // 3. Turn the UI stream off
      setIsStreaming(false);
    } catch (error) {
      console.error("Error stopping detection:", error);
    }
  };

  return (
    <div style={{ minHeight: "100vh", padding: "24px", fontFamily: "Inter, sans-serif", background: "#f4f7fb", color: "#172033" }}>
      <div style={{ maxWidth: "980px", margin: "0 auto" }}>
        <div style={{ marginBottom: "20px" }}>
          <h2 style={{ margin: "0 0 8px", fontSize: "28px", fontWeight: 700 }}>Robot Vision Dashboard</h2>
          <p style={{ margin: 0, color: "#5f6b82" }}>Monitor live detections with a clean, focused view.</p>
        </div>

        <div style={{ display: "flex", gap: "12px", marginBottom: "20px", flexWrap: "wrap" }}>
          <button
            onClick={detect}
            style={{
              border: "none",
              borderRadius: "999px",
              padding: "10px 18px",
              background: "#2563eb",
              color: "white",
              cursor: "pointer",
              fontWeight: 600,
              boxShadow: "0 6px 16px rgba(37, 99, 235, 0.18)",
            }}
          >
            Start Detect
          </button>
          <button
            onClick={stopDetect}
            style={{
              border: "1px solid #d6dce7",
              borderRadius: "999px",
              padding: "10px 18px",
              background: "white",
              color: "#334155",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Stop Detect
          </button>
        </div>

        <div
          style={{
            background: "white",
            borderRadius: "16px",
            boxShadow: "0 12px 30px rgba(15, 23, 42, 0.08)",
            padding: "18px",
            border: "1px solid #e9eef6",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <h3 style={{ margin: 0, fontSize: "18px" }}>Live Camera Feed</h3>
            <span
              style={{
                padding: "6px 10px",
                borderRadius: "999px",
                background: isStreaming ? "#dcfce7" : "#f1f5f9",
                color: isStreaming ? "#166534" : "#475569",
                fontSize: "13px",
                fontWeight: 600,
              }}
            >
              {isStreaming ? "Streaming" : "Idle"}
            </span>
          </div>

          {isStreaming ? (
            <img
              src={`http://127.0.0.1:8000/api/video-feed?t=${Date.now()}`}
              alt="Live YOLO Stream"
              style={{ width: "100%", maxWidth: "100%", borderRadius: "12px", display: "block", background: "#0f172a" }}
            />
          ) : (
            <div
              style={{
                minHeight: "320px",
                border: "1px dashed #d7deea",
                borderRadius: "12px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#64748b",
                background: "#fafcff",
              }}
            >
              Start detection to begin streaming the live feed.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
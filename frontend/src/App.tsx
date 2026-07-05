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
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h2>Robot Vision Dashboard</h2>
      
      <div style={{ marginBottom: "20px" }}>
        <button onClick={detect} style={{ marginRight: "10px", padding: "10px 20px" }}>Start Detect</button>
        <button onClick={stopDetect} style={{ padding: "10px 20px" }}>Stop Detect</button>
      </div>

      {/* 4. Conditionally render the image tag when streaming is true */}
      {isStreaming && (
        <div style={{ border: "2px solid #333", padding: "10px", display: "inline-block", borderRadius: "10px" }}>
          <h3 style={{ marginTop: 0 }}>Live Camera Feed</h3>
          
          {/* PRO TIP: We add ?t=${Date.now()} to the end of the URL. 
            This forces the browser to open a fresh connection every time you press Start, 
            preventing annoying browser caching bugs when starting/stopping the stream!
          */}
          <img 
            src={`http://127.0.0.1:8000/api/video-feed?t=${Date.now()}`} 
            alt="Live YOLO Stream" 
            style={{ width: "100%", maxWidth: "640px", borderRadius: "8px" }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
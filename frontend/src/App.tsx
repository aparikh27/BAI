import { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isShowingObjects, setIsShowingObjects] = useState<boolean>(false);
  const [visibleObjectsList, setVisibleObjectsList] = useState<any[]>([]);
  const [speechTranscript, setSpeechTranscript] = useState<string>("Waiting for voice command...");
  const [robotCommand, setRobotCommand] = useState<string>("No robot command yet.");
  const pollingRef = useRef<number | null>(null);
  const speechEventSourceRef = useRef<EventSource | null>(null);


  const detect = async () => {
    console.log("Sending start detect request...");
    const request = {
      source: 0,
      confidence: 0.5
    };

    try {
      const response = await fetch("/api/detect", {
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
        startStreaming();
      }
    } catch (error) {
      console.error("Error starting detection:", error);
    }
  };

  const stopDetect = async () => {
    console.log("Sending stop detect request");
    try {
      const response = await fetch("/api/stopDetect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      const data = await response.json();
      console.log(data);

      if (speechEventSourceRef.current) {
        speechEventSourceRef.current.close();
        speechEventSourceRef.current = null;
      }

      setSpeechTranscript("Listening stopped.");
      setRobotCommand("No robot command yet.");

      // 3. Turn the UI stream off
      setIsStreaming(false);
    } catch (error) {
      console.error("Error stopping detection:", error);
    }
  };

  const fetchVisibleObjects = async () => {
    try {
      const response = await fetch("/api/visible-objects");
      if (!response.ok) return;
      const data = await response.json();
      setVisibleObjectsList(data.visible_objects || []);
    } catch (error) {
      console.error("Error fetching visible objects:", error);
    }
  };

  const startStreaming = () => {
    if (speechEventSourceRef.current) {
      speechEventSourceRef.current.close();
    }

    setSpeechTranscript("Listening for a voice command...");
    setRobotCommand("Waiting for robot command...");

    const eventSource = new EventSource("/api/stream-speech");
    speechEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.transcript) {
          setSpeechTranscript(data.transcript);
        }
        if (data.command !== undefined) {
          setRobotCommand(data.command);
        }
      } catch (error) {
        console.error("Error parsing streaming chunk:", error);
      }
    };

    eventSource.onerror = (error) => {
      console.error("Speech stream disconnected; retrying:", error);
      // EventSource reconnects automatically. Closing it here prevents the
      // dashboard from recovering after a transient backend startup failure.
      if (speechEventSourceRef.current === eventSource) {
        setSpeechTranscript("Speech stream disconnected. Reconnecting...");
      }
    };

    return eventSource;
  };

  
  const toggleShowingObjects = () => {
    setIsShowingObjects((prev) => {
      const next = !prev;
      if (next) {
        fetchVisibleObjects();
      }
      return next;
    });
  };

  
  useEffect(() => {
    if (isShowingObjects) {
      pollingRef.current = window.setInterval(() => {
        fetchVisibleObjects();
      }, 1000);
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [isShowingObjects]);

  useEffect(() => {
    return () => {
      if (speechEventSourceRef.current) {
        speechEventSourceRef.current.close();
        speechEventSourceRef.current = null;
      }
    };
  }, []);

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
        <button className="objects-toggle" onClick={toggleShowingObjects}>
          {isShowingObjects ? "Hide" : "Show"} Objects
        </button>

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
              src={`/api/video-feed?t=${Date.now()}`}
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
        <div
          style={{
            marginTop: "20px",
            background: "white",
            borderRadius: "16px",
            boxShadow: "0 12px 30px rgba(15, 23, 42, 0.08)",
            padding: "18px",
            border: "1px solid #e9eef6",
          }}
        >
          <h3 style={{ margin: "0 0 8px", fontSize: "18px" }}>Voice Commands</h3>
          <p style={{ margin: 0, color: "#475569", lineHeight: 1.5 }}>{speechTranscript}</p>
          <p style={{ margin: "8px 0 0", color: "#2563eb", lineHeight: 1.5, fontWeight: 600, whiteSpace: "pre-wrap" }}>
            Robot Command: {robotCommand}
          </p>
        </div>
        {isShowingObjects && (
          <div className="objects-panel">
            <h4 className="objects-title">Visible Objects</h4>
            {visibleObjectsList.length === 0 ? (
              <div className="objects-empty">No objects currently visible</div>
            ) : (
              <div className="objects-list">
                {visibleObjectsList.map((obj: any, idx: number) => (
                  <div key={idx} className="object-card">
                    <div className="object-header">
                      <span className="object-class">{obj.class_name}</span>
                      <span className="object-lastseen">frame: {obj.last_seen_frame}</span>
                    </div>
                    <div className="object-body">Box: [{(obj.box || []).map((n: number) => Math.round(n)).join(", ")}]
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

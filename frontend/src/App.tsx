import { useState, useEffect, useRef } from "react";
import "./App.css";

type TaskState = "idle" | "running" | "completed" | "failed";

function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isShowingObjects, setIsShowingObjects] = useState<boolean>(false);
  const [visibleObjectsList, setVisibleObjectsList] = useState<any[]>([]);
  const [speechTranscript, setSpeechTranscript] = useState<string>("Waiting for voice command...");
  const [robotCommand, setRobotCommand] = useState<string>("No robot command yet.");
  const [telemetryConnected, setTelemetryConnected] = useState<boolean>(false);
  const [taskState, setTaskState] = useState<TaskState>("idle");
  const [taskMessage, setTaskMessage] = useState<string>("");
  const [stepLabel, setStepLabel] = useState<string>("");
  const [stepIndex, setStepIndex] = useState<number>(0);
  const [stepTotal, setStepTotal] = useState<number>(0);
  const [streamKey, setStreamKey] = useState<number>(0);
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
        // Fix the cache-buster once per session. Evaluating Date.now() in the
        // render body gave the <img> a new src on every re-render, which
        // aborted the in-flight MJPEG stream and restarted it each time state
        // changed — visible as flicker, and as ERR_ABORTED in the network log.
        setStreamKey(Date.now());
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
      setTelemetryConnected(false);
      resetTask();

      // 3. Turn the UI stream off
      setIsStreaming(false);
    } catch (error) {
      console.error("Error stopping detection:", error);
    }
  };

  const resetTask = () => {
    setTaskState("idle");
    setTaskMessage("");
    setStepLabel("");
    setStepIndex(0);
    setStepTotal(0);
  };

  const runVoiceCommand = async () => {
    console.log("Injecting recorded voice command...");
    resetTask();
    try {
      const response = await fetch("/api/inject-voice", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      const data = await response.json();
      console.log(data);

      if (!response.ok) {
        setTaskState("failed");
        setTaskMessage(data?.detail ?? "Could not start the voice command.");
      }
    } catch (error) {
      console.error("Error injecting voice command:", error);
      setTaskState("failed");
      setTaskMessage("Could not reach the backend to start the voice command.");
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

    // Live microphone capture is on by default. Opening the dashboard with
    // ?mic=0 streams telemetry without it, so a scripted run cannot be
    // derailed by ambient room noise being transcribed into a robot command.
    const micParam = new URLSearchParams(window.location.search).get("mic");
    const micEnabled = micParam === null ? "1" : micParam;
    const eventSource = new EventSource(`/api/stream-speech?mic=${encodeURIComponent(micEnabled)}`);
    speechEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.connected) {
          setTelemetryConnected(true);
        }
        if (data.transcript) {
          setSpeechTranscript(data.transcript);
        }
        if (data.command !== undefined) {
          setRobotCommand(data.command);
        }
        if (data.step_label) {
          setStepLabel(data.step_label);
        }
        if (typeof data.step_index === "number") {
          setStepIndex(data.step_index);
        }
        if (typeof data.step_total === "number") {
          setStepTotal(data.step_total);
        }
        if (data.task_state) {
          setTaskState(data.task_state as TaskState);
        }
        if (data.message) {
          setTaskMessage(data.message);
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
        setTelemetryConnected(false);
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

  const taskBanner: Record<TaskState, { label: string; bg: string; fg: string } | null> = {
    idle: null,
    running: { label: "Task Running", bg: "#fef3c7", fg: "#92400e" },
    completed: { label: "Task Completed", bg: "#dcfce7", fg: "#166534" },
    failed: { label: "Task Failed", bg: "#fee2e2", fg: "#991b1b" },
  };
  const banner = taskBanner[taskState];

  return (
    <div style={{ minHeight: "100vh", padding: "24px", fontFamily: "Inter, sans-serif", background: "#f4f7fb", color: "#172033" }}>
      <div style={{ maxWidth: "980px", margin: "0 auto" }}>
        <div style={{ marginBottom: "20px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "16px", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ margin: "0 0 8px", fontSize: "28px", fontWeight: 700 }}>Robot Vision Dashboard</h2>
            <p style={{ margin: 0, color: "#5f6b82" }}>Monitor live detections with a clean, focused view.</p>
          </div>
          <span
            data-testid="telemetry-indicator"
            data-connected={telemetryConnected ? "true" : "false"}
            style={{
              padding: "8px 14px",
              borderRadius: "999px",
              background: telemetryConnected ? "#dcfce7" : "#f1f5f9",
              color: telemetryConnected ? "#166534" : "#475569",
              fontSize: "13px",
              fontWeight: 600,
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              whiteSpace: "nowrap",
            }}
          >
            <span
              style={{
                width: "9px",
                height: "9px",
                borderRadius: "999px",
                background: telemetryConnected ? "#16a34a" : "#94a3b8",
                display: "inline-block",
              }}
            />
            {telemetryConnected ? "Telemetry Connected" : "Telemetry Offline"}
          </span>
        </div>

        <div style={{ display: "flex", gap: "12px", marginBottom: "20px", flexWrap: "wrap" }}>
          <button
            onClick={detect}
            data-testid="start-detect"
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
            data-testid="stop-detect"
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
          <button
            onClick={runVoiceCommand}
            data-testid="run-voice-command"
            disabled={!isStreaming}
            style={{
              border: "none",
              borderRadius: "999px",
              padding: "10px 18px",
              background: isStreaming ? "#7c3aed" : "#cbd5e1",
              color: "white",
              cursor: isStreaming ? "pointer" : "not-allowed",
              fontWeight: 600,
              boxShadow: isStreaming ? "0 6px 16px rgba(124, 58, 237, 0.18)" : "none",
            }}
          >
            Run Voice Command
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
              data-testid="stream-status"
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
              src={`/api/video-feed?t=${streamKey}`}
              alt="Live YOLO Stream"
              data-testid="video-feed"
              // Cap the feed height so the Voice Commands panel — including the
              // task state — stays above the fold on a 1080p viewport instead
              // of being pushed off-screen by a tall 4:3 camera image.
              style={{
                width: "100%",
                maxWidth: "100%",
                maxHeight: "52vh",
                objectFit: "contain",
                borderRadius: "12px",
                display: "block",
                background: "#0f172a",
                margin: "0 auto",
              }}
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            <h3 style={{ margin: "0 0 8px", fontSize: "18px" }}>Voice Commands</h3>
            {banner && (
              <span
                data-testid="task-state"
                data-state={taskState}
                style={{
                  padding: "6px 12px",
                  borderRadius: "999px",
                  background: banner.bg,
                  color: banner.fg,
                  fontSize: "13px",
                  fontWeight: 700,
                }}
              >
                {banner.label}
              </span>
            )}
          </div>
          <p data-testid="transcript" style={{ margin: 0, color: "#475569", lineHeight: 1.5 }}>{speechTranscript}</p>
          <p data-testid="robot-command" style={{ margin: "8px 0 0", color: "#2563eb", lineHeight: 1.5, fontWeight: 600, whiteSpace: "pre-wrap" }}>
            Robot Command: {robotCommand}
          </p>
          {stepTotal > 0 && (
            <p data-testid="step-progress" style={{ margin: "8px 0 0", color: "#475569", lineHeight: 1.5 }}>
              Step {stepIndex} of {stepTotal}
              {stepLabel ? ` — ${stepLabel}` : ""}
            </p>
          )}
          {taskMessage && (
            <p data-testid="task-message" style={{ margin: "8px 0 0", color: taskState === "failed" ? "#991b1b" : "#166534", lineHeight: 1.5 }}>
              {taskMessage}
            </p>
          )}
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

function App() {

  const detect = async () => {
    console.log("Sending start detect request...");
    const request = {
      source: 0,
      confidence: 0.5
    }

    const response = await fetch("http://127.0.0.1:8000/api/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    const data = await response.json();

    console.log(data);
  };

  const stopDetect = async () => {
    console.log("Sending stop detect request");
    const response = await fetch("http://127.0.0.1:8000/api/stopDetect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    const data = await response.json();
    console.log(data);

  }



  return (
    <div>
      <button onClick={detect}>Start Detect</button>
      <button onClick={stopDetect}>Stop Detect</button>
    </div>
  );
}

export default App;
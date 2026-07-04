function App() {

  const detect = async () => {
    console.log("Sending request...");
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

  return (
    <button onClick={detect}>Detect</button>
  );
}

export default App;
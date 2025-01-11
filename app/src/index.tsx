import ReactDOM from "react-dom/client";
import React, { useEffect, useRef, useState } from 'react';
import "./index.css";

function App() {
    const socket = useRef(null)
    const [isReady, setIsReady] = useState(false);
    const [val, setVal] = useState(null);

    const [file, setFile] = useState(null);

    // setup web socket
    useEffect(() => {
        const ws = new WebSocket(`ws://localhost:8000/ws`);  // Backend WebSocket URL

        ws.onopen = () => {
            console.log("socket opened")
            setIsReady(true)
        }

        ws.onclose = () => {
            console.log("socket closed")
            setIsReady(false)
        }

        ws.onmessage = (event) => {
            console.log("Message from backend:", event.data);
            setVal(event.data)
        };

        socket.current = ws;

        return () => {
            ws.close();  // Clean up the socket on component unmount
        };
    }, []);

    // Handle file selection
    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    }

        // Upload the file and send via WebSocket
    const handleFileUpload = async () => {
        if (!file) {
            alert("Please select a file first");
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            // Once file is read, send it through the WebSocket
            const fileData = reader.result;
            // Check if WebSocket is open and send the file data as binary
            if (socket && socket.current.readyState === WebSocket.OPEN) {
                socket.current.send(fileData);  // Send the audio file as binary data
                console.log("File sent successfully");
                // Close the WebSocket after sending the data (otherwise, it waits forever)
                // socket.close();
            } else {
                console.error("WebSocket is not open");
            }
        };

        reader.onerror = (error) => {
            console.error("FileReader error:", error);
        };

        reader.readAsArrayBuffer(file);
    }

    return (
      <div className="app-div">
        <h1>Real-Time Note Detection</h1>
        <div>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleFileUpload}>Upload</button>
        </div>
      </div>
    );
  }


const root = ReactDOM.createRoot(
    document.getElementById("root") as HTMLElement
);

root.render(
    <App />
);

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Optional: import CSS if you have one
// import "./index.css";

const container = document.getElementById("root");
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
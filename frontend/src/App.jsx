import React from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import UploadPage from "./components/uploadPage";

function LandingPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-100 to-purple-200 px-4">
      <h1 className="text-5xl font-extrabold text-blue-800 mb-4 text-center drop-shadow-lg">
        Chat Analyzer
      </h1>
      <p className="text-xl text-gray-700 mb-8 text-center max-w-xl">
        Unlock insights from your Instagram conversations! Upload your chat and
        let our AI-powered analyzer reveal patterns, trends, and more using
        advanced LLMs.
      </p>
      <button
        className="px-8 py-3 bg-purple-600 text-white rounded-lg shadow-lg hover:bg-purple-700 transition text-lg font-semibold"
        onClick={() => navigate("/upload")}
      >
        Try Now
      </button>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/upload" element={<UploadPage />} />
    </Routes>
  );
}

export default App;

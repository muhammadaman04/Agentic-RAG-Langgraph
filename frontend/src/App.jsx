import React from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import ChatbotPage from "./components/ChatbotPage";

function LandingPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 px-4">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6">
          IntelliRAG
        </h1>
        <p className="text-2xl text-gray-600 font-medium mb-4">
          Intelligent Retrieval-Augmented Generation
        </p>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          Transform your documents into intelligent conversations. Upload your
          files and experience the power of AI-driven document understanding and
          retrieval.
        </p>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-8 mb-12 max-w-4xl">
        <div className="text-center p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Smart Document Processing
          </h3>
          <p className="text-gray-600">
            Upload multiple document formats and let our AI understand and index
            your content.
          </p>
        </div>

        <div className="text-center p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow">
          <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-purple-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Intelligent Chatbot
          </h3>
          <p className="text-gray-600">
            Ask questions about your documents and get accurate, contextual
            answers instantly.
          </p>
        </div>

        <div className="text-center p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Lightning Fast
          </h3>
          <p className="text-gray-600">
            Optimized retrieval system that provides relevant answers in
            milliseconds.
          </p>
        </div>
      </div>

      {/* CTA Button */}
      <button
        className="px-10 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105 text-xl font-semibold"
        onClick={() => navigate("/upload")}
      >
        Get Started
      </button>

      {/* Footer */}
      <div className="mt-16 text-center text-gray-500">
        <p>Powered by advanced AI and intelligent document processing</p>
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/upload" element={<UploadPage />} />
      <Route path="/chatbot" element={<ChatbotPage />} />
    </Routes>
  );
}

export default App;

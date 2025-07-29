import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ChatbotPage = () => {
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (location.state?.sessionId && location.state?.uploadedFiles) {
      setSessionId(location.state.sessionId);
      setUploadedFiles(location.state.uploadedFiles);
      setIsLoading(false);
    } else {
      // Redirect to upload page if no session data
      navigate("/upload");
    }
  }, [location.state, navigate]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading chatbot...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                IntelliRAG Chatbot
              </h1>
              <p className="text-gray-600">
                Session ID:{" "}
                <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">
                  {sessionId}
                </span>
              </p>
            </div>
            <button
              onClick={() => navigate("/upload")}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
            >
              Upload New Documents
            </button>
          </div>
        </div>

        {/* Uploaded Files Summary */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Uploaded Documents
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                    <svg
                      className="w-5 h-5 text-green-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-gray-800 text-sm">
                      {file.original_name}
                    </p>
                    <p className="text-xs text-gray-500">{file.size} bytes</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Chatbot Interface Placeholder */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="text-center py-12">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg
                className="w-12 h-12 text-blue-600"
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
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Chatbot Interface
            </h2>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Your documents have been successfully uploaded and processed. The
              chatbot interface will be implemented here to allow you to ask
              questions about your documents.
            </p>
            <div className="bg-gray-50 rounded-lg p-6 max-w-2xl mx-auto">
              <p className="text-sm text-gray-500 mb-4">
                <strong>Ready for implementation:</strong>
              </p>
              <ul className="text-sm text-gray-600 space-y-2 text-left">
                <li>• Chat message interface with user input</li>
                <li>• Message history display</li>
                <li>• Integration with your RAG backend</li>
                <li>• Real-time document querying</li>
                <li>• Response streaming and formatting</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatbotPage;

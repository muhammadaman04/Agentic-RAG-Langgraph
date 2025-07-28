import React, { useState } from "react";

function UploadPage() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setMessage("Please select a JSON file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setMessage("File uploaded successfully!");
      } else {
        setMessage("Upload failed. Please try again.");
      }
    } catch (err) {
      setMessage("Error uploading file.");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-100 to-purple-200 px-4">
      <h2 className="text-3xl font-bold text-blue-800 mb-4">
        Upload Instagram Chat
      </h2>
      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-lg shadow-md flex flex-col items-center w-full max-w-md"
      >
        <input
          type="file"
          accept="application/json"
          onChange={handleFileChange}
          className="mb-4"
        />
        <button
          type="submit"
          className="px-6 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition font-semibold"
        >
          Upload
        </button>
        {message && <p className="mt-4 text-center text-red-600">{message}</p>}
      </form>
    </div>
  );
}

export default UploadPage;

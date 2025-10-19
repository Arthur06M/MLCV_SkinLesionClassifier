import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import ImageDisplay from './components/ImageDisplay';
import PredictionResults from './components/PredictionResults';
import { predictSkinLesion, checkHealth } from './api/skinLesionAPI';

function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImagePreview, setOriginalImagePreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on mount
  useEffect(() => {
    checkHealth()
      .then(health => {
        setApiStatus(health.status === 'ok' ? 'online' : 'offline');
      })
      .catch(() => setApiStatus('offline'));
  }, []);

  // Handle image selection
  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    setPredictions(null);
    setGradcamImage(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  // Handle prediction submission
  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const result = await predictSkinLesion(selectedFile);

    setLoading(false);

    if (result.success) {
      setPredictions(result.data.predictions);
      setGradcamImage(result.data.gradcam);
    } else {
      setError(result.error);
      setPredictions(null);
      setGradcamImage(null);
    }
  };

  // Reset everything
  const handleReset = () => {
    setSelectedFile(null);
    setOriginalImagePreview(null);
    setPredictions(null);
    setGradcamImage(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Skin Lesion Classifier
              </h1>
              <p className="mt-1 text-sm text-gray-500">
                AI-powered skin lesion analysis using EfficientNet-B0
              </p>
            </div>
            
            {/* API Status Indicator */}
            <div className="flex items-center">
              <div className={`
                h-3 w-3 rounded-full mr-2
                ${apiStatus === 'online' ? 'bg-green-500' : 
                  apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'}
              `} />
              <span className="text-sm text-gray-600">
                API {apiStatus}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        
        {/* Upload Section */}
        <div className="mb-8">
          <ImageUpload onImageSelect={handleImageSelect} />
          
          {/* Action Buttons */}
          {selectedFile && (
            <div className="flex justify-center gap-4 mt-6">
              <button
                onClick={handlePredict}
                disabled={loading}
                className={`
                  px-6 py-3 rounded-lg font-semibold text-white
                  transition-all duration-200 transform hover:scale-105
                  ${loading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700 shadow-lg'
                  }
                `}
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
              
              <button
                onClick={handleReset}
                disabled={loading}
                className="
                  px-6 py-3 rounded-lg font-semibold
                  bg-white text-gray-700 border-2 border-gray-300
                  hover:bg-gray-50 transition-all duration-200
                  disabled:opacity-50 disabled:cursor-not-allowed
                "
              >
                Reset
              </button>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-8">
            <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg
                    className="h-5 w-5 text-red-400"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        <PredictionResults predictions={predictions} loading={loading} />
        
        {/* Image Display Section */}
        <ImageDisplay 
          originalImage={originalImagePreview}
          gradcamImage={gradcamImage}
        />

        {/* Footer Info */}
        {!predictions && !loading && (
          <div className="mt-12 text-center">
            <div className="inline-block bg-white rounded-lg shadow-md p-6 max-w-2xl">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                How it works
              </h3>
              <div className="text-left text-sm text-gray-600 space-y-2">
                <p>• Upload a clear image of a skin lesion</p>
                <p>• The AI model analyzes the image using deep learning</p>
                <p>• Receive predictions for 7 different lesion types</p>
                <p>• View a heatmap showing where the model focused</p>
                <p className="text-xs text-gray-500 mt-4 pt-4 border-t">
                  Model: EfficientNet-B0 trained on HAM10000 dataset (10,015 images)
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 pb-8 text-center text-sm text-gray-500">
        <p>Built with React + FastAPI + PyTorch</p>
        <p className="mt-1">
          Created by Arthur • Portfolio Project
        </p>
      </footer>
    </div>
  );
}

export default App;
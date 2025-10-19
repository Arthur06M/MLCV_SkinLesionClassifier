import React from 'react';

const PredictionResults = ({ predictions, loading }) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Analyzing image...</span>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return null;
  }

  // Color coding based on class
  const getClassColor = (className) => {
    const colors = {
      'mel': 'bg-red-500',      // Melanoma - dangerous
      'bcc': 'bg-orange-500',   // Basal cell carcinoma
      'akiec': 'bg-yellow-500', // Actinic keratoses
      'bkl': 'bg-green-500',    // Benign keratosis
      'nv': 'bg-blue-500',      // Melanocytic nevi (common mole)
      'df': 'bg-purple-500',    // Dermatofibroma
      'vasc': 'bg-pink-500',    // Vascular lesion
    };
    return colors[className] || 'bg-gray-500';
  };

  const getClassFullName = (className) => {
    const names = {
      'akiec': 'Actinic Keratoses',
      'bcc': 'Basal Cell Carcinoma',
      'bkl': 'Benign Keratosis',
      'df': 'Dermatofibroma',
      'mel': 'Melanoma',
      'nv': 'Melanocytic Nevi (Mole)',
      'vasc': 'Vascular Lesion',
    };
    return names[className] || className;
  };

  return (
    <div className="w-full max-w-2xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Prediction Results</h2>
      
      {/* Top prediction highlight */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500">Most Likely Diagnosis</p>
            <p className="text-xl font-semibold text-gray-800">
              {getClassFullName(predictions[0].class)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-blue-600">
              {(predictions[0].probability * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Confidence</p>
          </div>
        </div>
      </div>

      {/* All predictions with bars */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">
          All Classifications
        </h3>
        <div className="space-y-3">
          {predictions.map((pred, index) => (
            <div key={pred.class} className="relative">
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm font-medium text-gray-700">
                  {index + 1}. {getClassFullName(pred.class)}
                </span>
                <span className="text-sm font-semibold text-gray-600">
                  {(pred.probability * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className={`h-2.5 rounded-full ${getClassColor(pred.class)} transition-all duration-500`}
                  style={{ width: `${pred.probability * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-yellow-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-yellow-700">
              <strong>Medical Disclaimer:</strong> This is an educational demonstration only. 
              Always consult a qualified dermatologist for actual medical advice and diagnosis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
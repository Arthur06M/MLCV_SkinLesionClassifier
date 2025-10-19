import React from 'react';

const ImageDisplay = ({ originalImage, gradcamImage }) => {
  if (!originalImage) {
    return null;
  }

  return (
    <div className="w-full max-w-4xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Visual Analysis</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Original Image */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">
            Original Image
          </h3>
          <img
            src={originalImage}
            alt="Original skin lesion"
            className="w-full h-auto rounded-lg"
          />
        </div>

        {/* Grad-CAM Heatmap */}
        {gradcamImage && (
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">
              Model Focus (Grad-CAM)
            </h3>
            <img
              src={gradcamImage}
              alt="Grad-CAM heatmap"
              className="w-full h-auto rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-2">
              Red areas indicate where the model focused its attention
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageDisplay;
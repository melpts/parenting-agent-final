import React, { useState, useEffect } from 'react';

const ParentingSupportBot = () => {
  const [userData, setUserData] = useState({
    parentName: '',
    childName: '', 
    childAge: '3-5 years',
    situation: ''
  });

  const [currentFeature, setCurrentFeature] = useState('info');

  // Check for pre-populated data from Qualtrics
  useEffect(() => {
    if (window.Qualtrics) {
      const embeddedData = Qualtrics.SurveyEngine.getEmbeddedData();
      if (embeddedData.parentName) {
        setUserData({
          parentName: embeddedData.parentName,
          childName: embeddedData.childName,
          childAge: embeddedData.childAge,
          situation: embeddedData.situation
        });
      }
    }
  }, []);

  // Save user input to Qualtrics Embedded Data
  useEffect(() => {
    if (window.Qualtrics) {
      Object.entries(userData).forEach(([key, value]) => {
        Qualtrics.SurveyEngine.setEmbeddedData(key, value);
      });
    }
  }, [userData]);

  const handleSubmit = (event) => {
    event.preventDefault();
    setCurrentFeature('advice');
    if (window.Qualtrics) {
      Qualtrics.SurveyEngine.setEmbeddedData('currentFeature', 'advice');
    }
  };

  const renderInfo = () => (
    <div className="max-w-lg mx-auto p-6 bg-gray-800 rounded-lg">
      <h1 className="text-3xl font-bold text-white mb-8">Welcome to Parenting Support Bot</h1>
      <form onSubmit={handleSubmit}>
        <div className="space-y-6">
          <div>
            <label className="block text-white mb-2">Your Name</label>
            <input
              type="text"
              className="w-full p-2 bg-gray-700 text-white rounded"
              value={userData.parentName}
              onChange={(e) => setUserData({...userData, parentName: e.target.value})}
              required
            />
          </div>
          <div>
          <label className="block text-white mb-2">Child's Name</label>
            <input
              type="text"
              className="w-full p-2 bg-gray-700 text-white rounded"
              value={userData.childName}
              onChange={(e) => setUserData({...userData, childName: e.target.value})}
              required
            />
          </div>
          <div>
          <label className="block text-white mb-2">Child's Age Range</label>
            <select
              className="w-full p-2 bg-gray-700 text-white rounded"
              value={userData.childAge}
              onChange={(e) => setUserData({...userData, childAge: e.target.value})}
            >
              <option>3-5 years</option>
              <option>6-9 years</option>
              <option>10-12 years</option>
            </select>
          </div>
          <div>
          <label className="block text-white mb-2">Describe the situation you'd like help with</label>
            <textarea
              className="w-full p-2 bg-gray-700 text-white rounded"
              value={userData.situation}
              onChange={(e) => setUserData({...userData, situation: e.target.value})}
              required
            />
          </div>
          <button type="submit" className="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700">
            Start
          </button>
        </div>
      </form>
    </div>
  );

  const renderFeature = () => {
    switch(currentFeature) {
      case 'advice':
        return (
          <div className="max-w-lg mx-auto p-6 bg-gray-800 rounded-lg text-white">
            <h2 className="text-2xl mb-4">Parenting Advice</h2>
            <div className="mb-6">
              {/* Advice content would go here */}
            </div>
            {/* Feature-specific questions */}
          </div>
        );
      // Add cases for other features
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      {currentFeature === 'info' ? renderInfo() : renderFeature()}
    </div>
  );
};

export default ParentingSupportBot;

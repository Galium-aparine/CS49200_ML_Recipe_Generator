import React, { useState } from 'react';
import axios from 'axios';

const RecipeGenerator = () => {
  const [ingredients, setIngredients] = useState("");
  const [allergies, setAllergies] = useState("");
  const [cuisine, setCuisine] = useState("Any");
  const [maxTime, setMaxTime] = useState(60);
  const [recipe, setRecipe] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [detectedIngredients, setDetectedIngredients] = useState([]);
  const [image, setImage] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [showLowConfidence, setShowLowConfidence] = useState(true);


  // Handle image upload
  const handleImageUpload = async (file) => {
    setImage(URL.createObjectURL(file)); // Display uploaded image
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/detect-ingredients', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      console.log("Detected Ingredients:", response.data.ingredients);
      setDetectedIngredients(response.data.ingredients);
      setBoundingBoxes(response.data.ingredients.map(item => ({
        ...item,
        color: item.confidence > 0.5 ? 'green' : 'red' // Color high-confidence green, low red
      })));

      setImageURL(`http://localhost:8000${response.data.output_image_url}`); // Show modified image
      setIngredients(response.data.ingredients.map(item => item.class_name).join(', ')); 

    } catch (err) {
      console.error("Image recognition failed", err);
      setError("Failed to detect ingredients from image.");
    }
  };

  // Handle recipe generation
  const handleGenerate = async () => {
    if (!ingredients.trim()) {
      setError("Please enter some ingredients.");
      return;
    }
    setLoading(true);
    setError("");

    try {
      const response = await axios.post('http://localhost:8000/generate', {
        ingredients,
        allergies,
        cuisine,
        max_time: maxTime
      });

      setRecipe(response.data.ai_recipe);
    } catch (err) {
      console.error(err);
      setError("Failed to generate recipe. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="generator-container">
      <h1>Smart Recipe Generator</h1>

      {/* Image Upload */}
      <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e.target.files[0])} className="file-upload" />
      
      {/* Image Preview */}
      {image && (
        <div className="image-container">
          <img src={imageURL || image} alt="Uploaded" className="uploaded-image" />
          {/* Bounding Boxes */}
          {boundingBoxes
            .filter(item => showLowConfidence || item.confidence > 0.5)
            .map((box, index) => (
              <div
                key={index}
                className="bounding-box"
                style={{
                  borderColor: box.color,
                  top: `${box.bbox[1]}px`,
                  left: `${box.bbox[0]}px`,
                  width: `${box.bbox[2] - box.bbox[0]}px`,
                  height: `${box.bbox[3] - box.bbox[1]}px`
                }}
              >
                {box.class_name} ({(box.confidence * 100).toFixed(1)}%)
              </div>
          ))}
        </div>
      )}

      {/* Detected Ingredients List */}
      {detectedIngredients.length > 0 && (
        <div className="detected-container">
          <h3>Detected Ingredients:</h3>
          <ul>
            {detectedIngredients.map((item, index) => (
              <li key={index} style={{ color: item.confidence > 0.5 ? 'green' : 'red' }}>
                {item.class_name} ({(item.confidence * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
          <button className="toggle-button" onClick={() => setShowLowConfidence(!showLowConfidence)}>
            {showLowConfidence ? "Hide Low Confidence" : "Show All"}
          </button>
        </div>
      )}

      {/* Ingredients Input */}
      <textarea
        rows="4"
        value={ingredients}
        onChange={(e) => setIngredients(e.target.value)}
        placeholder="List ingredients here, e.g., chicken, rice, onion..."
        className="textarea"
      />

      <input type="text" value={allergies} onChange={(e) => setAllergies(e.target.value)} placeholder="Allergies (optional)..." className="textarea" />
      <select value={cuisine} onChange={(e) => setCuisine(e.target.value)} className="textarea">
        <option value="Any">Any</option>
        <option value="Indian">Indian</option>
        <option value="Italian">Italian</option>
        <option value="Mexican">Mexican</option>
      </select>
      <input type="number" value={maxTime} onChange={(e) => setMaxTime(e.target.value)} min="10" max="120" className="textarea" placeholder="Max Cooking Time (minutes)" />

      {/* Generate Button */}
      <button onClick={handleGenerate} className="button">
        {loading ? "Generating..." : "Generate Recipe"}
      </button>

      {/* Error Message */}
      {error && <p className="error-message">{error}</p>}

      {/* Recipe Result */}
      {recipe && (
        <div className="recipe-box">
          <h2>Generated Recipe:</h2>
          <pre>{recipe}</pre>
        </div>
      )}
    </div>
  );
};

export default RecipeGenerator;

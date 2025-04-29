import RecipeGenerator from './RecipeGenerator';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <div className="logo-title">
        <img src="/logo.png" alt="App Logo" className="logo-small" />
        <h1>🍳 Recipe Generator</h1>
      </div>
      <RecipeGenerator />
    </div>
  );
}

export default App;

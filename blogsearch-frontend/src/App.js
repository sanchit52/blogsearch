import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState("");
  const [type, setType] = useState(""); // 'personal' or 'non_personal'
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    if (!query) return;

    setLoading(true);
    try {
      const res = await fetch(
        http://localhost:5000/search?q=${encodeURIComponent(query)}&type=${type}
      );
      const data = await res.json();

      // Handle both array or object with results key
      const resultData = Array.isArray(data) ? data : data.results || [];
      setResults(resultData);
    } catch (err) {
      console.error("Error fetching results:", err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Authentic Blog Search</h1>

      <div className="search-controls">
        <input
          type="text"
          value={query}
          placeholder="Search blogs..."
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="">All</option>
          <option value="personal">Personal</option>
          <option value="non_personal">Non-Personal</option>
        </select>
        <button onClick={search}>Search</button>
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : results.length === 0 ? (
        <p>No results found.</p>
      ) : (
        <ul className="results">
          {results.map((item, idx) => (
            <li key={idx}>
              <a href={item.url} target="_blank" rel="noopener noreferrer">
                <strong>{item.title}</strong>
              </a>
              <p>{item.snippet || item.content}</p>
              <small>
                Type: <strong>{item.type}</strong> | Score: {item.score}
              </small>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
 

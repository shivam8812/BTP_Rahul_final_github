import logo from "./logo.svg"
import "semantic-ui-css/semantic.min.css"
import { Dropdown, Button } from "semantic-ui-react"
import { Header } from "semantic-ui-react"
import "./App.css"
import { useState } from "react"
import Home from "./Home"
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom"
import Weather from "./Weather"
import Cyclone from "./Cyclone"

function App() {
  const [weather, setWeather] = useState([])
  const [cyclone, setCyclone] = useState([])

  return (
    <div className="cont">
      <Router>
        <Routes>
          <Route path="/weather" element={<Weather weather={weather} />} />
          <Route path="/cyclone" element={<Cyclone cyclone={cyclone} />} />
          <Route path="/" element={<Home weather={weather} cyclone={cyclone} setWeather={setWeather} setCyclone={setCyclone} />} />
        </Routes>
      </Router>
    </div>
  )
}

export default App

import logo from "./logo.svg"
import "semantic-ui-css/semantic.min.css"
import { Dropdown, Button } from "semantic-ui-react"
import { Header } from "semantic-ui-react"
import "./App.css"
import { useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"
const areas = [
  {
    key: "Lakshdweep",
    lat: "10.5",
    lon: "72.3",
  },
  {
    key: "Arabian_sea",
    lat: "19.2",
    lon: "71.4",
  },
  {
    key: "Tanintharyi_Region",
    lat: "13.2",
    lon: "97.6",
  },
  {
    key: "Andhra",
    lat: "13.1",
    lon: "79.8",
  },
  {
    key: "Odisha",
    lat: "21.2",
    lon: "87.1",
  },
  {
    key: "Pottangi",
    lat: "18.4",
    lon: "83",
  },
  {
    key: "Telanghana",
    lat: "17.6",
    lon: "79.5",
  },
  {
    key: "Bengal",
    lat: "21.9",
    lon: "88.4",
  },
]

function Home(props) {
  const [areaSelected, setAreaSelected] = useState("Select Area")
  const [lat, setLat] = useState("")
  const [lon, setLon] = useState("")

  const setParams = area => {
    setAreaSelected(area.key)
    setLat(area.lat)
    setLon(area.lon)
  }

  const navigate = useNavigate()
  const cheakWeather = () => {
    console.log("in cw")
    axios
      .post("http://127.0.0.1:5000/weather", {
        area: areaSelected,
        lat: lat,
        lon: lon,
      })
      .then(res => {
        console.log(res)
        props.setWeather(res)
        navigate("/weather")
      })
  }

  const predictCyclone = () => {
    axios
      .post("http://127.0.0.1:5000/cyclone", {
        area: areaSelected,
        lat: lat,
        lon: lon,
      })
      .then(res => {
        console.log(res)
        props.setCyclone(res)
        navigate("/cyclone")
      })
    // navigate("/cyclone")
  }

  return (
    <div className="App">
      <Header as="h1" className="textwhite">
        Weather and Cyclone Prediction
      </Header>
      {/* <Dropdown
        placeholder="Select Area"
        fluid
        selection
        options={friendOptions}
        onClick={}
        className="dd"
      /> */}
      <Dropdown selection className={"dd"} placeholder={areaSelected}>
        <Dropdown.Menu className="dd-menu">
          {areas.map(area => (
            <Dropdown.Item
              className="ditem"
              value={area.key}
              onClick={() => setParams(area)}
            >
              {area.key}
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown>
      <div className="bu">
        <Button className="b1 textwhite" onClick={() => cheakWeather()}>
          Check Weather
        </Button>
        <Button className="b2" onClick={() => predictCyclone()}>
          Predict Cyclone
        </Button>
      </div>
    </div>
  )
}

export default Home

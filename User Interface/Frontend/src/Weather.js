import { useEffect, useState } from "react"
import "semantic-ui-css/semantic.min.css"
import "./App.css"
import { Table } from "semantic-ui-react"
import { useNavigate } from "react-router-dom"

function Weather(props) {
  let navigate = useNavigate()
  console.log(props.weather)
  const getdate = i => {
    const today = new Date()
    const tomorrow = new Date(today)
    tomorrow.setDate(today.getDate() + i-2)

    const dd = String(tomorrow.getDate()).padStart(2, "0")
    const mm = String(tomorrow.getMonth() + 1).padStart(2, "0") // January is 0!
    const yyyy = tomorrow.getFullYear()
    const formattedDate = dd + "-" + mm + "-" + yyyy
    return formattedDate
  }
  return (
    <div className="cc">
      <div className="container">
        <h2 className="">
          The weather parameters of {props.weather.data[1]} region for the next day
        </h2>
        <Table color={"blue"} definition>
          <Table.Header>
            <Table.Row>
              <Table.HeaderCell />
              <Table.HeaderCell>
                Max Temperature
                <br />
                (in &deg;C)
              </Table.HeaderCell>
              <Table.HeaderCell>
                Min Temperature
                <br />
                (in &deg;C)
              </Table.HeaderCell>
              <Table.HeaderCell>
              Relative Humidity
              </Table.HeaderCell>
              <Table.HeaderCell>
              Precipitation
                <br />
                (in mm/day)
              </Table.HeaderCell>
              <Table.HeaderCell>
                Surface Pressure
                <br />
                (in kPa)
              </Table.HeaderCell>
              <Table.HeaderCell>
                Max Wind Speed
                <br />
                (in m/s)
              </Table.HeaderCell>
              <Table.HeaderCell>
                Min Wind Speed
                <br />
                (in m/s)
              </Table.HeaderCell>
            </Table.Row>
          </Table.Header>

          <Table.Body>
            {props.weather.data[0].map((w, i) => (
              <Table.Row>
                <Table.Cell>{getdate(i)}</Table.Cell>
                <Table.Cell>{w[0].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[1].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[2].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[3].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[4].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[5].toFixed(4)}</Table.Cell>
                <Table.Cell>{w[6].toFixed(4)}</Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </div>
  )
}

export default Weather

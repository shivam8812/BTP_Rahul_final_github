import { useEffect, useState } from "react"
import "semantic-ui-css/semantic.min.css"
import "./App.css"
import { Table } from "semantic-ui-react"
import { useNavigate } from "react-router-dom"

function Cyclone(props) {
  let navigate = useNavigate()
  console.log(props.cyclone)
  const getdate = i => {
    const today = new Date()
    const tomorrow = new Date(today)
    tomorrow.setDate(today.getDate() + i - 2)

    const dd = String(tomorrow.getDate()).padStart(2, "0")
    const mm = String(tomorrow.getMonth() + 1).padStart(2, "0") // January is 0!
    const yyyy = tomorrow.getFullYear()
    const formattedDate = dd + "-" + mm + "-" + yyyy
    return formattedDate
  }
  const cycloneMapping = {
    0: 'D',
    1: 'DD',
    2: 'CS',
    3: 'SCS',
    4: 'VSCS',
    5: 'ESCS',
    6: 'SuCS',
    7: 'No Cyclone'
  };
  return (
    <div className="cc">
      <div className="container">
        <h2 className="">
          The cyclone severity of {props.cyclone.data[1]} region for the next
          day
        </h2>
        <Table color={"blue"} definition>
          <Table.Header>
            <Table.Row>
              <Table.HeaderCell />
              <Table.HeaderCell>Grade of Cyclone</Table.HeaderCell>
            </Table.Row>
          </Table.Header>

          <Table.Body>
          <Table.Row>
                <Table.Cell>{getdate(0)}</Table.Cell>
                <Table.Cell>{cycloneMapping[props.cyclone.data[2]]}</Table.Cell>
                
              </Table.Row>
          </Table.Body>
        </Table>
      </div>
    </div>
  )
}

export default Cyclone

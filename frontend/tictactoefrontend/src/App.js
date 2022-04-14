import logo from './logo.svg';
import './App.css';
import React, {useEffect, useState,useCallback} from 'react';
import { SwishSpinner } from "react-spinners-kit";
import Tree from 'react-d3-tree';
import ToggleButton from 'react-toggle-button'


export const useCenteredTree = (defaultTranslate = { x: 0, y: 0 }) => {
  const [translate, setTranslate] = useState(defaultTranslate);
  const [dimensions, setDimensions] = useState();
  const containerRef = useCallback((containerElem) => {
    if (containerElem !== null) {
      const { width, height } = containerElem.getBoundingClientRect();
      // console.log(height)
      setDimensions({ width, height });
      // setTranslate({ x: 15, y: height / 2 });
      setTranslate({ x: width/2, y: 15 });
    }
  }, []);
  return [dimensions, translate, containerRef];
};


function App(){
  
  const [dimensions, translate, containerRef] = useCenteredTree();
  const [boardState,setBoardState] = useState(["","","","","","","","",""])
  const [stepCount,setStepCount] = useState(0)
  const [botPlayer,setBotPlayer] = useState(-1)
  const [winner,setWinner] = useState(null)
  const [lastAction,setLastAction] = useState(-1)
  const [sessionId,setSessionId] = useState(String(Math.floor(Math.random() * 101)))
  const [history,setHistory] = useState({})
  const [disable,setDisable] = useState(false)
  const [loading,setLoading] = useState(false)
  const [monteCarloTree,setMonteCarloTree] = useState({name:"_", attributes:{nodeBoardState:[0,0,0,0,0,0,0,0,0],pl:1,visits:0},
    children: []})
  const [showTree,setShowTree] = useState(false)
  

  
  
  const getActionFromBot = async (action) =>{
    var formdata = new FormData();
    formdata.append("botPlayer",botPlayer)
    formdata.append("records_storage",JSON.stringify(history))
    var requestOptions = {
      method: 'POST',
      body: formdata,
      redirect: 'follow'
    };
    var response = await fetch("https://tictactoealphazero.herokuapp.com//play", requestOptions)
    response = await response.json()
    // console.log(response)
    // console.log(history)
    setMonteCarloTree(response["tree"])
    setHistory(response["records"])
    
    
    return response["next_action"]
    
  }

  const appendToHistory = (step,action,player) => {
    var boardEncoded = boardState.map((val)=>{
                switch (val){
                  case "X":
                    return 1
                    break;
                  case "O":
                    return -1
                    break;
                  default:
                    return 0
                          }
              })
    var record = {
                  "state":boardEncoded,
                  "player":player,
                  "action":action
                  }
    var history_copy = {...history}
    history_copy[step] = record
    setHistory(history_copy)
  }

  
  useEffect(()=>{
    reset(botPlayer)
    
  },[])
  useEffect(()=>{
    async function fetchData(){
    setLoading(true)
    // console.log(botPlayer,stepCount)
    if (botPlayer===1){
      if (stepCount%2===0){
        // console.log("bot playing")
        var action = await getActionFromBot(lastAction)
        // console.log(action)
        play_game(action)
      }
    }
    if (botPlayer===-1){
      if (stepCount%2!==0){
        // console.log("bot playing")
        var action = await getActionFromBot(lastAction)
        // console.log(action)
        play_game(action)
      }
    }
    setLoading(false)
  }
  fetchData();

  },[stepCount,botPlayer])
  

  const play_game = (index)=>{
    if (!disable){
    // console.log("action",index)
    setLastAction(index)
    if (stepCount%2===0){var currentPlayer=1;var mark="X"}
    else{var currentPlayer=-1;var mark="O"}
    appendToHistory(stepCount,index,currentPlayer)
    if (boardState[index]===""){
      boardState[index] = mark
      setStepCount(stepCount+1)
      // console.log("step count",stepCount)
      checkWinner(mark,currentPlayer)
    }
    // console.log(history)
  }
  }
  const reset = (player)=>{  
    setShowTree(false)
    setMonteCarloTree({name:"_", attributes:{nodeBoardState:[0,0,0,0,0,0,0,0,0],pl:1,visits:0},
    children: []}) 
      
    setBotPlayer(player);
    setBoardState(["","","","","","","","",""]);
    setStepCount(0);
    setWinner(null)
    setDisable(false)
    setHistory({})
  }
  const checkWinner = (currentTurn,currentPlayer) =>{
    // console.log("checking",currentTurn,boardState)
    if (boardState.filter((elem)=>elem==="").length===0){
      // console.log("draw")
      setWinner(0)
      setDisable(true)
    }
    else{
    var symbols = boardState
    var winningCombos = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    winningCombos.map((combo)=>{
      if (symbols[combo[0]]===currentTurn & symbols[combo[1]]===currentTurn & symbols[combo[2]]===currentTurn){
        setWinner(currentPlayer);
        // console.log("Winner ",currentPlayer)
        setDisable(true)
      }
    }) 
  }
  }
  
  
  const nodeDatum = {
    nodeBoardState:[0,1,0,0,0,0,-1,0,0],pl:1,visits:100
  }
  // console.log(nodeDatum)
  const rendernodeboard = ({nodeDatum,toggleNode})=>{
    // console.log(nodeDatum)
    var nodeboard = nodeDatum.attributes.nodeBoardState.map((val)=>{
      if (nodeDatum.attributes.pl===1){
      switch (val){
        case 1:
          return "X"
          break;
        case -1:
          return "O"
          break;
        default:
          return ""
                }}
    else{
      switch (val){
        case -1:
          return "X"
          break;
        case 1:
          return "O"
          break;
        default:
          return ""
                }
    }
    })
    // console.log(nodeboard)
    // console.log(monteCarloTree)
    return (<g>
    <rect className="nodeBoard"  onClick={toggleNode} ></rect>
    {nodeboard.map((val,index)=>{
      // console.log(val,index)
      return (<g key={index}><rect x="-10" y="-10" key={index} className="nodeSquare" transform={`translate(${(index%3)*8},${parseInt(index/3)*8})`} 
       ></rect>
      <text x="-9" y="-3" transform={`translate(${(index%3)*8},${parseInt(index/3)*8})`} fill={val==="X"?"rgb(144, 25, 25)":"rgb(5, 95, 52)"} font-family="Verdana" font-size="7" stroke="none">{val}</text></g>)
    })}
    <text className="nodeCaption" y="-15" x="-5">
      Visits: {nodeDatum.attributes.visits}
      </text>
      {/* <text className="nodeCaption" y="-25" x="-5">
      Average Value: {nodeDatum.attributes["Avg Value"]}
      </text> */}
    </g>)
  }
  
  
  return (
    <div className="master-container">
      <h1>Tic Tac Toe with AlphaZero</h1> 
    <div className="app-container">
      
      <div className="first-container">
      
      {botPlayer===1?<h5>Human: O Bot: X</h5>:<h5>Human: X Bot: O</h5>}
     
      {(winner===1) ? (<h3 style={{color:"rgb(144, 25, 25)"}}>Winner X</h3>) : ((winner===-1) ? (<h3 style={{color:"rgb(5, 95, 52)"}}>Winner O</h3>) : 
    ((winner===0) ? (<h3 style={{color:"rgb(142, 142, 142)"}}>Draw</h3>) : null))}


      <div className='board'>
        {boardState.map((val,index)=>{
          return <div key={index} className="square" 
          style={{color:val==="X"?"rgb(144, 25, 25)":"rgb(5, 95, 52)"}} onClick={()=>play_game(index)}>{val}</div>
        })}
        <div className="loadingspinner">
      <SwishSpinner  size={200} frontColor="#286090" loading={loading} /></div>
      </div>
      <div className="ButtonGroup">
        
      <button className="ResetButton" onClick={()=>{reset(1)}}>
        Reset Game (Human O)
      </button>
      <button className="ResetButton" onClick={()=>{reset(-1)}}>
        Reset Game (Human X)
      </button>
      </div>
      <div className="btnGrp">
      <p>
        Toggle to show trees.(Will slow down game!!)
        </p>
      <ToggleButton className="toggleButton"
        value={ showTree ||false }
        onToggle={(value) => {
          setShowTree(!value)
        }} />
       
     
</div>
</div>
        
        {showTree &&
         <div className="TreeContainer"  ref={containerRef}>
        <Tree
        data={monteCarloTree}
        renderCustomNodeElement={rendernodeboard}
        dimensions={dimensions}
        translate={translate}
        orientation="vertical"
        pathFunc="diagonal"
      />
       </div>}
       
      </div>
      </div>
  )

}

export default App;

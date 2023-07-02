import React from "react";
import "./Login.css";
import { Container } from "react-bootstrap";
function Login() {
  return (
    <Container>
      <div className="login">
        <font color="green">
          {" "}
          <h3>
            There's always another way to deal with Tough Agricultural Problems
          </h3>
        </font>
        <h1 color="Black">Login to you Account from here</h1>
        <font size="5" color="black">
          {" "}
          <label className="">Last Name: </label> <br />
        </font>
        <input
          className="DemoinputTxt"
          placeholder="Enter Your Last name"
          type="text"
          required
        ></input>
        <font size="5" color="#000000">
          {" "}
          <label className="">Password: </label> <br />
        </font>
        <input
          className="DemoinputTxt"
          color="Black"
          placeholder="Enter Password"
          type="password"
          required
        ></input>
        <br />
        <font color="green">
          <h5>Forgot Password?</h5>
        </font>
        <button className="DemoBtn">Login</button>
        <br /> <br /> <br /> <br />
     
      </div>
    </Container>
  );
}
export default Login;

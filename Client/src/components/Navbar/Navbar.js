import React, { useEffect, useState } from "react";
import { Container } from "react-bootstrap";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import { LinkContainer } from "react-router-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars } from "@fortawesome/free-solid-svg-icons";

const NavBar = () => {
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  useEffect(() => {
    // Retrieve query parameters from the URL
    const params = new URLSearchParams(window.location.search);
    const loggedInParam = params.get("loggedIn");
    const usernameParam = params.get("username");

    // Update the state based on the query parameters
    setLoggedIn(loggedInParam === "true");
    setUsername(usernameParam);
  }, []);

  function toggleNav() {
    document.getElementById("nav-content").classList.toggle("active");
  }
  return (
    <Navbar bg="dark" variant="dark" className="nav">
      <Container className="nav-container">
        <Navbar.Brand href="/" className="logo">
          Plant
        </Navbar.Brand>
        <Nav className="nav-content" id="nav-content">
          <div className="nav-links">
            <div className="pages-link">
              <LinkContainer to="/">
                <Nav.Link>Home</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/news">
                <Nav.Link>News</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/check">
                <Nav.Link>Check</Nav.Link>
              </LinkContainer>
            </div>
            <div className="nav-forms">
              {loggedIn ? (
                <>
                  <Nav.Link>{username}</Nav.Link>
                  <LinkContainer to="//localhost:5000/logout">
                    <Nav.Link>Logout</Nav.Link>
                  </LinkContainer>
                </>
              ) : (
                <>
                  <LinkContainer to="//localhost:5000/login">
                    <Nav.Link>Login</Nav.Link>
                  </LinkContainer>
                  <LinkContainer to="//localhost:5000/sign-up">
                    <Nav.Link>SignUp</Nav.Link>
                  </LinkContainer>
                </>
              )}
            </div>
          </div>
        </Nav>
        <FontAwesomeIcon
          icon={faBars}
          className="nav-button"
          id="nav-button"
          onClick={toggleNav}
        />
      </Container>
    </Navbar>
  );
};

export default NavBar;

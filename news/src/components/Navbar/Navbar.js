import React from "react";
import { Container } from "react-bootstrap";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import { LinkContainer } from "react-router-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars } from "@fortawesome/free-solid-svg-icons";

const NavBar = () => {
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
              <LinkContainer to="//localhost:5000/login">
                <Nav.Link>Login</Nav.Link>
              </LinkContainer>
              <LinkContainer to="//localhost:5000/sign-up">
                <Nav.Link>SignUp</Nav.Link>
              </LinkContainer>
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

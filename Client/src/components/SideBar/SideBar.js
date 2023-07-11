import "./SideBar.css";
import { FaSistrix } from "react-icons/fa";
import { LinkContainer } from "react-router-bootstrap";
const SideBar = (props) => {
  return (
    <div className="sideBarContent">
      <aside className="sideBox">
        <div className="searchInput">
          <input
            type="text"
            className="plantInput"
            placeholder="Enter plant name"
          ></input>
          <span
            className="searchIcon"
            onClick={() =>
              props.search(document.querySelector(".plantInput").value)
            }
          >
            <FaSistrix className="icon" />
          </span>
        </div>
      </aside>
      <aside className="sideBox">
        <h3 className="sideTitle">Recent Posts</h3>
        <ul>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Post title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Post title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Post title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Post title</p>
            </LinkContainer>
          </li>
        </ul>
      </aside>
      <aside className="sideBox">
        <h3 className="sideTitle">Categories</h3>
        <ul>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Categorie title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Categorie title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Categorie title</p>
            </LinkContainer>
          </li>
          <li>
            <LinkContainer to="/news">
              <p className="sideLink">Categorie title</p>
            </LinkContainer>
          </li>
        </ul>
      </aside>
    </div>
  );
};

export default SideBar;

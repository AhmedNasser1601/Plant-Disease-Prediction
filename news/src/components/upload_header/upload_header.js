import "./upload_header.css";

const Header = () => {
  function handleClick() {
    document.getElementById("uploadBox").classList.remove("hidden");
  }
  return (
    <div className="checkHeader">
      <div className="headerContent">
        <h1>Check Your Plant Health</h1>
        <p>
          Feel Free to capture your plant photo and upload it to get an
          immediate response about the whole current state of the plant
        </p>
        <button className="checkBtn" onClick={handleClick}>
          Check Now!
        </button>
      </div>
    </div>
  );
};

export default Header;

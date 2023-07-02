import "./NewHead.css";

const NewHead = ({ data }) => {
  const slideStyle = {
    backgroundImage: `url('${data["img"]}')`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center center",
    backgroundSize: "cover",
  };

  return (
    <div className="newHeadContent" style={slideStyle}>
      <div className="slideContent">
        <h2>{data["name"]}</h2>
        <p>{data["description"]}</p>
      </div>
    </div>
  );
};

export default NewHead;

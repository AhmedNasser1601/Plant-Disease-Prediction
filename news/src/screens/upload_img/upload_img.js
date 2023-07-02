import "./upload_img.css";
import Header from "../../components/upload_header/upload_header";
import { useEffect, useState } from "react";
import { FaFileUpload, FaTimesCircle } from "react-icons/fa";
import FormData from "form-data";
import Report from "../../components/Report/Report";

const UploadImg = () => {
  const [file, setFile] = useState();
  const [disease_name, set_disease_name] = useState("");
  function handleSelectingImage(event) {
    setFile(event.target.files[0]);
  }

  const predict_image = async (event) => {
    event.preventDefault();
    closeUpload();
    let formData = new FormData();
    formData.append("image", file);

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });
    try {
      const data = await response.json();
      if (data.disease) {
        set_disease_name(data.disease);
      } else {
        console.error("Invalid response data: ", data);
      }
    } catch (error) {
      console.error("Error parsing JSON response: ", error);
    }
  };

  useEffect(() => {
    if (disease_name !== "") {
      document.querySelector(".disease_report").classList.remove("hidden");
    }
  }, [disease_name]);

  function closeUpload() {
    document.getElementById("uploadBox").classList.add("hidden");
  }

  return (
    <div className="checkContent">
      <Header />
      <div id="uploadBox" className="uploadImage hidden">
        <FaTimesCircle className="closeButton" onClick={closeUpload} />
        <div className="uploadContent">
          <FaFileUpload className="fileIcon" />
          <p>Select an image</p>
          <form id="file-upload-form" class="uploader">
            <input
              id="file-upload"
              type="file"
              name="fileUpload"
              accept="image/*"
              onChange={handleSelectingImage}
            />
            <label for="file-upload">
              <span className="btn">Choose image</span>
            </label>
            <input className="btn" type="submit" onClick={predict_image} />
          </form>
        </div>
      </div>
      <Report disease_name={disease_name} set_disease_name={set_disease_name} image={file} />
    </div>
  );
};

export default UploadImg;

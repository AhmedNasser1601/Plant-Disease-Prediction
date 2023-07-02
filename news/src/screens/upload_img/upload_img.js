import "./upload_img.css";
import Header from "../../components/upload_header/upload_header";
import { useEffect, useState } from "react";
import { FaFileUpload, FaTimesCircle } from "react-icons/fa";
import FormData from "form-data";
import Report from "../../components/Report/Report";
import { Select, Text } from "@chakra-ui/react";

const UploadImg = () => {
  const [file, setFile] = useState();
  const [disease_name, set_disease_name] = useState("");
  const [model, set_model] = useState("VGG16");

  function handleSelectingImage(event) {
    setFile(event.target.files[0]);
  }

  const selectModel = (event) => {
    set_model(event.target.value);
  };

  const predict_image = async (event) => {
    event.preventDefault();
    closeUpload();
    let formData = new FormData();
    formData.append("image", file);
    formData.append("model", model);

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
          <div style={{ width: "75%", margin: "auto" }}>
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
              <div className="form_controlers">
                <label for="file-upload">
                  <span className="btn">Choose image</span>
                </label>
                <Select
                  onChange={selectModel}
                  size="md"
                  width="25"
                  borderColor="#454cad"
                >
                  <option value={"VGG16"}>VGG16</option>
                  <option value={"Efficient Net"}>Efficient Net</option>
                  <option value={"ResNet50"}>ResNet50</option>
                  <option value={"MixedNet"}>MixedNet</option>
                </Select>
                <input
                  className="btn"
                  type="submit"
                  onClick={predict_image}
                  value="Apply"
                />
              </div>
            </form>
          </div>
          <Text fontSize="small">
            Note: MixedNet has provided a promising performance and Accuracies
            among the models
          </Text>
        </div>
      </div>
      <Report
        disease_name={disease_name}
        set_disease_name={set_disease_name}
        image={file}
      />
    </div>
  );
};

export default UploadImg;

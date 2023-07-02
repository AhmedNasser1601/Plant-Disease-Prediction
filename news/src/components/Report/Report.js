import React from "react";
import "./Report.css";
import { Heading } from "@chakra-ui/react";
import { FaTimesCircle } from "react-icons/fa";

let links = {
  "Tomato___Late_blight":
    "https://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/",
  "Orange__Haunglongbing(Citrus_greening)":
    "https://en.wikipedia.org/wiki/Citrus_greening_disease",
  "Squash___Powdery_mildew":
    "https://www.ruralsprout.com/treat-powdery-mildew/",
  "Corn_(maize)___Northern_Leaf_Blight":
    "https://en.wikipedia.org/wiki/Northern_corn_leaf_blight",
  "Tomato___Early_blight":
    "https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/",
  "Tomato___Septoria_leaf_spot":
    "https://www.thespruce.com/identifying-and-controlling-septoria-leaf-spot-of-tomato-1402974",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
    "https://en.wikipedia.org/wiki/Corn_grey_leaf_spot",
  "Strawberry___Leaf_scorch":
    "https://www.gardeningknowhow.com/edible/fruits/strawberry/strawberries-with-leaf-scorch.htm",
  "Apple___Apple_scab": "https://en.wikipedia.org/wiki/Apple_scab",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus",
  "Tomato___Bacterial_spot":
    "https://ipm.ucanr.edu/agriculture/tomato/bacterial-spot/",
  "Apple___Black_rot":
    "https://extension.umn.edu/plant-diseases/black-rot-apple",
  "Cherry_(including_sour)___Powdery_mildew":
    "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
  "Peach___Bacterial_spot":
    "https://www.canr.msu.edu/ipm/agriculture/fruit/bacterial_spot_of_peach_and_plum",
  "Apple___Cedar_apple_rust":
    "https://extension.umn.edu/plant-diseases/cedar-apple-rust",
  "Tomato___Target_Spot":
    "https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/tomato_target_spot_163.htm",
  "Grape__Leaf_blight(Isariopsis_Leaf_Spot)":
    "https://poe.com/s/3QKKdYrvwwDHKPVKPHJV",
  "Potato___Late_blight":
    "https://extension.umn.edu/disease-management/late-blight",
  "Tomato___Tomato_mosaic_virus":
    "https://en.wikipedia.org/wiki/Tomato_mosaic_virus",
  "Grape___Black_rot":
    "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)",
  "Potato___Early_blight":
    "https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/",
  "Corn_(maize)___Common_rust_":
    "https://extension.umn.edu/corn-pest-management/common-rust-corn",
  "Grape__Esca(Black_Measles)":
    "https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/",
  "Tomato___Leaf_Mold":
    "https://extension.umn.edu/disease-management/tomato-leaf-mold",
  "Tomato___Spider_mites Two-spotted_spider_mite":
    "https://extension.psu.edu/two-spotted-spider-mite-on-vegetables",
  "Pepper,bell__Bacterial_spot":
    "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper",
};

export default function Report(props) {
  const isHealthy = props.disease_name.toLowerCase().includes("healthy");
  const closeReport = () => {
    document.querySelector(".disease_report").classList.add("hidden");
    if (props.set_disease_name) {
      props.set_disease_name("");
    }
  };
  return (
    <div className="disease_report hidden">
      <div className="report_content">
        <FaTimesCircle className="closeReportButton" onClick={closeReport} />
        <div className="report_header">
          <Heading as="h2" size="2xl">
            {props.disease_name && props.disease_name}
          </Heading>
        </div>
        <div className="disease_image">
          <img
            src={props.image && URL.createObjectURL(props.image)}
            alt="plant disease"
          />
        </div>
        <div className="show_more">
          <button
            className="checkBtn"
            onClick={() => {
              if (!isHealthy) window.open(links[props.disease_name], "_blank");
            }}
          >
            {isHealthy ? "This plant is healthy" : "Show Informations"}
          </button>
        </div>
      </div>
    </div>
  );
}

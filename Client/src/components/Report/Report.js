import React from "react";
import "./Report.css";
import { Heading } from "@chakra-ui/react";
import { FaTimesCircle } from "react-icons/fa";

export default function Report(props) {
  const isHealthy = props.disease_name.toLowerCase().includes("healthy");
  const closeReport = () => {
    document.querySelector(".disease_report").classList.add("hidden");
    if (props.set_disease_name) {
      props.set_disease_name("");
    }
  };

  const openFirstSearchResult = async (diseaseName) => {
    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(
      diseaseName
    )}`;
    const response = await fetch(searchUrl);
    const text = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, "text/html");
    const searchResults = doc.querySelectorAll(".g");

    if (searchResults.length > 0) {
      const firstResult = searchResults[0].querySelector("a");
      const url = firstResult.getAttribute("href");
      if (url) {
        const decodedUrl = decodeURIComponent(url);
        window.open(decodedUrl, "_blank");
      }
    }
  };

  return (
    <div className="disease_report hidden">
      <div className="report_content">
        <FaTimesCircle className="closeReportButton" onClick={closeReport} />
        <div className="report_header">
          <Heading as="h2" size="2xl">
            Plant Name:{" "}
            {props.disease_name && props.disease_name.split("__")[0]}
          </Heading>
          <Heading as="h2" size="2xl" marginTop="10">
            Disease Name:{" "}
            {props.disease_name &&
              props.disease_name.split("__")[1].replace(/_/g, " ")}
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
              openFirstSearchResult(props.disease_name);
            }}
          >
            {isHealthy ? "This plant is healthy" : "Show Informations"}
          </button>
        </div>
      </div>
    </div>
  );
}

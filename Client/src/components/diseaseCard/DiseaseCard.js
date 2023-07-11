import React from "react";
import {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Image,
  Heading,
  Text,
  Divider,
  Button,
} from "@chakra-ui/react";
import { Stack } from "react-bootstrap";

import Report from "../Report/Report";

const DiseaseCard = ({ data }) => {
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
    <>
      <Card className="diseaseCard">
        <CardBody>
          <Image src={data.img} alt="disease Image" className="diseaseImg" />
          <Stack>
            <Heading size="md">{data.diseaseName}</Heading>
          </Stack>
        </CardBody>
        <Divider />
        <CardFooter>
          <Button
            className="showMore"
            onClick={() => openFirstSearchResult(data.diseaseName)}
          >
            Show More
          </Button>
        </CardFooter>
      </Card>
    </>
  );
};

export default DiseaseCard;

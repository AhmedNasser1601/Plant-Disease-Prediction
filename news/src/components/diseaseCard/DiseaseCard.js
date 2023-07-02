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

const DiseaseCard = ({ data, set_disease_name }) => {
  const showMore = () => {
    document.querySelector(".disease_report").classList.remove("hidden");
    set_disease_name(data.diseaseName);
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
          <Button className="showMore" onClick={showMore}>
            Show More
          </Button>
        </CardFooter>
      </Card>
    </>
  );
};

export default DiseaseCard;

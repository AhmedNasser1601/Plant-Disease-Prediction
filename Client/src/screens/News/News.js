import Slider from "../../components/slider/Slider";
import SideBar from "../../components/SideBar/SideBar";
import diseases from "../../diseases.js";
import DiseaseCard from "../../components/diseaseCard/DiseaseCard.js";
import { Box, Heading, SimpleGrid, Text } from "@chakra-ui/react";
import "bootstrap/dist/css/bootstrap.min.css";
import "./News.css";
import { useState } from "react";
import { Container } from "react-bootstrap";
import Report from "../../components/Report/Report";

const News = () => {
  const [plantName, setPlantName] = useState("");

  function handleSearch(searchValue) {
    setPlantName(searchValue);
  }

  return (
    <Container>
      <Slider />
      <div className="newsMain">
        <SimpleGrid
          columns={{ sm: 1, md: 2, lg: 3 }}
          spacing="50px"
          className="newsContent"
        >
          {diseases.map((disease) => {
            if (disease.plantName === plantName || plantName === "") {
              return (
                <Box className="newsBox">
                  <DiseaseCard
                    data={disease}
                    key={disease.key}
                  />
                </Box>
              );
            }
          })}
        </SimpleGrid>
        <SideBar search={handleSearch} />
      </div>
    </Container>
  );
};

export default News;

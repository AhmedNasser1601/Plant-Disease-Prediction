import "./Home.css";
import HomeHeader from "../../components/homeHeader/homeHeader";
import { Container } from "@chakra-ui/react";
import AboutUs from "../../components/AboutUs/AboutUs";
import Goals from "../../components/Goals/Goals";

const Home = () => {
  return (
    <div id="Home">
      <HomeHeader />
      <Container maxW="7xl">
        <div className="homeContent">
          <AboutUs />
          <Goals />
        </div>
      </Container>
    </div>
  );
};

export default Home;

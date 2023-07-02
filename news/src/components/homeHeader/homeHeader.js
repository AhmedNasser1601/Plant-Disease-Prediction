import { Button } from "@chakra-ui/react";
import { Link } from "react-router-dom";

const HomeHeader = () => {
  return (
    <div className="homeHeader mb">
      <div className="headerContent">
        <h1>Save your plants!</h1>
        <p>
          Save your crops health by discovering the diseases early by shooting
          them and get more informations about the disease to be able to deal
          with it.
        </p>
        <Button colorScheme="green" marginTop="20px">
          <Link to="/check">Go To Check</Link>
        </Button>
      </div>
    </div>
  );
};

export default HomeHeader;

import { Box, Heading, SimpleGrid, Text } from "@chakra-ui/react";

const Goals = () => {
  return (
    <div className="goals mb">
      <h2 className="goalsHead headers">Our Goals</h2>
      <SimpleGrid columns={{ sm: 1, md: 2, lg: 2 }} spacing="50px">
        <Box textAlign="left" className="goalBox">
          <div className="goalContent">
            <Heading
              as="h3"
              size="lg"
              fontWeight="normal"
              className="goalBoxHead"
            >
              Discovering the plant diseases ASAP
            </Heading>
            <Text fontSize="md" marginTop="5">
              Environmental activism has led to an Increase in the awareness
              among people about various issues human activities have had on the
              environment.
            </Text>
          </div>
        </Box>
        <Box textAlign="left" className="goalBox">
          <div className="goalContent">
            <Heading
              as="h3"
              size="lg"
              fontWeight="normal"
              className="goalBoxHead"
            >
              Facilitate disease detection process
            </Heading>
            <Text fontSize="md" marginTop="5">
              Environmental activism has led to an Increase in the awareness
              among people about various issues human activities have had on the
              environment.
            </Text>
          </div>
        </Box>
        <Box textAlign="left" className="goalBox">
          <div className="goalContent">
            <Heading
              as="h3"
              size="lg"
              fontWeight="normal"
              className="goalBoxHead"
            >
              Spreading awareness about plants
            </Heading>
            <Text fontSize="md" marginTop="5">
              Environmental activism has led to an Increase in the awareness
              among people about various issues human activities have had on the
              environment.
            </Text>
          </div>
        </Box>
        <Box textAlign="left" className="goalBox">
          <div className="goalContent">
            <Heading
              as="h3"
              size="lg"
              fontWeight="normal"
              className="goalBoxHead"
            >
              Optimizing the food ecosystem
            </Heading>
            <Text fontSize="md" marginTop="5">
              Environmental activism has led to an Increase in the awareness
              among people about various issues human activities have had on the
              environment.
            </Text>
          </div>
        </Box>
      </SimpleGrid>
    </div>
  );
};

export default Goals;

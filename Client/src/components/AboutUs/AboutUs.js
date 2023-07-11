import React from "react";
import { Box, SimpleGrid, Image, Text } from "@chakra-ui/react";

export default function AboutUs() {
  return (
    <div className="about mb">
      <h2 className="headers">About Us</h2>
      <SimpleGrid minChildWidth="500px" columns={[1, null, 2]} spacing="50px">
        <Box className="about_photo" height="500px">
          <Image
            src="https://i.pinimg.com/564x/39/e6/d4/39e6d4c4f0572317e8a6c123e5d21015.jpg"
            height="100%"
            width="100%"
          />
        </Box>
        <Box className="about_content" height="200px">
          <Text fontSize="2xl" textAlign="left" lineHeight="10">
            We are a Software Engineering students, Studying at Ain Shams
            University. We are at the 4th year and this is our graduation
            project. We are looking for to help you to save your crops health by
            discovering diseases because we found that there are too much crops
            that are dying cause of diseases, So we are trying to provide some
            solutions to discover them early.
          </Text>
        </Box>
      </SimpleGrid>
    </div>
  );
}

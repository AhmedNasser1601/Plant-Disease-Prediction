import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import News from "./screens/News/News.js";
import UploadImg from "./screens/upload_img/upload_img.js";
import { ChakraProvider } from "@chakra-ui/react";
import Home from "./screens/Home/Home.js";
import NavBar from "./components/Navbar/Navbar.js";
import Footer from "./components/footer/Footer.js";

function App() {
  return (
    <ChakraProvider>
      <BrowserRouter>
        <NavBar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/news" element={<News />} />
          <Route path="/check" element={<UploadImg />} />
          
        </Routes>
        <Footer />
      </BrowserRouter>
    </ChakraProvider>
  );
}

export default App;

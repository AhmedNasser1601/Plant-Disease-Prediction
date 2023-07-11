import React from "react";
import "./Slider.css";
import { Swiper, SwiperSlide } from "swiper/react";
import { FreeMode, Navigation } from "swiper";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/free-mode";
import "bootstrap/dist/css/bootstrap.min.css";
import diseases from "../../images.js";
import NewHead from "../NewHead/NewHead";

const Slider = () => {
  return (
    
    <div className="containe py-4 justify-content-center">
      <Swiper
        grabCursor={true}
        navigation={true}
        modules={[Navigation]}
        className="mySwiper"
        slidesPerView={1}
        spaceBetween={30}
      >
        {diseases.map((disease) => {
          return (
            <SwiperSlide key={disease} className="imgSlide">
              <NewHead data={disease} />
            </SwiperSlide>
          );
        })}
      </Swiper>
    </div>
  );
};

export default Slider;

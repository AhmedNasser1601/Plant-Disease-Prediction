import React from "react";
import "./Footer_update.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAddressBook, faVoicemail } from "@fortawesome/free-solid-svg-icons";
import {
  faYoutube,
  faFacebook,
  faWhatsapp,
  faYahoo,
} from "@fortawesome/free-brands-svg-icons";

const Footer = () => {
  const adr = <FontAwesomeIcon icon={faAddressBook} className="social_icons" />;
  const yah = <FontAwesomeIcon icon={faYahoo} className="social_icons" />;
  const call = <FontAwesomeIcon icon={faVoicemail} className="social_icons" />;
  const youtube = <FontAwesomeIcon icon={faYoutube} className="social_icons" />;
  const Facebook = (
    <FontAwesomeIcon icon={faFacebook} className="social_icons" />
  );
  const Whatsapp = (
    <FontAwesomeIcon icon={faWhatsapp} className="social_icons" />
  );

  return (
    <>
      <div className="main_footer">
        <div className="container">
          <div className="row">
            {/* Column1 */}
            <div className="col">
              <h4 className="footer_headers">About us</h4>
              
              <ui className="list-unstyled">
                <li>Our Website Target to Detect and Predict </li>
                <li>the Diseases of the inserted Plant Image</li>
                <li></li>
              </ui>
            </div>

            {/* Column3 */}
            <div className="col">
              <h4 className="footer_headers">Follow Us</h4>
              <ui className="list-unstyled social_icons_container">
                <li> <a href="#"> {Facebook} </a></li>
                <li> <a href="#"> {youtube}  </a></li>
                <li> <a href="#"> {Whatsapp} </a> </li>
              </ui>
            </div>
            {/* Column2 */}
            <div className="col">
              <h4 className="footer_headers">Contact Us</h4>
              <ui className="list-unstyled">
                <li>{adr} Address : 13 share3 Abdelhamed fe el Gizza</li>
                <li>{yah} E-mail : ay7aga@yahoo.com</li>
                <li>{call} Mobile : (+20)775-468-1234 - office</li>
              </ui>
            </div>
          </div>
          <hr />
          <div className="row">
            <p className="col-sm">
              &copy;{new Date().getFullYear()} Dr.Plant | All rights reserved |
              Terms Of Service | Privacy
            </p>
          </div>
        </div>
      </div>
    </>
  );
};
export default Footer;

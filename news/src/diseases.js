import apple_scab from "./images/diseases/Apple-scab-on-fruit-grabowski.jpg";
import apple_black_rot from "./images/diseases/blackRotLesionsApple.jpg";
import strawberry_leaf_scorch from "./images/diseases/Strawberry___Leaf_scorch.jpg";
import peach_bacterial_spot from "./images/diseases/peach-bacterial-spot.jpg";
import grape_leaf_blight from "./images/diseases/grape-leaf-blight.jpg";
import grape_black_rot from "./images/diseases/grape-black-rot.jpg";

const diseaseData = [
  {
    key: 1,
    img: apple_scab,
    plantName: "apple",
    diseaseName: "Apple Scab",
    facts: `Apple scab is the most common disease of apple and crabapple trees in Minnesota.,
    Scab is caused by a fungus that infects both leaves and fruit.,
    Scabby fruit are often unfit for eating.,
    Infected leaves have olive green to brown spots.,
    Leaves with many leaf spots turn yellow and fall off early.,
    Leaf loss weakens the tree when it occurs many years in a row.,
    Planting disease resistant varieties is the best way to manage scab.,
    Fungicides can be used to manage apple scab. Proper timing of sprays is needed for fungicides to control disease.`,
    identify: `Leaf spots are round, olive-green in color and up to ½-inch across., 
    Spots are velvet-like with fringed borders.,
    As they age, leaf spots turn dark brown to black, get bigger and grow together.,
    Leaf spots often form along the leaf veins.,
    Leaves with many leaf spots turn yellow and drop by mid-summer.,
    Infected fruit have olive-green spots that turn brown and corky with time.,
    Fruit that are infected when very young become deformed and cracked as the fruit grows.`,
  },
  {
    key: 2,
    img: apple_black_rot,
    diseaseName: "Apple Black Rot",
    plantName: "apple",
  },
  {
    key: 3,
    img: strawberry_leaf_scorch,
    diseaseName: "Strawberry Leaf Scorch",
    plantName: "strawberry",
  },
  {
    key: 4,
    img: peach_bacterial_spot,
    diseaseName: "Peach Bacterial Spot",
    plantName: "peach",
  },
  {
    key: 5,
    img: grape_leaf_blight,
    diseaseName: "Grape Leaf Blight",
    plantName: "grape",
  },
  {
    key: 6,
    img: grape_black_rot,
    diseaseName: "Grape Black Rot",
    plantName: "grape",
  },
];

export default diseaseData;

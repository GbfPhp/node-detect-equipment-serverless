// Load 'opencv.js' assigning the value to the global variable 'cv'
import { getOpenCv } from "./src/opencv.mjs";

const { cv } = await getOpenCv();

const orb = new cv.ORB(200)

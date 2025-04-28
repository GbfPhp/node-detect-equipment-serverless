const express = require('express');
const fs = require('fs/promises'); // Import fs promises API
const path = require('path'); // Import path module
const cv = require("@techstark/opencv-js");
const dotenv = require('dotenv');
const cors = require('cors');

let OPENCV_INITIALIZED = false;

cv.onRuntimeInitialized = () => {
  console.log("OpenCV.js is ready!");
  OPENCV_INITIALIZED = true;
};

async function openCvLoaded() {
  return new Promise((resolve) => {
    if (OPENCV_INITIALIZED) {
      resolve();
    } else {
      for (let i = 0; i < 10; i++) {
        setTimeout(() => {
          if (OPENCV_INITIALIZED) {
            resolve();
          }
        }, 1000);
      }
    }
  });
}

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
// Allow all origins for preflight requests
// app.options("*", cors());

const CACHE_DIR = process.env.CACHE_DIR;
// Define cache directory relative to the server file
if (!CACHE_DIR) {
  throw new Error("CACHE_DIR is not set");
}
const cacheDir = path.join(__dirname, CACHE_DIR);

// Template Features object - will be populated lazily from cache
const templateFeatures = {
  "summon/party_main": {},
  "summon/party_sub": {},
  "weapon/main": {},
  "weapon/normal": {},
  "chara": {},
};

// --- Function to Load Features from Cache ---
async function loadFeaturesFromCache(equipmentType) {
  // if (!cv) throw new Error("OpenCV not initialized before loading cache");

  // LAZY LOADING CHECK: If features for this type are already loaded, skip.
  // Basic check: assumes an empty object means not loaded.
  if (templateFeatures[equipmentType] && Object.keys(templateFeatures[equipmentType]).length > 0) {
    console.log(`Features for ${equipmentType} already loaded.`);
    return;
  }

  // Ensure the object exists before trying to load into it
  if (!templateFeatures[equipmentType]) {
      templateFeatures[equipmentType] = {};
  }

  const cacheFileName = `${equipmentType}_features.json`;
  const cacheFilePath = path.join(cacheDir, cacheFileName);

  console.log(`LAZY LOADING cache for ${equipmentType} from: ${cacheFilePath}`);

  try {
    await fs.access(cacheFilePath);
    const cacheContent = await fs.readFile(cacheFilePath, 'utf-8');
    const cachedData = JSON.parse(cacheContent);

    if (!cachedData.template_names || !cachedData.descriptors_list || cachedData.template_names.length !== cachedData.descriptors_list.length) {
      console.error(`Invalid cache file format for ${equipmentType}.`);
      // Mark as failed? Or just leave empty? Leaving empty for now.
      return;
    }

    const templateNames = cachedData.template_names;
    const serializedDescriptors = cachedData.descriptors_list;
    let loadedCount = 0;

    console.log(`Loading ${templateNames.length} features for ${equipmentType} from cache...`);

    for (let i = 0; i < templateNames.length; i++) {
      const name = templateNames[i];
      const desStr = serializedDescriptors[i];
      let desMat = null;
      if (desStr) {
        try {
          desMat = deserializeDescriptors(desStr);
          if (desMat && !desMat.empty()) {
            templateFeatures[equipmentType][name] = desMat;
            loadedCount++;
          } else {
            console.warn(`Deserialized descriptor for ${name} (${equipmentType}) is null or empty.`);
            if (desMat && desMat.delete) desMat.delete();
          }
        } catch (deserializeError) {
          console.error(`Error deserializing descriptor for ${name} (${equipmentType}):`, deserializeError);
          if (desMat && desMat.delete) desMat.delete();
        }
      } else {
        console.warn(`Null or empty descriptor string found for ${name} (${equipmentType}) in cache.`);
      }
    }
    console.log(`Successfully loaded ${loadedCount} features for ${equipmentType}.`);

  } catch (error) {
    if (error.code === 'ENOENT') {
      console.warn(`Cache file not found for ${equipmentType}: ${cacheFilePath}.`);
    } else {
      console.error(`Failed to load cache for ${equipmentType}:`, error);
    }
    // Ensure the type object exists even if loading failed, but it will be empty
    if (!templateFeatures[equipmentType]) {
        templateFeatures[equipmentType] = {};
    }
  }
}

// --- Deserialization Logic ---
// once initialized, don't delete the mat
function deserializeDescriptors(base64Content) {
  try {
    const buffer = Buffer.from(base64Content, 'base64');
    const descriptorSize = 32; // ORB descriptors are 32 bytes
    if (buffer.length % descriptorSize !== 0) {
      throw new Error(`Invalid descriptor buffer length: ${buffer.length}. Must be a multiple of ${descriptorSize}.`);
    }
    const numDescriptors = buffer.length / descriptorSize;
    if (numDescriptors === 0) {
        // Return an empty Mat if buffer is empty
        return new cv.Mat();
    }
    // Create a Mat from the buffer (rows = numDescriptors, cols = descriptorSize, type = CV_8U)
    // Need to wrap the buffer in a Uint8Array for matFromArray
    const data = new Uint8Array(buffer);
    const mat = cv.matFromArray(numDescriptors, descriptorSize, cv.CV_8U, data);
    return mat;
  } catch (error) {
    console.error("Error deserializing descriptors:", error);
    throw new Error(`Failed to deserialize base64 content: ${error.message}`);
  }
}

// --- Matching Logic (now async for lazy loading) ---
/**
 * Match equipment descriptors against templates
 * @param {cv.Mat} queryDescriptors - OpenCV Mat containing query descriptors
 * @param {string} equipmentType - Type of equipment to match against (e.g. "weapon/main")
 * @param {number} threshold - Matching threshold value (default: 10)
 * @param {number} top_n - Number of top matches to return (default: 3)
 * @returns {Promise<Array>} Array of matched equipment with scores
 */
async function matchEquipment(queryDescriptors, equipmentType, threshold = 10, top_n = 3, earlyReturn = false) {
  const bfMatcher = new cv.BFMatcher();

  // --- LAZY LOADING TRIGGER ---
  // Check if features for this type exist and are loaded
  if (!templateFeatures[equipmentType] || Object.keys(templateFeatures[equipmentType]).length === 0) {
    console.log(`Features for ${equipmentType} not loaded. Triggering lazy load...`);
    await loadFeaturesFromCache(equipmentType); // Wait for loading to complete

    // Check again after loading attempt
    if (!templateFeatures[equipmentType] || Object.keys(templateFeatures[equipmentType]).length === 0) {
        console.warn(`Features for ${equipmentType} could not be loaded or are empty after attempt. Skipping match.`);
        return []; // Cannot match if features didn't load
    }
  }
  // --- End Lazy Loading Trigger ---

  if (queryDescriptors.empty()) {
    console.log("Query descriptors are empty, skipping matching.");
    return [];
  }

  const typeTemplates = templateFeatures[equipmentType];
  const matches = [];

  // Ensure queryDescriptors is CV_8U for knnMatch
  let queryDescriptorsToUse = queryDescriptors;
  let convertedMat = null; // Keep track if we converted
  if (queryDescriptors.type() !== cv.CV_8U) {
      console.warn("Query descriptors are not CV_8U. Attempting conversion.");
      try {
          convertedMat = new cv.Mat();
          queryDescriptors.convertTo(convertedMat, cv.CV_8U);
          queryDescriptorsToUse = convertedMat; // Use the converted mat for matching
      } catch(conversionError) {
          console.error("Failed to convert query descriptors to CV_8U:", conversionError);
          if (convertedMat && convertedMat.delete) convertedMat.delete(); // Clean up
          return []; // Cannot proceed
      }
  }

  for (const templateName in typeTemplates) {
    const templateDes = typeTemplates[templateName];
    if (!templateDes || templateDes.empty()) {
        console.warn(`Cached template descriptors for "${templateName}" are empty or invalid.`);
        continue;
    }
    if (templateDes.type() !== cv.CV_8U) {
        console.warn(`Cached template descriptors for "${templateName}" are not CV_8U. Skipping.`);
        continue;
    }

    let knnMatches = null;
    try {
        knnMatches = new cv.DMatchVectorVector();
        bfMatcher.knnMatch(queryDescriptorsToUse, templateDes, knnMatches, 2); // Use queryDescriptorsToUse

        let goodMatchesCount = 0;
        const ratioThresh = 0.75;
        for (let i = 0; i < knnMatches.size(); ++i) {
            const matchVec = knnMatches.get(i);
            if (matchVec.size() >= 2) {
                const m = matchVec.get(0);
                const n = matchVec.get(1);
                if (m.distance < ratioThresh * n.distance) {
                    goodMatchesCount++;
                }
            }
        }

      if (goodMatchesCount >= threshold) {
        matches.push({ id: templateName, confidence: goodMatchesCount });
        if (earlyReturn) {
          break
        }
      }
    } catch (error) {
        console.error(`Error matching against template "${templateName}"`);
    } finally {
        if (knnMatches && knnMatches.delete) {
            try { knnMatches.delete(); } catch (e) { console.error("Error deleting knnMatches:", e); }
        }
    }
  }

  // Clean up the converted matrix if one was created
  if (convertedMat && convertedMat.delete) {
      try { convertedMat.delete(); } catch (e) { console.error("Error deleting convertedMat:", e); }
  }

  matches.sort((a, b) => b.confidence - a.confidence);
  return matches.slice(0, top_n);
}


// --- Main Detection Handler (now handles async matchEquipment) ---
async function handleDetectRequest(req, res, equipmentType) {
  console.log("loading opencv");
  await openCvLoaded();
  console.log("handleDetectRequest", equipmentType);

  try {
    const reqBody = req.body;
    if (!reqBody || !Array.isArray(reqBody.contents)) {
      return res.status(400).json({ error: `Invalid request body. Expecting { contents: [base64string] } ${reqBody}` });
    }

    const earlyReturn = reqBody.earlyReturn === "true";

    const results = [];
    if (reqBody.contents.length === 0) {
      return res.json(results);
    }

    let queryDes = null;

    // Process descriptors sequentially because matchEquipment might trigger loading
    for (const content of reqBody.contents) {
      try {
        queryDes = deserializeDescriptors(content);
        // Use await because matchEquipment is now async
        const topMatches = await matchEquipment(queryDes, equipmentType, 10, 3, earlyReturn);
        results.push(topMatches);
      } catch (error) {
          console.error(`Error processing descriptor content for type ${equipmentType}:`, error);
          results.push({ error: error.message });
      } finally {
          if (queryDes && queryDes.delete) {
              try { queryDes.delete(); } catch (e) { console.error("Error deleting queryDes:", e); }
          }
      }
    }
    res.json(results);
  } catch (e) {
    console.error(`Error in /v1/detect/${equipmentType.replace('/', '-')}:`, e);
    res.status(500).json({ error: e.message || "Internal server error" });
  }
}


// --- API Endpoints ---
app.post("/v1/detect/summon/party_main", (req, res) => handleDetectRequest(req, res, "summon/party_main"));
app.post("/v1/detect/summon/party_sub", (req, res) => handleDetectRequest(req, res, "summon/party_sub"));
app.post("/v1/detect/weapon/main", (req, res) => handleDetectRequest(req, res, "weapon/main"));
app.post("/v1/detect/weapon/normal", (req, res) => handleDetectRequest(req, res, "weapon/normal"));
app.post("/v1/detect/priority/weapon/main", (req, res) => handleDetectRequest(req, res, "priority/weapon/main"));
app.post("/v1/detect/priority/weapon/normal", (req, res) => handleDetectRequest(req, res, "priority/weapon/normal"));
app.post("/v1/detect/chara", (req, res) => handleDetectRequest(req, res, "chara"));


// --- Basic health check endpoint ---
app.get('/', (req, res) => {
  res.send('Server is running');
});

module.exports = app;

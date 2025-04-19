const dotenv = require("dotenv");
dotenv.config({ path: ".env" });
const app = require("./app.js");

const port = process.env.PORT || 2580;


app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

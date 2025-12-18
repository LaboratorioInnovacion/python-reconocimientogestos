const express = require("express");
const { parseCamlyticsXML } = require("./xmlParser");
const { processDetection, getCount } = require("./faceTracker");

const app = express();
app.use(express.text({ type: "*/*" }));

app.post("/camlytics/event", (req, res) => {
  const detection = parseCamlyticsXML(req.body);

  if (!detection) {
    return res.send("No detection");
  }

  const track = processDetection(detection);

  console.log("Track:", track.id, "Total:", getCount());

  res.json({
    trackId: track.id,
    totalPeople: getCount(),
  });
});

app.listen(3000, () =>
  console.log("Face Counter IOU listo en puerto 3000")
);

// const express = require("express")
// const cors = require("cors")F

// const { parseCamlyticsXML } = require("./xmlParser")
// const { processFace, getCount } = require("./faceTracker")

// const app = express()
// app.use(cors())
// app.use(express.text({ type: "*/*" }))

// app.post("/camlytics/event", (req, res) => {
//   try {
//     const face = parseCamlyticsXML(req.body)

//     if (!face) {
//       return res.status(200).send("No face detected")
//     }

//     const person = processFace(face)

//     console.log("Persona detectada:", person.id)
//     console.log("Conteo actual:", getCount())

//     res.json({
//       personId: person.id,
//       totalPeople: getCount()
//     })
//   } catch (err) {
//     console.error(err)
//     res.status(500).send("Error")
//   }
// })

// app.listen(3000, () => {
//   console.log("Camlytics Face Counter corriendo en puerto 3000")
// })
